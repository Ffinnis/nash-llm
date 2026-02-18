import math
import torch
import torch.nn as nn
from nash_llm.config import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class _ExpertMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class MoEFeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.capacity_factor = config.moe_capacity_factor
        self.router_jitter = config.moe_router_jitter

        self.router = nn.Linear(config.d_model, self.num_experts)
        self.experts = nn.ModuleList(
            [_ExpertMLP(config.d_model, config.moe_expert_d_ff, config.dropout) for _ in range(self.num_experts)]
        )

        self.last_aux_loss: torch.Tensor | None = None
        self.last_z_loss: torch.Tensor | None = None
        self.last_moe_metrics: dict[str, torch.Tensor] = {
            "aux_loss": torch.tensor(0.0),
            "z_loss": torch.tensor(0.0),
            "dropped_frac": torch.tensor(0.0),
            "expert_entropy": torch.tensor(0.0),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)

        router_in = x_flat
        if self.training and self.router_jitter > 0:
            router_in = router_in + torch.randn_like(router_in) * self.router_jitter

        router_logits = self.router(router_in)
        router_probs = torch.softmax(router_logits, dim=-1)

        topk_probs, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        total_tokens = x_flat.size(0)
        total_assignments = max(total_tokens * self.top_k, 1)
        capacity = max(1, int(math.ceil(self.capacity_factor * total_assignments / self.num_experts)))

        out_flat = torch.zeros_like(x_flat)
        expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)

        # Flatten routing assignments once and group by expert with a single sort.
        assign_tokens = (
            torch.arange(total_tokens, device=x.device)
            .unsqueeze(1)
            .expand(total_tokens, self.top_k)
            .reshape(-1)
        )
        assign_experts = topk_idx.reshape(-1)
        assign_weights = topk_probs.reshape(-1)

        order = torch.argsort(assign_experts)
        sorted_experts = assign_experts[order]
        sorted_tokens = assign_tokens[order]
        sorted_weights = assign_weights[order]

        counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        starts = torch.cumsum(counts, dim=0) - counts
        positions = torch.arange(sorted_experts.numel(), device=x.device)
        rank_within_expert = positions - starts[sorted_experts]
        keep_mask = rank_within_expert < capacity

        kept_tokens = sorted_tokens[keep_mask]
        kept_experts = sorted_experts[keep_mask]
        kept_weights = sorted_weights[keep_mask]

        kept_counts = torch.bincount(kept_experts, minlength=self.num_experts)
        expert_counts = kept_counts.to(torch.float32)
        dropped_assignments = (counts - kept_counts).sum().to(torch.float32)

        for expert_id in range(self.num_experts):
            expert_mask = kept_experts == expert_id
            expert_tokens = kept_tokens[expert_mask]
            expert_weights = kept_weights[expert_mask]
            expert_out = self.experts[expert_id](x_flat[expert_tokens])
            out_flat.index_add_(0, expert_tokens, expert_out * expert_weights.unsqueeze(-1))

        importance = router_probs.mean(dim=0)
        total_kept = max(kept_assignments, 1)
        load = expert_counts / float(total_kept)
        aux_loss = self.num_experts * torch.sum(importance * load)
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1).square())

        dropped_frac = dropped_assignments / float(total_assignments)
        load_dist = expert_counts / expert_counts.sum().clamp_min(1.0)
        entropy = -torch.sum(load_dist * torch.log(load_dist.clamp_min(1e-9)))
        if self.num_experts > 1:
            expert_entropy = entropy / math.log(self.num_experts)
        else:
            expert_entropy = entropy.new_zeros(())

        self.last_aux_loss = aux_loss
        self.last_z_loss = z_loss
        self.last_moe_metrics = {
            "aux_loss": aux_loss.detach(),
            "z_loss": z_loss.detach(),
            "dropped_frac": dropped_frac.detach(),
            "expert_entropy": expert_entropy.detach(),
        }
        return out_flat.view(bsz, seq_len, dim)
