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
        self.last_moe_metrics: dict[str, float] = {
            "aux_loss": 0.0,
            "z_loss": 0.0,
            "dropped_frac": 0.0,
            "expert_entropy": 0.0,
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
        kept_assignments = 0
        dropped_assignments = 0

        for expert_id in range(self.num_experts):
            token_idx, route_idx = torch.where(topk_idx == expert_id)
            n_assignments = token_idx.numel()
            if n_assignments == 0:
                continue

            n_kept = min(n_assignments, capacity)
            kept_tokens = token_idx[:n_kept]
            kept_routes = route_idx[:n_kept]
            weights = topk_probs[kept_tokens, kept_routes]

            expert_out = self.experts[expert_id](x_flat[kept_tokens])
            out_flat.index_add_(0, kept_tokens, expert_out * weights.unsqueeze(-1))

            expert_counts[expert_id] = float(n_kept)
            kept_assignments += n_kept
            dropped_assignments += n_assignments - n_kept

        importance = router_probs.mean(dim=0)
        total_kept = max(kept_assignments, 1)
        load = expert_counts / float(total_kept)
        aux_loss = self.num_experts * torch.sum(importance * load)
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1).square())

        dropped_frac = float(dropped_assignments) / float(total_assignments)
        load_dist = expert_counts / expert_counts.sum().clamp_min(1.0)
        entropy = -torch.sum(load_dist * torch.log(load_dist.clamp_min(1e-9)))
        expert_entropy = float((entropy / math.log(self.num_experts)).item()) if self.num_experts > 1 else 0.0

        self.last_aux_loss = aux_loss
        self.last_z_loss = z_loss
        self.last_moe_metrics = {
            "aux_loss": float(aux_loss.detach().item()),
            "z_loss": float(z_loss.detach().item()),
            "dropped_frac": dropped_frac,
            "expert_entropy": expert_entropy,
        }
        return out_flat.view(bsz, seq_len, dim)
