import torch
import torch.nn as nn
from nash_llm.config import ModelConfig
from nash_llm.model.attention import MultiHeadAttention
from nash_llm.model.layers import FeedForward, MoEFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)

        self.use_moe = (
            config.moe_enabled
            and layer_idx >= config.moe_start_layer
            and (layer_idx - config.moe_start_layer) % config.moe_layer_stride == 0
        )
        self.ff: nn.Module
        if self.use_moe:
            self.ff = MoEFeedForward(config)
        else:
            self.ff = FeedForward(config)

        self.last_moe_metrics: dict[str, float] | None = None
        self.last_aux_loss: torch.Tensor | None = None
        self.last_z_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        ff_out = self.ff(self.ln2(x))
        if self.use_moe:
            assert isinstance(self.ff, MoEFeedForward)
            self.last_moe_metrics = self.ff.last_moe_metrics
            self.last_aux_loss = self.ff.last_aux_loss
            self.last_z_loss = self.ff.last_z_loss
        else:
            self.last_moe_metrics = None
            self.last_aux_loss = None
            self.last_z_loss = None
        x = x + ff_out
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.last_moe_metrics: dict[str, float] = {
            "aux_loss": 0.0,
            "z_loss": 0.0,
            "dropped_frac": 0.0,
            "expert_entropy": 0.0,
        }
        self._last_moe_aux_loss: torch.Tensor | None = None
        self._last_moe_z_loss: torch.Tensor | None = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        tok_emb = self.token_emb(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        block_aux_losses: list[torch.Tensor] = []
        block_z_losses: list[torch.Tensor] = []
        block_metrics: list[dict[str, float]] = []
        for block in self.blocks:
            x = block(x)
            if block.last_aux_loss is not None and block.last_z_loss is not None and block.last_moe_metrics is not None:
                block_aux_losses.append(block.last_aux_loss)
                block_z_losses.append(block.last_z_loss)
                block_metrics.append(block.last_moe_metrics)

        if block_aux_losses:
            self._last_moe_aux_loss = torch.stack(block_aux_losses).mean()
            self._last_moe_z_loss = torch.stack(block_z_losses).mean()
            self.last_moe_metrics = {
                "aux_loss": float(self._last_moe_aux_loss.detach().item()),
                "z_loss": float(self._last_moe_z_loss.detach().item()),
                "dropped_frac": sum(m["dropped_frac"] for m in block_metrics) / len(block_metrics),
                "expert_entropy": sum(m["expert_entropy"] for m in block_metrics) / len(block_metrics),
            }
        else:
            zero = x.new_zeros(())
            self._last_moe_aux_loss = zero
            self._last_moe_z_loss = zero
            self.last_moe_metrics = {
                "aux_loss": 0.0,
                "z_loss": 0.0,
                "dropped_frac": 0.0,
                "expert_entropy": 0.0,
            }

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return logits, loss

    def get_moe_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._last_moe_aux_loss is None or self._last_moe_z_loss is None:
            zero = self.token_emb.weight.new_zeros(())
            return zero, zero
        return self._last_moe_aux_loss, self._last_moe_z_loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=-1)
        return idx
