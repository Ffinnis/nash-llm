import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nash_llm.config import ModelConfig


class MultiHeadAttention(nn.Module):
    mask: torch.Tensor

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})")
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({config.n_heads}) must be divisible by n_kv_heads ({config.n_kv_heads})"
            )

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.queries_per_kv = self.n_heads // self.n_kv_heads
        self.position_embedding = config.position_embedding

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, self.kv_dim)
        self.v_proj = nn.Linear(config.d_model, self.kv_dim)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout_p = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.sdpa_supports_gqa = self._check_sdpa_gqa_support()

        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

        if self.position_embedding == "rope":
            if self.head_dim % 2 != 0:
                raise ValueError(
                    f"RoPE requires an even head_dim, got {self.head_dim} from "
                    f"d_model={config.d_model}, n_heads={config.n_heads}"
                )
            inv_freq = 1.0 / (
                config.rope_base
                ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            )
            positions = torch.arange(config.max_seq_len, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            self.register_buffer("rope_cos", torch.cos(freqs), persistent=False)
            self.register_buffer("rope_sin", torch.sin(freqs), persistent=False)

    @staticmethod
    def _check_sdpa_gqa_support() -> bool:
        try:
            q = torch.empty(1, 2, 1, 1)
            k = torch.empty(1, 1, 1, 1)
            v = torch.empty(1, 1, 1, 1)
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        except TypeError:
            return False
        return True

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        if seq_len > self.rope_cos.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds RoPE cache size {self.rope_cos.size(0)}")

        cos = self.rope_cos[:seq_len].to(device=x.device, dtype=x.dtype).view(1, 1, seq_len, -1)
        sin = self.rope_sin[:seq_len].to(device=x.device, dtype=x.dtype).view(1, 1, seq_len, -1)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rotated = torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        )
        return x_rotated.flatten(-2)

    def _expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads == self.n_heads:
            return x
        return x.repeat_interleave(self.queries_per_kv, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.position_embedding == "rope":
            q = self._apply_rope(q, T)
            k = self._apply_rope(k, T)

        if hasattr(F, "scaled_dot_product_attention"):
            sdpa_kwargs = {
                "attn_mask": None,
                "dropout_p": self.dropout_p if self.training else 0.0,
                "is_causal": True,
            }
            if self.sdpa_supports_gqa:
                sdpa_kwargs["enable_gqa"] = self.n_kv_heads != self.n_heads
                out = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
            else:
                out = F.scaled_dot_product_attention(
                    q,
                    self._expand_kv_heads(k),
                    self._expand_kv_heads(v),
                    **sdpa_kwargs,
                )
        else:
            scale = math.sqrt(self.head_dim)
            k = self._expand_kv_heads(k)
            v = self._expand_kv_heads(v)
            attn = (q @ k.transpose(-2, -1)) / scale
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))
