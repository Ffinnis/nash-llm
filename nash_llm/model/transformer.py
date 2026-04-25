import torch
import torch.nn as nn
from nash_llm.config import ModelConfig
from nash_llm.model.attention import MultiHeadAttention
from nash_llm.model.layers import FeedForward

if not hasattr(nn, "RMSNorm"):
    raise RuntimeError(
        "torch.nn.RMSNorm is required for this project. "
        "Please install a PyTorch version that provides nn.RMSNorm."
    )


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.d_model, eps=1e-5)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.RMSNorm(config.d_model, eps=1e-5)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.byte_patch_size = config.byte_patch_size

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_proj = None
        if self.byte_patch_size > 1:
            self.patch_proj = nn.Linear(config.d_model * self.byte_patch_size, config.d_model)
        self.pos_emb = None
        if config.position_embedding == "learned":
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.RMSNorm(config.d_model, eps=1e-5)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.patch_head = None
        if self.byte_patch_size > 1:
            self.patch_byte_pos_emb = nn.Embedding(self.byte_patch_size, config.d_model)
            self.local_byte_proj = nn.Linear(config.d_model * 2, config.d_model)
            self.patch_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if self.byte_patch_size > 1:
            return self._forward_patched(idx, targets)

        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        tok_emb = self.token_emb(idx)
        x = tok_emb
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return logits, loss

    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=x.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def _forward_patched(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        assert self.patch_proj is not None
        assert self.patch_head is not None
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        P = self.byte_patch_size
        if targets is not None:
            sequence = torch.cat([idx, targets[:, -1:]], dim=1)
            n_patches = sequence.size(1) // P
            if n_patches < 2:
                raise ValueError(
                    f"byte_patch_size={P} requires at least {2 * P - 1} byte ids for training"
                )
            input_patches = sequence[:, : (n_patches - 1) * P].reshape(B, n_patches - 1, P)
            target_patches = sequence[:, P : n_patches * P].reshape(B, n_patches - 1, P)
        else:
            n_patches = T // P
            if n_patches < 1:
                raise ValueError(f"byte_patch_size={P} requires at least {P} byte ids")
            input_patches = idx[:, : n_patches * P].reshape(B, n_patches, P)
            target_patches = None

        patch_emb = self.token_emb(input_patches).reshape(B, input_patches.size(1), P * self.config.d_model)
        x = self.patch_proj(patch_emb)
        x = self._run_blocks(x)
        logits = self._decode_patch_bytes(x, target_patches)

        if target_patches is None:
            return logits

        loss = nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            target_patches.reshape(-1),
        )
        return logits, loss

    def _decode_patch_bytes(
        self,
        patch_states: torch.Tensor,
        target_patches: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.patch_head is not None
        assert self.local_byte_proj is not None
        B, N, C = patch_states.shape
        P = self.byte_patch_size
        if target_patches is None:
            prev_tokens = torch.zeros((B, N, P), dtype=torch.long, device=patch_states.device)
        else:
            bos = torch.zeros((B, N, 1), dtype=torch.long, device=target_patches.device)
            prev_tokens = torch.cat([bos, target_patches[:, :, :-1]], dim=2)

        prev_emb = self.token_emb(prev_tokens)
        positions = torch.arange(P, device=patch_states.device)
        prev_emb = prev_emb + self.patch_byte_pos_emb(positions).view(1, 1, P, C)
        patch_context = patch_states.unsqueeze(2).expand(B, N, P, C)
        local_hidden = self.local_byte_proj(torch.cat([patch_context, prev_emb], dim=-1))
        logits = self.patch_head(local_hidden)
        return logits.reshape(B, N * P, self.config.vocab_size)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        if self.byte_patch_size > 1:
            return self._generate_patched(idx, max_new_tokens, temperature=temperature, top_k=top_k)

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

    @torch.no_grad()
    def _generate_patched(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        P = self.byte_patch_size
        prompt = idx
        original_len = idx.size(1)
        pad_len = (-original_len) % P
        if pad_len:
            pad = torch.zeros((idx.size(0), pad_len), dtype=idx.dtype, device=idx.device)
            idx = torch.cat([idx, pad], dim=1)

        target_len = original_len + pad_len + max_new_tokens
        while idx.size(1) < target_len:
            idx_cond = idx[:, -self.config.max_seq_len :]
            trim = idx_cond.size(1) % P
            if trim:
                idx_cond = idx_cond[:, trim:]
            logits = self(idx_cond)
            next_patch = []
            for local_index in range(P):
                token_logits = logits[:, -P + local_index, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(token_logits, min(top_k, token_logits.size(-1)))
                    token_logits = token_logits.masked_fill(
                        token_logits < v[:, [-1]],
                        float("-inf"),
                    )
                probs = torch.softmax(token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_patch.append(next_token)
            next_patch = torch.cat(next_patch, dim=-1)
            idx = torch.cat([idx, next_patch], dim=-1)

        generated = idx[:, original_len + pad_len : original_len + pad_len + max_new_tokens]
        return torch.cat([prompt, generated], dim=-1)
