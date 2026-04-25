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
            self.global_patch_pad = nn.Parameter(torch.zeros(1, 1, config.d_model * self.byte_patch_size))
            self.global_to_local = nn.Linear(config.d_model, config.d_model * self.byte_patch_size)
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
            self.local_blocks = nn.ModuleList([
                TransformerBlock(config) for _ in range(config.byte_local_layers)
            ])
            self.local_ln_f = nn.RMSNorm(config.d_model, eps=1e-5)
            self.patch_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.patch_head.weight = self.token_emb.weight
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
            sequence = torch.cat([idx[:, :1], targets], dim=1)
            n_patches = sequence.size(1) // P
            if n_patches < 1:
                raise ValueError(f"byte_patch_size={P} requires at least {P} byte ids for training")
            target_patches = sequence[:, : n_patches * P].reshape(B, n_patches, P)
            global_states = self._encode_global_patches(sequence, n_patches)
        else:
            n_patches = T // P
            if n_patches < 1:
                raise ValueError(f"byte_patch_size={P} requires at least {P} byte ids")
            target_patches = None
            global_states = self._encode_global_patches(idx, n_patches)

        logits = self._decode_patch_bytes(global_states, target_patches)
        if target_patches is None:
            return logits[:, : n_patches * P]

        target_flat = target_patches.reshape(B, n_patches * P)
        logits = logits[:, : target_flat.size(1)]
        loss = nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            target_flat.reshape(-1),
        )
        return logits, loss

    def _encode_global_patches(self, sequence: torch.Tensor, n_patches: int) -> torch.Tensor:
        assert self.patch_proj is not None
        B = sequence.size(0)
        P = self.byte_patch_size
        D = self.config.d_model
        if n_patches <= 0:
            raise ValueError("n_patches must be positive")
        if n_patches == 1:
            global_input = self.global_patch_pad.expand(B, 1, P * D)
        else:
            previous_patches = sequence[:, : (n_patches - 1) * P].reshape(B, n_patches - 1, P)
            previous_emb = self.token_emb(previous_patches).reshape(B, n_patches - 1, P * D)
            global_input = torch.cat(
                [self.global_patch_pad.expand(B, 1, P * D), previous_emb],
                dim=1,
            )
        x = self.patch_proj(global_input)
        return self._run_blocks(x)

    def _encode_next_patch_context(self, sequence: torch.Tensor) -> torch.Tensor:
        assert self.patch_proj is not None
        B = sequence.size(0)
        P = self.byte_patch_size
        D = self.config.d_model
        n_patches = sequence.size(1) // P
        if n_patches < 1:
            global_input = self.global_patch_pad.expand(B, 1, P * D)
        else:
            patches = sequence[:, : n_patches * P].reshape(B, n_patches, P)
            patch_emb = self.token_emb(patches).reshape(B, n_patches, P * D)
            global_input = torch.cat(
                [self.global_patch_pad.expand(B, 1, P * D), patch_emb],
                dim=1,
            )
        x = self.patch_proj(global_input)
        return self._run_blocks(x)[:, -1:]

    def _build_local_inputs(
        self,
        patch_states: torch.Tensor,
        target_patches: torch.Tensor | None,
        generated_prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = patch_states.shape
        P = self.byte_patch_size
        if target_patches is not None:
            bos = torch.zeros((B, N, 1), dtype=torch.long, device=target_patches.device)
            prev_tokens = torch.cat([bos, target_patches[:, :, :-1]], dim=2)
        else:
            prev_tokens = torch.zeros((B, N, P), dtype=torch.long, device=patch_states.device)
            if generated_prefix is not None and generated_prefix.numel() > 0:
                prefix_len = min(generated_prefix.size(1), P - 1)
                prev_tokens[:, :, 1 : prefix_len + 1] = generated_prefix[:, None, :prefix_len]
        prev_emb = self.token_emb(prev_tokens)
        positions = torch.arange(P, device=patch_states.device)
        prev_emb = prev_emb + self.patch_byte_pos_emb(positions).view(1, 1, P, C)
        patch_context = self.global_to_local(patch_states).reshape(B, N, P, C)
        return patch_context + prev_emb

    def _decode_patch_bytes(
        self,
        patch_states: torch.Tensor,
        target_patches: torch.Tensor | None,
        generated_prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.patch_head is not None
        B, N, C = patch_states.shape
        P = self.byte_patch_size
        local_hidden = self._build_local_inputs(patch_states, target_patches, generated_prefix)
        local_hidden = local_hidden.reshape(B * N, P, C)
        local_hidden = self.drop(local_hidden)
        for block in self.local_blocks:
            local_hidden = block(local_hidden)
        local_hidden = self.local_ln_f(local_hidden)
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
            patch_state = self._encode_next_patch_context(idx_cond)
            next_patch = []
            for local_index in range(P):
                prefix = torch.cat(next_patch, dim=-1) if next_patch else None
                logits = self._decode_patch_bytes(
                    patch_state,
                    target_patches=None,
                    generated_prefix=prefix,
                )
                token_logits = logits[:, local_index, :] / temperature
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
