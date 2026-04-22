# RoFormer / RoPE Summary

Paper: *RoFormer: Enhanced Transformer with Rotary Position Embedding* (`arXiv:2104.09864`)

## Core idea

RoPE replaces additive position embeddings with a multiplicative rotation applied to `q` and `k`. In each 2D pair of hidden dimensions, the model rotates the projected vector by an angle proportional to token position. In higher dimensions, that becomes a block-diagonal stack of 2x2 rotations.

The attention score becomes position-aware through:

- absolute positions encoded as rotations on `q` and `k`
- relative positions emerging in the inner product `q_m^T k_n`
- no change needed for `v`

This is the main implementation takeaway for this repo: rotate `q` and `k` per head after projection and before attention, instead of adding a learned or sinusoidal embedding to the token stream.

## Why it fits `nash-llm`

Current repo shape:

- decoder-only GPT
- separate `q_proj`, `k_proj`, `v_proj`
- causal self-attention in [nash_llm/model/attention.py](/Users/roman/Documents/nash-llm/nash_llm/model/attention.py)
- token embedding plus learned absolute position table in [nash_llm/model/transformer.py](/Users/roman/Documents/nash-llm/nash_llm/model/transformer.py)

That makes RoPE a clean swap:

- keep token embeddings unchanged
- stop depending on an additive absolute position table for the default path
- rotate `q` and `k` inside attention
- preserve the rest of the training stack

## Paper details that matter for implementation

- RoPE is applied to queries and keys, not values.
- The embedding dimension is handled in pairs, so the per-head dimension must be even.
- The default frequency schedule uses `theta_i = 10000^(-2(i-1)/d)` in the paper's notation.
- The efficient implementation does not build the full rotation matrix; it uses paired `cos`/`sin` mixing of even/odd channels.
- The paper motivates better length flexibility and an implicit distance decay in attention similarity.

## Decisions applied in this repo

Implemented direction:

- default `model.position_embedding = "rope"`
- optional fallback `model.position_embedding = "learned"` for ablations
- new `model.rope_base = 10000.0`
- RoPE cache is precomputed up to `max_seq_len`
- validation rejects RoPE when `head_dim` is odd

Why keep the learned fallback:

- it preserves a direct baseline for comparison
- it keeps older experiments easy to reproduce from config only

## Expected impact

- fewer learned parameters because the default path drops the learned position table
- cleaner relative-position behavior in attention
- a better foundation if this repo later adds KV-cache inference or longer-context experiments

## Limits to remember

- RoPE does not by itself extend the trained context window; `max_seq_len` is still the active bound in this codebase.
- True long-context gains still depend on training setup, data, and whether the model is exposed to longer sequences.
- This implementation keeps the existing quadratic causal attention. The paper also discusses linear-attention compatibility, but that is not relevant to the current repo yet.
