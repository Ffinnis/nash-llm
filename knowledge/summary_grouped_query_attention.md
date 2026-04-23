# GQA Summary

Paper: *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (`arXiv:2305.13245`)

## Core idea

Grouped-query attention keeps the usual number of query heads, but reduces the number of key/value heads.

- `Q` still uses `n_heads`
- `K` and `V` use `n_kv_heads`
- each KV head is shared by a group of query heads

This makes GQA an interpolation between classic multi-head attention and multi-query attention:

- `n_kv_heads == n_heads` gives standard MHA
- `n_kv_heads == 1` gives MQA
- `1 < n_kv_heads < n_heads` gives GQA

The paper's main result is pragmatic: GQA keeps model quality much closer to MHA than MQA does, while preserving most of the inference-speed benefit from shrinking the KV side.

## Why it fits `nash-llm`

Current repo shape:

- decoder-only GPT
- separate `q_proj`, `k_proj`, `v_proj`
- RoPE already applied inside attention
- no KV-cache implementation yet, but the attention module already owns the right projection boundary

That makes GQA a clean architectural upgrade:

- keep the query path unchanged
- shrink only the `k_proj` / `v_proj` output width
- share each KV head across multiple query heads inside attention
- preserve MHA behavior by default when `n_kv_heads` is omitted

## Paper details that matter for implementation

- GQA groups query heads and shares one key head and one value head per group.
- The paper converts MHA checkpoints to GQA by mean-pooling the original key/value heads inside each group.
- GQA is presented as the stable middle ground between MHA and MQA.
- The biggest practical win is decoder inference, especially once a KV cache exists.
- The paper reports that MQA is more unstable than GQA; GQA uptraining behaves much closer to the original model.

## Decisions applied in this repo

Implemented direction:

- new `model.n_kv_heads` config field
- default `n_kv_heads = n_heads`, so existing configs remain MHA
- `q_proj` still outputs `d_model`
- `k_proj` and `v_proj` now output `n_kv_heads * head_dim`
- runtime validation requires `n_heads % n_kv_heads == 0`
- attention uses grouped KV sharing and falls back to explicit KV expansion when PyTorch does not expose native GQA SDPA support

## What this changes right now

- parameter count drops when `n_kv_heads < n_heads` because `k_proj` and `v_proj` get narrower
- model configs can now express MHA, GQA, and MQA with one code path
- checkpoint compatibility is preserved for existing MHA runs because the default remains `n_kv_heads == n_heads`

## Limits to remember

- This repo still trains full-sequence attention and does not yet implement a KV cache, so the paper's largest decoding-speed gains are not fully realized here yet.
- We did not implement checkpoint conversion / mean-pooling from existing MHA checkpoints in this pass.
- The current code adds the architectural primitive first; the next natural step would be KV-cache generation plus optional MHA-to-GQA checkpoint conversion.
