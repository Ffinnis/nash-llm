# QV / QV-Ka attention for `nash-llm`

Paper: *QV May Be Enough: Toward the Essence of Attention in LLMs* (`arXiv:2603.15665`)

## Core idea

The paper argues that standard attention may be over-parameterized on the key side.

It proposes two related directions:

- `QV`: drop the standalone `K` projection and score attention with `Q @ V^T`
- `QV-Ka`: keep a normal `Q` and `V`, but derive `K` from `V` plus a compact context projection instead of using a full-size independent `k_proj`

The empirical claim is not that pure `QV` is strictly better. In the paper:

- plain `QV` is a bit worse than standard `QKV`
- the stronger result comes from `QV-Ka`
- the best comparisons depend on replacing normal positional handling with an AGF-style relative-position setup

## What matters for this repo

Current repo shape:

- decoder-only GPT with per-block [`MultiHeadAttention`](/Users/roman/Documents/nash-llm/nash_llm/model/attention.py:8)
- separate [`q_proj`, `k_proj`, `v_proj`](/Users/roman/Documents/nash-llm/nash_llm/model/attention.py:20)
- default positional mode is [`rope`](/Users/roman/Documents/nash-llm/nash_llm/config/config.py:15)
- optimizer logic explicitly groups `q_proj.weight`, `k_proj.weight`, `v_proj.weight` into TEON stacks in [`nash_llm/optim/adamw.py`](/Users/roman/Documents/nash-llm/nash_llm/optim/adamw.py:32)

So this paper touches a very central seam in the codebase:

- attention math
- model config surface
- optimizer parameter grouping
- test assumptions that currently expect explicit `k_proj`

## Can it be tested in a separate branch?

Yes, as a contained research branch.

But it should be treated as a hypothesis test, not as an obviously correct architecture upgrade.

Why the idea is testable here:

- attention is isolated in one module
- config-driven ablations are already normal in this repo
- there is already precedent for paper-driven architectural experiments in `knowledge/`

Why this is not a low-risk “paper says so” change:

- the paper’s experiments are on a much smaller translation setup, not a decoder-only GPT pretraining setup
- the paper’s best results lean on AGF/relative-position logic, while this repo defaults to RoPE
- pure `QV` is not the promising part of the paper; `QV-Ka` is

## Main compatibility concern: RoPE

This is the biggest issue for `nash-llm`.

Today the repo applies RoPE to `q` and `k`, not to `v`, which is the standard and clean formulation. The paper’s argument for stronger `QV` behavior relies on reducing or restructuring positional interference. In practice:

- current `QKV + RoPE` is a well-understood decoder setup
- naive `QV` under RoPE is not well supported by the paper
- removing `k_proj` without rethinking positional handling would be the wrong experiment

So the safest interpretation is:

- do **not** replace the current default attention with pure `QV`
- if we test this paper, the primary target should be `QV-Ka`, or a very narrow “key derived from value/context” variant that still preserves a proper RoPE-compatible score path

## Best experiment to try first

Recommended first branch scope:

1. Add an opt-in attention variant config, for example `model.attention_variant = qkv | qv_ka`
2. Keep `qkv` as the default
3. Implement `qv_ka` inside `MultiHeadAttention` without changing the rest of the model stack
4. Keep RoPE on the score path only, exactly as today for `q` and the final derived `k`
5. Run only small ablations first (`pretrain_small` / debug-scale), looking at:
   - training stability
   - tokens/sec
   - validation loss
   - parameter count

Why `QV-Ka` first:

- it is the paper’s strongest practical result
- it preserves the familiar attention structure `softmax(QK^T)V`
- it avoids the most aggressive semantic/positional entanglement of plain `QV`
- it can reduce key-side parameters while still staying close to the current code structure

## What I would not test first

I would avoid these as the first branch version:

- full replacement of `QKV` with plain `QV`
- simultaneous introduction of both `QV` and a new positional system
- any refactor that breaks TEON grouping and optimizer behavior at the same time as attention math changes

That would confound too many variables.

## Minimal implementation shape

A practical `QV-Ka` adaptation for this repo could look like:

- keep `q_proj`
- keep `v_proj`
- remove or bypass full-size `k_proj` in the experimental path
- add a compact context projection from the block input
- derive per-head `k` from `[ctx ; v]`
- apply RoPE to `q` and the derived `k`
- keep the output path unchanged

This keeps the external interface stable:

- block API stays the same
- `GPT.forward()` stays the same
- training loop stays the same

Only the attention internals and optimizer grouping need experimental branching.

## Optimizer impact

This repo’s optimizer setup is tightly coupled to the existence of `k_proj.weight`.

If `QV-Ka` is added, the optimizer split should become variant-aware:

- standard `qkv` path: keep current TEON grouping
- experimental path: keep TEON on `q_proj.weight` and `v_proj.weight`
- put the new key-derivation weights in Muon or AdamW first, not necessarily TEON

The conservative choice is to start the new key-derivation weights in plain Muon-per-layer or AdamW, not invent a new TEON grouping immediately.

## Expected upside

If the branch works, the likely wins are:

- fewer attention parameters
- somewhat lower attention projection FLOPs
- a cleaner ablation story around “how necessary is an explicit key projection in this codebase?”

The likely outcome is not “better than baseline everywhere”.
The more realistic goal is:

- near-baseline quality
- slightly cheaper attention
- useful research signal for future KV-sharing or inference-oriented work

## Expected risks

- worse optimization than baseline because the paper’s evidence does not directly transfer to decoder-only GPT + RoPE
- degraded TEON effectiveness if the new projection layout breaks the clean repeated matrix families
- confusion from mixing architectural novelty with optimizer novelty

## Recommended conclusion

My verdict for this repo:

- **yes**, it is worth testing in a separate branch
- **no**, it should not be merged as the new default based on this paper alone
- the right first experiment is **`QV-Ka`, not pure `QV`**
- the branch should keep **RoPE and the rest of the training stack unchanged**
- success criterion should be **baseline-like loss at lower parameter/FLOP cost**, not a guaranteed quality gain

## Suggested branch plan

If we decide to implement it, the clean order is:

1. add config flag for attention variant
2. implement `qv_ka` path in attention only
3. update optimizer grouping for the new parameter names
4. update unit tests for variant-specific expectations
5. run short training ablations outside this session

That gives a clean research branch without destabilizing the main path.
