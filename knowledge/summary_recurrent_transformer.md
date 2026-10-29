# Recurrent Transformer Summary

Paper: *The Recurrent Transformer: Greater Effective Depth and Efficient Decoding* (`arXiv:2604.21215`, submitted April 23, 2026)

## Short verdict for `nash-llm`

Yes, this is implementable here as a research branch.

But there are really two different scopes:

1. `Architecture-only prototype`
   We can implement the paper's recurrent-in-layer semantics in pure PyTorch inside the current repo.
2. `Paper-faithful efficient training path`
   We cannot get that just by editing the current attention module. The paper's practical training story depends on a tiled forward schedule, preallocated persistent KV storage, and a custom backward path to avoid autograd/in-place issues.

So the right answer is: **yes as research, no as a cheap drop-in baseline replacement**.

## What the paper changes

A standard decoder block computes `q`, `k`, and `v` from the current layer input `x_i`, attends over the prefix, then produces the output.

This paper changes only the source of the **persistent** key/value memory:

- `q_i`, `k_i_temp`, `v_i_temp` are still computed from the current layer input `x_i`
- attention at position `i` uses earlier persistent pairs plus the current temporary pair
- the layer output `z_i` is computed after attention + MLP
- the persistent `k_i`, `v_i` that future positions see are projected from `z_i`, not from `x_i`

That gives each layer a recurrent temporal memory without introducing a bounded state bottleneck.

## Why it fits this repo conceptually

Current repo shape already matches the paper at a high level:

- decoder-only model with explicit block structure in [nash_llm/model/transformer.py](/Users/roman/Documents/nash-llm/nash_llm/model/transformer.py:14)
- separate `q_proj`, `k_proj`, `v_proj`, `out_proj` in [nash_llm/model/attention.py](/Users/roman/Documents/nash-llm/nash_llm/model/attention.py:27)
- RMSNorm-based block structure in [nash_llm/model/transformer.py](/Users/roman/Documents/nash-llm/nash_llm/model/transformer.py:17)
- model/training configuration already exposed through dataclasses and CLI overrides in [nash_llm/config/config.py](/Users/roman/Documents/nash-llm/nash_llm/config/config.py:5)
- trainer already supports ablation-style experiments via config-driven runs in [nash_llm/training/trainer.py](/Users/roman/Documents/nash-llm/nash_llm/training/trainer.py:52)

This means the repo is a good place to test the idea:

- there is a clean attention boundary to replace
- there is no huge inference-serving stack to untangle
- the repo already treats paper-driven architecture changes as experiments

## What does **not** fit cleanly right now

The current attention path assumes fully parallel causal attention:

- all `k` and `v` come from the same `x`
- one `scaled_dot_product_attention(...)` call handles the whole sequence
- generation currently re-runs the whole prefix each step instead of maintaining a KV cache in [nash_llm/model/transformer.py](/Users/roman/Documents/nash-llm/nash_llm/model/transformer.py:82)

The Recurrent Transformer breaks that assumption:

- persistent KV for token `i` depends on `z_i`
- `z_i` itself depends on attention over previous persistent KV
- naive training/prefill becomes sequential in position

That means a pure replacement of `MultiHeadAttention.forward()` is possible, but it will be much slower than the repo's current SDPA path and much slower than the paper's optimized implementation.

## Practical feasibility by stage

### Stage 1: Architecture-only research prototype

Feasible now.

Implement a new attention/block path that:

- computes `q`, `k_temp`, `v_temp` from normalized `x`
- iterates positions left-to-right
- attends over `persistent_kv[:i-1]` plus the temporary pair at `i`
- forms the token output
- projects persistent `k`, `v` from the resulting output

This should be introduced behind a config flag, not as a replacement for the default model.

Suggested config additions:

- `model.attention_variant: "vanilla" | "recurrent"`
- optionally `model.recurrent_kv_norm: true`
- optionally `model.recurrent_residual_scale: true`

This stage is enough to answer the research question:

- does the architecture train stably in our codebase
- does it beat a matched baseline on early loss
- how bad is the throughput penalty without custom kernels

### Stage 2: Repo-native stable experiment path

Feasible, but more invasive.

Needed work:

- new recurrent attention module instead of overloading current SDPA code
- block-level refactor, because persistent KV depends on the post-MLP token state, not just the attention sublayer output
- careful optimizer routing review so new trainable weights, if any, stay in the intended Muon/SAGE buckets
- experiment configs that compare matched parameter budgets and matched token budgets

This is still Python-level repo work and is realistic for a feature branch.

### Stage 3: Paper-faithful efficient implementation

Possible, but this is a serious systems project.

The paper's speed story relies on:

- exact tiled prefill/training schedule
- preallocated persistent KV writes
- custom backward because autograd does not like the required in-place cache writes
- CUDA Graphs to reduce kernel launch overhead
- layout choices tuned for cache locality

That is beyond the current repo's implementation level. It likely means Triton or custom CUDA work, plus a dedicated benchmark harness.

## Main mismatches versus the paper

### 1. Position encoding mismatch

The paper reports its main pretraining experiments with ALiBi, while this repo currently supports `rope` and `learned` in [nash_llm/config/config.py](/Users/roman/Documents/nash-llm/nash_llm/config/config.py:16).

This is not a blocker, but it means:

- first prototype can run with RoPE for minimal code churn
- if results are promising, a fairer paper-style comparison likely wants ALiBi added too

### 2. Optimizer mismatch

The paper reports Adam/AdamW-style runs, while this repo's baseline stack is Muon + SAGE in the trainer setup at [nash_llm/training/trainer.py](/Users/roman/Documents/nash-llm/nash_llm/training/trainer.py:73).

That is also not a blocker, but it changes interpretation:

- best first experiment is "same repo optimizer stack, only architecture changed"
- separate optimizer matching to the paper should be treated as a second ablation

### 3. Throughput expectations

Do not expect a speedup from a naive implementation here.

The paper itself reports better quality and a better depth/width tradeoff, but its raw layer throughput is still much worse than vanilla Transformer unless the optimized tiled schedule is used. In the appendix, their 300M C4 setup reports about `42k-49k tok/s` for RT vs `132k-153k tok/s` for vanilla Transformer at matched settings.

For `nash-llm`, that means the first win to look for is **quality per parameter / quality per layer**, not training throughput.

## What I would implement first

### Recommendation

Build a narrow research branch with:

1. `RecurrentTransformerBlock` and `RecurrentAttention`
2. config flag to switch between vanilla and recurrent blocks
3. naive exact forward implementation first
4. no attempt at fused kernels in v1
5. small matched experiments on existing configs

### Minimum experiment matrix

Use the current repo commands and configs, but run short controlled comparisons:

1. Baseline current model on `configs/pretrain_small.yaml`
2. Same config with `model.attention_variant=recurrent`
3. If recurrent is slower, reduce `n_layers` and increase width to keep parameters roughly matched
4. Compare early `val_loss`, stability, and tokens/sec

The most meaningful research question for this repo is:

`Does recurrent attention let us trade some depth for width and recover better loss at the same parameter budget, even if raw training throughput gets worse?`

## Proposed acceptance criteria

I would treat the idea as promising only if all of these happen:

- no obvious instability in the first short runs
- early loss is not clearly worse by around the first 100 steps
- reduced-depth recurrent variants stay competitive or better at similar parameter count
- throughput penalty is large but still operational for research

If the recurrent branch is already clearly worse early, it should be stopped quickly rather than defended with "maybe later it catches up."

## Concrete next step

If we decide to pursue it, the next implementation should be:

1. add `attention_variant` to `ModelConfig`
2. split block code into `TransformerBlock` and `RecurrentTransformerBlock`
3. implement a naive exact recurrent forward path
4. add one dedicated experiment config derived from `pretrain_small`
5. run short early-loss comparisons on the same dataset and token budget

That is the right scale for this repo. The tiled algorithm and custom backward should only happen if the naive branch shows clear modeling upside.
