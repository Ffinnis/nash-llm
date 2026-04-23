# TEON for `nash-llm`: what the paper says and what it implies for AdamW

Paper: `https://arxiv.org/abs/2601.23261`

## Core idea

The paper proposes **TEON**, a tensorized extension of Muon. Instead of orthogonalizing each matrix update independently, it stacks multiple same-type matrices from nearby layers into a tensor, unfolds that tensor into a matrix, then applies the usual Muon-style `Ortho(.)` update on the unfolded view.

Practical recipe from the paper:

- use momentum, then orthogonalize the momentum tensor
- use **mode-1** matricization
- use **K = 2** consecutive layers
- stack only **same-type Q/K/V projection weights**
- avoid stacking too many layers or too many heterogeneous matrix types

Why:

- biggest gains come when stacked matrices have aligned top singular vectors
- in transformers, this alignment is strongest for same-type `Q`, `K`, `V` across adjacent layers
- MLP / output projections degrade the approximation more often, especially with approximate orthogonalization

## What matters for this repo

The current optimizer split in the repo already follows the paper's practical guidance closely:

- [`nash_llm/optim/adamw.py`](/Users/roman/Documents/nash-llm/nash_llm/optim/adamw.py:8) builds a two-optimizer setup
- `q_proj.weight`, `k_proj.weight`, `v_proj.weight` are grouped into `K=2` TEON stacks
- `out_proj.weight`, `fc1.weight`, `fc_gate.weight`, `fc2.weight` go through per-layer Muon
- the remaining parameters go to AdamW
- [`nash_llm/optim/muon.py`](/Users/roman/Documents/nash-llm/nash_llm/optim/muon.py:188) performs the actual concatenation + orthogonalization for TEON groups

So the repo is already implementing the exact direction of the paper on the parameter classes where the paper expects gains.

## Can a similar optimization be applied "on top of AdamW"?

Short answer: **not as a clean extra layer on the current AdamW branch**.

Reasons:

1. TEON is not a generic "make any optimizer better" wrapper.
   It is specifically an orthogonalized update geometry for matrix-shaped parameters with meaningful cross-layer correlation.

2. AdamW already changes the update geometry using elementwise second-moment preconditioning:

   - update direction is roughly `m_t / (sqrt(v_t) + eps)`
   - TEON/Muon replace singular values by an orthogonalized matrix structure

   Applying TEON after Adam-style preconditioning would create a different optimizer, not "AdamW plus a free extra improvement".

3. The parameters currently left in AdamW in this repo are mostly the ones the paper does **not** target well:

   - biases
   - norm parameters
   - embeddings / other leftover matrices without obvious adjacent same-type layer structure

   These do not naturally satisfy the paper's alignment assumption.

## What is plausible instead

There are two plausible experimental directions:

### 1. `AdamW -> TEON` on selected matrix params

Take Adam's preconditioned matrix update

`U_t = m_t / (sqrt(v_t) + eps)`

and then apply TEON/Muon-style orthogonalization only to selected matrix groups:

- adjacent `q_proj`
- adjacent `k_proj`
- adjacent `v_proj`

This is conceptually similar to "Adam-like preconditioning + structured matrix geometry". But this is **not validated by the paper**, and it can partially destroy Adam's useful per-coordinate scaling by re-normalizing singular values afterward.

### 2. Extend TEON coverage, not AdamW coverage

A safer repo-grounded experiment is:

- keep current AdamW split intact
- try whether more matrix params should move from AdamW into Muon/TEON
- only do this for repeated matrix families with matching shapes and plausible cross-layer alignment

This is more faithful to the paper than trying to wrap TEON around all AdamW updates.

## Recommended conclusion for this repo

If the question is whether to add a TEON-like step **additionally over the current AdamW bucket**, the answer is:

- **probably not by default**
- **only as a separate experiment**
- **only for matrix-shaped params with repeated cross-layer structure**

If the goal is better training efficiency, the first things worth trying are:

1. verify that current TEON coverage exactly matches the intended transformer blocks in every config
2. sweep `K`, `muon_lr`, and `ns_steps` only for TEON/Muon params
3. optionally prototype an `adam_teon` optimizer variant behind a config flag for `q/k/v` only

## Concrete hypothesis

For this repo, a naive "TEON on the whole AdamW branch" is unlikely to help much and may hurt.

A narrow variant like:

- Adam-style preconditioning
- then TEON on `q/k/v` only
- keep norms/biases/embeddings on plain AdamW

is technically defensible, but it should be treated as a new optimizer research experiment, not as an obvious paper-backed extension.
