# SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization

Paper: [arXiv 2604.07663](https://arxiv.org/abs/2604.07663)

## What the paper changes

SAGE replaces the usual AdamW fallback in a hybrid optimizer stack.

- Dense matrix weights stay on a matrix-aware optimizer.
- Embeddings move to a Lion-style sign optimizer with an `O(d)` adaptive damper.
- 1D parameters such as norms and biases use the same SAGE update instead of AdamW.

The key state reduction is on embeddings: AdamW needs two full `O(Vd)` moments, while SAGE keeps one full momentum tensor plus a reduced `O(d)` damper state.

## Core update

For a parameter tensor `theta`:

1. Apply decoupled weight decay.
2. Build `s_t` from absolute gradients:
   - 2D embedding-like tensors: mean over rows, shape `1 x d`
   - 1D tensors: elementwise absolute value
3. Update EMA state `S_t` with `beta2`, then bias-correct it.
4. Compute a relative RMS damper:
   - `ema_damper = rms(S_hat) / (S_hat + eps)`
   - `instant_damper = rms(s_t) / (s_t + eps)`
   - `H_t = min(ema_damper, instant_damper, 1.0)`
5. Use a Lion-style sign direction:
   - `sign(beta1 * m_prev + (1 - beta1) * grad)`
6. Multiply the sign direction by `H_t` and step with learning rate `lr`.
7. Update the momentum buffer with `beta2`.

Reference defaults used by the authors' public implementation:

- `beta1 = 0.9`
- `beta2 = 0.99`
- `eps = 1e-8`

## What matters for nash-llm

`nash-llm` already has a clean optimizer split:

- `TEON/Muon`: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc_gate`, `fc2`
- fallback optimizer: embeddings, norms, biases, and leftovers

That means the paper maps cleanly onto the current codebase:

- Keep `TEON + Muon` exactly where it already helps: the big transformer matrices.
- Replace only the fallback bucket with SAGE.
- Let token embeddings and any learned positional embeddings use the 2D SAGE path.
- Let RMSNorm weights and biases use the 1D SAGE path.

This is closer to the paper than trying to push SAGE into the higher-dimensional matrix path, because the paper's point is to solve the embedding-layer dilemma without giving up a separate matrix-aware optimizer.

## Practical expectation for this repo

- Memory should drop versus AdamW in the fallback bucket, especially around embeddings.
- The main risk is not routing: it is tuning. `nash-llm` currently uses `learning_rate = 3e-4` defaults built around AdamW-ish behavior.
- The SAGE paper reports best results with a noticeably higher SAGE learning rate than Lion/Adam-style baselines, so this branch should be treated as an optimizer experiment, not as a guaranteed drop-in win at the old LR.

## Suggested follow-up experiment

Use the current branch to compare:

1. Existing `TEON + Muon + AdamW` baseline.
2. `TEON + Muon + SAGE` with unchanged `learning_rate`.
3. `TEON + Muon + SAGE` with a higher fallback LR sweep.

For this repo, the most important comparison is early loss and stability on the same branch/runtime setup, because the user already treats a clearly worse first ~100 steps as a practical stop signal for optimizer ablations.
