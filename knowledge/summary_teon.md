# TEON: Tensorized Orthonormalization Beyond Layer-Wise Muon

**Paper**: [arXiv:2601.23261](https://arxiv.org/abs/2601.23261)
**Authors**: Ruijie Zhang, Yequan Zhao, Ziyue Liu, Zhengyang Wang, Dongyang Li, Yupeng Su, Sijia Liu, Zheng Zhang (UC Santa Barbara, Michigan State University)
**Venue**: ICML 2026 submission

---

## Core Idea

TEON generalizes Muon from **layer-wise** gradient orthogonalization to **cross-layer tensor-level** orthogonalization. Instead of orthogonalizing each layer's gradient independently, TEON stacks gradient matrices from K consecutive layers of the same type into a 3D tensor, applies mode-1 matricization (horizontal concatenation), orthogonalizes the resulting wide matrix, and splits the result back into per-layer updates.

**Key insight**: Q, K, V projection gradients across consecutive layers share aligned top right singular vectors. TEON captures these cross-layer correlations, achieving up to sqrt(K) better convergence over Muon.

## Algorithm (Mode-1 Orthogonalization)

```
For each group of K consecutive same-type layers (e.g., Q projections):
  1. Compute momentum: M_t = μ * M_{t-1} + G_t
  2. Mode-1 matricization: Z = [M^(1) | M^(2) | ... | M^(K)] ∈ R^{m × nK}
  3. Orthogonalize: Q = Ortho(Z) = UV^T
  4. Split back: O^(k) = Q[:, (k-1)*n : k*n] for k=1..K
  5. Update: W^(k) = W^(k) - η * sqrt(m/n) * O^(k)
```

## Key Design Choices (from ablation studies)

| Decision | Best Choice | Reasoning |
|----------|------------|-----------|
| Matricization mode | Mode-1 | Top *right* singular vectors are aligned across Q/K/V layers |
| Group size K | K=2 | Larger K reduces alignment quality; K=2 is optimal |
| Which layers to stack | Q, K, V only | These share retrieval-oriented role; MLP layers are heterogeneous |
| Orthogonalization | PolarExpress | Most accurate approximate SVD; robust to aspect ratio |
| Scale factor | sqrt(m/n) | Based on individual layer dimensions (not stacked) |

## Theoretical Results

- **Convergence**: TEON achieves same or up to sqrt(K)x better convergence bound than Muon
- **Condition**: Maximum gain when top right singular vectors are perfectly aligned across layers
- **Framework**: Non-Euclidean Trust Region (NTR) formulation

## Experimental Results

Validated on GPT (124M-774M) and LLaMA (60M-1B) models on FineWeb:

| Model | Muon PPL | TEON PPL | Improvement |
|-------|----------|----------|-------------|
| GPT-Small (10B tokens) | 28.53 | 27.12 | -4.9% |
| GPT-Base (10B tokens) | 21.64 | 20.92 | -3.3% |
| GPT-Large (10B tokens) | 19.26 | 18.73 | -2.8% |
| LLaMA-130M (2.2B tokens) | 19.45 | 18.92 | -2.7% |
| LLaMA-1B (13.1B tokens) | 11.19 | 10.84 | -3.1% |

TEON and Muon have nearly identical per-step computational cost.

---

## Verification Against nash-llm Implementation

### Checklist: Paper vs Code

| Aspect | Paper | Code (`nash_llm/optim/muon.py`, `adamw.py`) | Status |
|--------|-------|----------------------------------------------|--------|
| Mode-1 matricization | Z = [M^(1) \| M^(2) \| ... \| M^(K)] | `torch.cat(momentums, dim=1)` | CORRECT |
| Inverse matricization | Split Q back into K slices | `Q_batch[gi].split(n, dim=1)` | CORRECT |
| K=2 consecutive layers | Stack same-type params from consecutive blocks | Groups `params[i*K : (i+1)*K]` per pattern | CORRECT |
| Only Q/K/V for TEON | Q,K,V projections stacked; O,MLP use per-layer Muon | `teon_patterns = ("q_proj.weight", "k_proj.weight", "v_proj.weight")` | CORRECT |
| Scale factor | sqrt(m/n) based on individual layer dims | `scale = (m / n) ** 0.5` | CORRECT |
| Momentum | M_t = μ M_{t-1} + G_t | `buf.mul_(mu).add_(grad, alpha=1-mu)` | EQUIVALENT (see note) |
| Polar Express ortho | Degree-5 polynomial iterations | `_polar_express_impl()` with pre-computed coefficients | CORRECT |
| AdamW for rest | Embeddings, norms, biases | 2D+ weights → decay, <2D → no decay | CORRECT |
| Remainder handling | Not specified (implicit: standard Muon) | Odd-layer-count remainder → per-layer Muon | REASONABLE |

### Notes on Differences

1. **Momentum formula**: Code uses `M_t = μ*M_{t-1} + (1-μ)*G_t` while the paper's Algorithm 1 uses `M_t = μ*M_{t-1} + G_t`. These are **equivalent** for Muon/TEON because `Ortho(αM) = Ortho(M)` for any scalar α > 0 — the orthogonalization only cares about direction, not magnitude.

2. **Double safety factor in Polar Express**: The code applies the safety factor **twice**:
   - In normalization: `X / (||X||_F * 1.01 + eps)` (line 40)
   - In coefficients: `(a/1.01, b/1.01³, c/1.01⁵)` (lines 20-23)

   This means the effective safety margin is ~1.01² ≈ 1.02 instead of 1.01. This is **not a correctness bug** (the polynomial still converges to UV^T), but it means slightly slower convergence per iteration. With 5 iterations this is unlikely to matter in practice. To fix, either:
   - Remove `* 1.01` from the normalization (line 40), OR
   - Use `_POLAR_EXPRESS_COEFFS` (raw) instead of `_POLAR_EXPRESS_COEFFS_SAFE`

3. **Batched orthogonalization**: The code groups same-shape params into batches for efficient GPU orthogonalization via `torch.stack()`. This is a performance optimization not described in the paper, but mathematically equivalent.

4. **Separate LR schedulers**: The code uses separate cosine schedulers for Muon and AdamW, matching the paper's practice of different learning rates for these optimizer groups.

### Overall Assessment

**The TEON implementation in nash-llm is correct and faithfully follows the paper's Algorithm 1 and recommended hyperparameter choices.** The only notable deviation is the double safety factor in Polar Express normalization, which is a minor performance issue, not a correctness bug.
