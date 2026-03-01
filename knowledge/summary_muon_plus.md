# Muon+: Towards Better Muon via One Additional Normalization Step

**Paper**: [arXiv:2602.21545](https://arxiv.org/abs/2602.21545)
**Authors**: Ruijie Zhang, Yequan Zhao, Ziyue Liu, Zhengyang Wang, Zheng Zhang (UC Santa Barbara)
**Code**: https://github.com/K1seki221/MuonPlus

## Core Idea

Muon+ adds **one normalization step after orthogonalization** in the Muon update rule. That's it — the entire contribution is a single line of code.

### Standard Muon Update:
```
M_t = μ * M_{t-1} + (1 - μ) * G_t
O_t = Ortho(M_t)
W_t = W_{t-1} - lr * sqrt(m/n) * O_t
```

### Muon+ Update:
```
M_t = μ * M_{t-1} + (1 - μ) * G_t
O_t = Norm_d(Ortho(M_t))    # <-- one extra normalization
W_t = W_{t-1} - lr * sqrt(m/n) * O_t
```

## Normalization Directions

Four variants studied:

| Direction | Operation |
|-----------|-----------|
| `col` | L2-normalize each column: `X / sqrt(sum(X^2, dim=0))` |
| `row` | L2-normalize each row: `X / sqrt(sum(X^2, dim=1))` |
| `col_row` | Column-normalize, then row-normalize |
| `row_col` | Row-normalize, then column-normalize |

### Python Implementation (from paper):
```python
def norm_dir(X, d="col", eps=1e-8):
    if d == "col":
        denom = (X.square().sum(dim=-2, keepdim=True) + eps).sqrt()
        return X / denom
    if d == "row":
        denom = (X.square().sum(dim=-1, keepdim=True) + eps).sqrt()
        return X / denom
    if d == "col_row":
        return norm_dir(norm_dir(X, "col", eps), "row", eps)
    if d == "row_col":
        return norm_dir(norm_dir(X, "row", eps), "col", eps)
```

## Key Results

### GPT Models (FineWeb, Chinchilla-optimal T2P ≈ 20)
| Model | Params | Muon PPL | Muon+ PPL | Improvement |
|-------|--------|----------|-----------|-------------|
| GPT-Small | 124M | 29.66 | 27.64 | **-2.02** |
| GPT-Base | 362M | 21.70 | 19.98 | **-1.72** |
| GPT-Large | 774M | 17.82 | 16.91 | **-0.91** |

### LLaMA Models (FineWeb, Chinchilla-optimal)
| Model | Params | AdamW | Muon | Muon+ | Improvement |
|-------|--------|-------|------|-------|-------------|
| 60M | 58M | 33.10 | 25.75 | 25.25 | **-0.50** |
| 130M | 134M | 23.64 | 19.06 | 18.65 | **-0.41** |
| 350M | 368M | 16.18 | 14.02 | 13.41 | **-0.61** |
| 1B | 1339M | 14.38 | 10.68 | 10.31 | **-0.37** |

### Overtraining (T2P ≈ 200)
| Model | Tokens | Muon | Muon+ | Improvement |
|-------|--------|------|-------|-------------|
| GPT-Base 362M | 72B | 16.97 | 15.84 | **-1.13** |
| LLaMA-350M | 72B | 11.48 | 11.03 | **-0.45** |

## Key Findings

1. **`col_row` and `row_col` perform best** — nearly identical results, both better than single-direction normalization
2. **`row` normalization > `col` normalization** — consistent asymmetry across all scales
3. **No LR retuning needed** — optimal LR for Muon+ is comparable to Muon
4. **More stable at large LRs** — Muon+ degrades more gracefully with suboptimal learning rates
5. **Orthogonalization-agnostic** — works equally well with Jordan, You, and PolarExpress methods
6. **Consistent across architectures** — GPT (pre-RMSNorm) and LLaMA (RMSNorm + SwiGLU) both benefit
7. **Scales to long training** — benefit persists at T2P ≈ 200 (industrial level)

## Ablation: Muon+ vs NorMuon

The paper shows that NorMuon's improvement comes primarily from the normalization, **not** the second-moment adaptation:

| Method | GPT-Small | GPT-Base |
|--------|-----------|----------|
| Muon | 29.66 | 21.70 |
| **Muon+ (norm only)** | **27.91** | **19.98** |
| NorMuon (β₂=0, norm only) | 28.29 | 20.67 |
| NorMuon (β₂=0.95, full) | 28.42 | 20.72 |

Muon+ is simpler AND better than NorMuon.

---

## Integration Analysis: nash-llm TEON Optimizer

### Current Code (`nash_llm/optim/muon.py`)

The optimizer has two paths where normalization should be inserted:

#### 1. MUON path (batched per-shape, line ~159-164):
```python
bufs = torch.stack([self.state[p]["momentum_buffer"] for p in params])
ortho = orthogonalize(bufs, ns_steps)
# INSERT: ortho = norm_dir(ortho, d=norm_d)
scale = (m / n) ** 0.5
p.data.add_(ortho[i], alpha=-lr * scale)
```

#### 2. TEON path (cross-layer stacking, line ~205-212):
```python
Q_batch = orthogonalize(Z_batch, ns_steps)
# INSERT: Q_batch = norm_dir(Q_batch, d=norm_d)
ortho_slices = Q_batch[gi].split(n, dim=1)
param.data.add_(ortho_slices[i], alpha=-lr * scale)
```

#### 3. Fallback paths (per-param MUON, lines ~171, ~220):
```python
o = orthogonalize(buf, ns_steps)
# INSERT: o = norm_dir(o, d=norm_d)
```

### Implementation Plan

**Minimal changes required:**

1. **Add `norm_dir()` function** to `muon.py` (~10 lines)
2. **Add `muon_norm_dir` config field** to `TrainConfig` (default `"none"` for backward compatibility)
3. **Apply normalization after every `orthogonalize()` call** in `Muon.step()` (4 insertion points)
4. **Pass `norm_dir` as constructor parameter** to `Muon.__init__`

### Config Addition (`config.py`):
```python
muon_norm_dir: str = "none"  # "none", "col", "row", "col_row", "row_col"
```

### Recommended Default for nash-llm
Based on paper results for GPT-Small (124M, which matches our model):
- **Best direction**: `col_row` (27.64 PPL, vs 29.66 baseline)
- **Best LR**: 0.01 (vs current default 0.02 — but 0.02 also works well with Muon+)
- **No other hyperparameter changes needed**

### Considerations for TEON Specifically

The paper only evaluates Muon+ with standard per-layer Muon, **not with TEON's cross-layer stacking**. However:

1. The normalization is orthogonalization-agnostic (proven with Jordan/You/PolarExpress)
2. TEON stacking produces the same kind of orthogonalized matrix UV^T — normalization should apply equally
3. For TEON's stacked matrix `[M^(1)|M^(2)|...|M^(K)]`, normalization can be applied either:
   - **Before splitting**: normalize the full stacked Q matrix → columns/rows span K layers
   - **After splitting**: normalize each per-layer slice O^(k) independently
   - The paper doesn't address this, but **before splitting** seems more natural since orthogonalization operates on the full stacked matrix

### Complexity Impact
- Negligible: one L2 norm reduction + elementwise division per parameter per step
- No new state buffers needed (unlike NorMuon's second-moment)
- Works with existing `torch.compile` and bfloat16

### Risk Assessment
- **Low risk**: completely backward-compatible with `norm_dir="none"` default
- **Easy to A/B test**: just toggle the config flag
- **Expected gain**: ~1-2 PPL improvement on GPT-Small 124M (based on paper Table 1)
