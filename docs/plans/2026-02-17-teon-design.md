# TEON + Polar Express Optimizer Integration

## Summary

Add MUON/TEON optimizer with Polar Express orthogonalization to nash-llm. TEON stacks gradients from K=2 consecutive transformer blocks and orthogonalizes them jointly, capturing cross-layer correlations. Two separate optimizers: Muon (TEON+MUON) for 2D weights, AdamW for the rest.

## Decision Record

| Decision | Choice | Rationale |
|----------|--------|-----------|
| QKV split | Refactor to separate q_proj, k_proj, v_proj | Required for per-projection TEON stacking |
| Architecture | Two separate optimizers (Muon + AdamW) | Clean separation, easier to test |
| Layer assignment | TEON: Q/K/V; MUON: out_proj+MLP; AdamW: rest | Per paper recommendations |
| LR | muon_lr=0.02 separate from adamw lr=3e-4 | Different optimal LRs per paper |
| Config | optimizer field in TrainConfig | Simple, backward-compatible |

## Changes

### 1. Refactor Attention (split QKV)

**File:** `nash_llm/model/attention.py`

Replace combined `self.qkv = nn.Linear(d_model, 3 * d_model)` with:
```python
self.q_proj = nn.Linear(d_model, d_model)
self.k_proj = nn.Linear(d_model, d_model)
self.v_proj = nn.Linear(d_model, d_model)
```

Forward: `q = self.q_proj(x)`, `k = self.k_proj(x)`, `v = self.v_proj(x)`.

Parameter count unchanged. Slight perf difference (3 matmuls vs 1).

### 2. Muon Optimizer

**New file:** `nash_llm/optim/muon.py`

Class `Muon(Optimizer)` handles two param types:
- **muon_params**: per-layer orthogonalization (out_proj, MLP weights)
- **teon_params**: groups of K params for cross-layer stacking (Q/K/V)

```python
Muon(
    muon_params: list[nn.Parameter],
    teon_params: list[list[nn.Parameter]],
    lr: float = 0.02,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    ns_steps: int = 5,
)
```

**step() logic:**
1. MUON params: momentum → orthogonalize(buf) → update with sqrt(m/n) scaling
2. TEON groups: momentum → cat(bufs, dim=1) → orthogonalize(concatenated) → split → update

**Orthogonalization:** Polar Express with pre-computed coefficients from paper. bfloat16 compute, safety factor, epsilon normalization. `@torch.compile` decorated.

### 3. Parameter Splitting

| Parameter pattern | Optimizer | Method |
|-------------------|-----------|--------|
| `blocks.{i}.attn.q_proj.weight` | Muon | TEON (K=2 stacking) |
| `blocks.{i}.attn.k_proj.weight` | Muon | TEON (K=2 stacking) |
| `blocks.{i}.attn.v_proj.weight` | Muon | TEON (K=2 stacking) |
| `blocks.{i}.attn.out_proj.weight` | Muon | MUON (per-layer) |
| `blocks.{i}.ff.fc1.weight` | Muon | MUON (per-layer) |
| `blocks.{i}.ff.fc2.weight` | Muon | MUON (per-layer) |
| token_emb, pos_emb, LayerNorm, biases | AdamW | Standard |

### 4. Config

TrainConfig additions:
```python
optimizer: str = "adamw"      # "adamw" | "muon" | "teon"
muon_lr: float = 0.02
muon_momentum: float = 0.95
ns_steps: int = 5
```

### 5. Trainer

- `self.optimizers: list[Optimizer]` (1 for adamw, 2 for muon/teon)
- zero_grad/step for all optimizers
- LR scheduler applies to both
- GradScaler support for both

### 6. Tests

- polar_express correctness vs exact SVD
- muon single step
- teon stacking/splitting
- attention split QKV equivalence
- configure_optimizer parameter assignment
