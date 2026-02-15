# Shallow-Wide Model Configs

## Goal

Create 3 shallow-wide model configurations at ~124M parameters to compare against the baseline 12L/768d GPT. Test whether wider, shallower models perform differently on the same data budget.

## Baseline

- `n_layers=12`, `n_heads=12`, `d_model=768`, `d_ff=3072` — ~124M params

## New configs

### A: pretrain_small_8L1024d.yaml — "Moderate wide"

- `n_layers=8`, `n_heads=16`, `d_model=1024`, `d_ff=4096` — ~126M params
- 33% fewer layers, 33% wider

### B: pretrain_small_6L1280d.yaml — "Wide"

- `n_layers=6`, `n_heads=16`, `d_model=1280`, `d_ff=5120` — ~127M params
- 50% fewer layers, 67% wider

### C: pretrain_small_4L1536d.yaml — "Ultra-wide"

- `n_layers=4`, `n_heads=16`, `d_model=1536`, `d_ff=6144` — ~122M params
- 67% fewer layers, 100% wider

## Training

All configs inherit training params from `pretrain_small.yaml` (same batch size, LR, token budget, etc.). Only `model` section differs.
