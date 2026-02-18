# MoE Experiment Playbook (100M -> 1B)

## Goal
Compare dense baseline vs optional MoE under equal token budget and comparable step compute.

## Stage A (100M smoke)

### Dense baseline
```bash
uv run python scripts/train.py --config configs/pretrain_100m.yaml
```

### MoE run
```bash
uv run python scripts/train.py --config configs/pretrain_100m_moe.yaml
```

### Go / No-Go to Stage B
Proceed to 1B only if all are true:
- `val_loss` improves by at least 1% vs dense baseline
- `moe_dropped_frac < 0.02`
- throughput slowdown is no worse than 25%

## Stage B (1B main)

### Dense baseline
Use existing 1B-equivalent dense config for your dataset/token budget.

### MoE run
```bash
uv run python scripts/train.py --config configs/pretrain_1b_moe.yaml
```

### Success criteria
Treat MoE as successful if all are true:
- `val_loss` / perplexity improves by at least 2% vs dense baseline
- expert usage does not collapse (`moe_expert_entropy` remains stable, no single-expert dominance)
- wall-clock slowdown remains operationally acceptable
