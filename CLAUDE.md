# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nash-llm is a research LLM framework for pretraining and fine-tuning GPT-style transformers, inspired by nanoGPT. It targets single NVIDIA GPU (CUDA) training with configurable mixed precision (`bf16`/`fp16`, default `bf16`).

## Commands

```bash
# Install (uses uv, not pip)
uv sync --group dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_transformer.py

# Run a specific test
uv run pytest tests/test_transformer.py::test_gpt_forward -v

# Prepare data (encode UTF-8 byte streams into binary shards)
uv run python scripts/prepare_data.py --dataset tinystories_10M
uv run python scripts/prepare_data.py --dataset openwebtext_100M
uv run python scripts/prepare_data.py --dataset fineweb_1B
uv run python scripts/prepare_data.py --dataset fineweb_2_5B

# Train
# pretrain_small.yaml: 104M MEGABYTE-style byte-level model on 10M GPT2-token-equivalent data
uv run python scripts/train.py --config configs/pretrain_small.yaml
# pretrain_100m.yaml: same model on 100M GPT2-token-equivalent data
uv run python scripts/train.py --config configs/pretrain_100m.yaml
# pretrain_1b.yaml: 104M MEGABYTE-style byte-level model on 1B GPT2-token-equivalent fineweb data
uv run python scripts/train.py --config configs/pretrain_1b.yaml
# pretrain_chinchilla.yaml: 104M MEGABYTE-style byte-level model on 2.5B GPT2-token-equivalent data
uv run python scripts/train.py --config configs/pretrain_chinchilla.yaml
# pretrain_debug.yaml: tiny debug preset
uv run python scripts/train.py --config configs/pretrain_debug.yaml

# CLI overrides (short keys auto-resolve to their section)
uv run python scripts/train.py --config configs/pretrain_small.yaml --max_steps 50 --learning_rate 1e-3
uv run python scripts/train.py --config configs/pretrain_small.yaml --muon_lr 0.03

# Generate text from checkpoint
uv run python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"

# Evaluate checkpoint
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --data_dir datasets/bytes/tinystories_10M

# Plot metrics / compare runs (wandb)
uv run python scripts/plot_metrics.py --run_id <wandb_run_id>
uv run python scripts/compare_runs.py --run_ids <id1> <id2>
```

## Architecture

### Config system (3-level cascade)

`NashConfig` is composed of `ModelConfig`, `TrainConfig`, `DataConfig`, `MetricsConfig` dataclasses. Values cascade: **code defaults → YAML file → CLI overrides**. CLI override keys can be short (`--max_steps 50`) or qualified (`--train.max_steps 50`); short keys are auto-resolved via `_resolve_short_key()`.

### Model

Decoder-only GPT with pre-RMSNorm, GELU activation, and weight tying (`lm_head.weight = token_emb.weight`) when `byte_patch_size=1`. The MEGABYTE-style byte-level default config is 12 global layers, 2 local byte-decoder layers, 12 heads, 768 d_model, `vocab_size=259`, `byte_patch_size=4`, and 104,350,976 params. The causal mask is registered as a buffer in `MultiHeadAttention`.

`GPT.forward(idx, targets)` returns logits when targets is None, or `(logits, loss)` when targets are provided.

### Data pipeline

`scripts/prepare_data.py` downloads from HuggingFace, encodes raw UTF-8 bytes by default, appends byte-level EOS, and writes numpy binary shards to `datasets/bytes/<name>/`. `PretrainDataset` uses `np.memmap` to load shards with the dtype recorded in `meta.json` and serves `(x, y)` pairs of `seq_len` ids. Legacy GPT-2 tiktoken encoding is still available with `data.representation=tiktoken` / `--representation tiktoken`. `SFTDataset` reads JSONL with `{"prompt", "completion"}` fields and returns `(input_ids, targets, loss_mask)` where `loss_mask` masks out the prompt portion.

### Training

`Trainer` orchestrates the loop: AMP autocast (`bf16`/`fp16`), gradient accumulation (`grad_accum_steps`), cosine LR schedule with linear warmup, gradient clipping, periodic eval (val_loss, accuracy, perplexity), checkpointing, and wandb logging.

`configure_optimizers()` creates a [Muon, SAGE] optimizer pair using TEON: cross-layer Q/K/V stacking (K=2) + per-layer MUON (out_proj, MLP) + SAGE (embeddings, normalization params, biases, and remaining params). Polar Express (Amsel et al., 2025) orthogonalization with pre-computed degree-5 coefficients. Attention uses separate `q_proj`, `k_proj`, `v_proj` projections. Config fields: `muon_lr` (default 0.02), `muon_momentum` (0.95), `ns_steps` (5), `sage_beta1` (0.9), `sage_beta2` (0.99), `sage_eps` (1e-8).

### Rules

- **Do not run tests locally.** The user will run them manually.

### Key conventions

- Representation: tokenizer-free UTF-8 bytes by default (vocab_size=259: 256 byte ids plus EOS/PAD/BOS)
- Data storage: `datasets/` (gitignored), checkpoints in `checkpoints/` (gitignored)
- Metrics: wandb (configurable via `metrics.wandb_enabled`)
- Package manager: **uv** (not pip) — uses `[dependency-groups]` in pyproject.toml
- Python ≥ 3.10 required (uses `X | Y` union types)
