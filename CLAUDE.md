# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nash-llm is a research LLM framework for pretraining and fine-tuning GPT-style transformers, inspired by nanoGPT. It targets single NVIDIA GPU (CUDA) training with mixed precision (float16).

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

# Prepare data (tokenize dataset into binary shards)
uv run python scripts/prepare_data.py --dataset tinystories_10M
uv run python scripts/prepare_data.py --dataset openwebtext_100M
uv run python scripts/prepare_data.py --dataset fineweb_1B

# Train
uv run python scripts/train.py --config configs/pretrain_small.yaml
uv run python scripts/train.py --config configs/pretrain_100m.yaml

# CLI overrides (short keys auto-resolve to their section)
uv run python scripts/train.py --config configs/pretrain_small.yaml --max_steps 50 --learning_rate 1e-3

# Generate text from checkpoint
uv run python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"

# Evaluate checkpoint
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --data_dir datasets/tokenized/tinystories_10M

# Plot metrics / compare runs (wandb)
uv run python scripts/plot_metrics.py --run_id <wandb_run_id>
uv run python scripts/compare_runs.py --run_ids <id1> <id2>
```

## Architecture

### Config system (3-level cascade)

`NashConfig` is composed of `ModelConfig`, `TrainConfig`, `DataConfig`, `MetricsConfig` dataclasses. Values cascade: **code defaults → YAML file → CLI overrides**. CLI override keys can be short (`--max_steps 50`) or qualified (`--train.max_steps 50`); short keys are auto-resolved via `_resolve_short_key()`.

### Model

Decoder-only GPT with pre-LayerNorm, GELU activation, and weight tying (`lm_head.weight = token_emb.weight`). The 100M config is 12 layers, 12 heads, 768 d_model (124M params). The causal mask is registered as a buffer in `MultiHeadAttention`.

`GPT.forward(idx, targets)` returns logits when targets is None, or `(logits, loss)` when targets are provided.

### Data pipeline

`scripts/prepare_data.py` downloads from HuggingFace, tokenizes with GPT-2 tiktoken, and writes numpy uint16 binary shards to `datasets/tokenized/<name>/`. `PretrainDataset` uses `np.memmap` to load shards and serves `(x, y)` pairs of `seq_len` tokens. `SFTDataset` reads JSONL with `{"prompt", "completion"}` fields and returns `(input_ids, targets, loss_mask)` where `loss_mask` masks out the prompt portion.

### Training

`Trainer` orchestrates the loop: AMP autocast, gradient accumulation (`grad_accum_steps`), cosine LR schedule with linear warmup, gradient clipping, periodic eval (val_loss, accuracy, perplexity), checkpointing, and wandb logging.

`configure_optimizer()` splits params into decay (2D+ tensors) and no-decay (biases, layernorm) groups for AdamW.

### Key conventions

- Tokenizer: GPT-2 via tiktoken (vocab_size=50257)
- Data storage: `datasets/` (gitignored), checkpoints in `checkpoints/` (gitignored)
- Metrics: wandb (configurable via `metrics.wandb_enabled`)
- Package manager: **uv** (not pip) — uses `[dependency-groups]` in pyproject.toml
- Python ≥ 3.10 required (uses `X | Y` union types)
