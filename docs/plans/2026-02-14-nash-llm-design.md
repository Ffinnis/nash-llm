# Nash-LLM: Design Document

Research LLM framework for pretraining and fine-tuning transformer models, inspired by nanoGPT.

## Decisions

- **Hardware**: Single NVIDIA GPU (CUDA)
- **Language**: English datasets
- **Logging**: Weights & Biases
- **Config**: YAML + dataclasses + CLI overrides
- **Tokenizer**: GPT-2 via tiktoken (50257 tokens)
- **Architecture approach**: Modular sub-packages (Approach B)

## Project Structure

```
nash-llm/
├── nash_llm/
│   ├── model/
│   │   ├── transformer.py       # GPT model: embeddings, transformer blocks, lm_head
│   │   ├── attention.py         # MultiHeadAttention (causal)
│   │   └── layers.py            # FeedForward, LayerNorm, residual connections
│   ├── optim/
│   │   └── adamw.py             # AdamW with weight decay splitting
│   ├── data/
│   │   ├── dataset.py           # Map-style Dataset for pretrain (memory-mapped shards)
│   │   ├── tokenizer.py         # Wrapper over tiktoken GPT-2
│   │   └── sft_dataset.py       # Dataset for SFT/fine-tuning (JSONL format)
│   ├── training/
│   │   ├── trainer.py           # Train loop: forward, backward, logging, checkpointing
│   │   ├── scheduler.py         # LR schedulers (cosine with warmup)
│   │   └── checkpoint.py        # Save/load checkpoints
│   ├── config/
│   │   └── config.py            # Dataclasses: ModelConfig, TrainConfig, DataConfig, MetricsConfig
│   ├── metrics/
│   │   └── logger.py            # WandB logger with configurable metrics
│   └── eval/
│       ├── evaluate.py          # val_loss, accuracy, perplexity
│       └── generate.py          # Text generation (top-k, top-p, temperature)
├── scripts/
│   ├── train.py                 # Entry point: config + CLI overrides -> Trainer
│   ├── evaluate.py              # Evaluate checkpoint on val set
│   ├── generate.py              # Generate text from checkpoint
│   ├── prepare_data.py          # Download + tokenize dataset -> binary shards
│   ├── plot_metrics.py          # Plot training metrics from wandb
│   └── compare_runs.py          # Compare multiple training runs
├── configs/
│   ├── pretrain_100m.yaml       # 100M model preset
│   ├── pretrain_small.yaml      # Small model for debugging
│   └── sft.yaml                 # SFT/fine-tuning preset
├── datasets/                    # Data files (.gitignore)
│   ├── raw/
│   └── tokenized/
├── checkpoints/                 # Checkpoints (.gitignore)
└── pyproject.toml
```

## Model Architecture (100M parameters)

Decoder-only GPT-style transformer.

| Parameter | Value |
|-----------|-------|
| `n_layers` | 12 |
| `n_heads` | 12 |
| `d_model` | 768 |
| `d_ff` | 3072 (4 * d_model) |
| `vocab_size` | 50257 |
| `max_seq_len` | 1024 |
| `dropout` | 0.1 |

~124M parameters (GPT-2 small equivalent).

Key choices:
- **Position encoding**: Learned positional embeddings
- **Normalization**: Pre-LayerNorm (before attention/FFN)
- **Activation**: GELU
- **Weight tying**: Embedding and lm_head share weights

## Configuration System

Three levels of configuration (by priority):

1. **Defaults** in dataclasses
2. **YAML file** — presets
3. **CLI flags** — override at launch

Dataclasses:
- `ModelConfig` — architecture parameters
- `TrainConfig` — batch_size, lr, weight_decay, max_steps, warmup, grad_clip, eval/checkpoint intervals
- `DataConfig` — dataset name, size, tokenized_dir
- `MetricsConfig` — wandb_project, log_interval, list of metrics to track

CLI parsing via argparse, auto-generated from dataclass fields.

Example:
```bash
python scripts/train.py --config configs/pretrain_100m.yaml --learning_rate 1e-4 --batch_size 32
```

## Data Pipeline

### Sources

| Size | Dataset | Source |
|------|---------|--------|
| 10M tokens | TinyStories subset | HuggingFace |
| 100M tokens | OpenWebText subset | HuggingFace |
| 1B tokens | FineWeb-Edu sample | HuggingFace |

### Preparation (`scripts/prepare_data.py`)

Pipeline: Download -> Tokenize (tiktoken GPT-2) -> Binary shards (numpy memmap, uint16)

Split: 99% train / 1% val.

Output format:
```
datasets/tokenized/openwebtext_100M/
├── train_000.bin
├── train_001.bin
├── val_000.bin
└── meta.json        # {vocab_size, total_tokens, shard_size, split_ratio}
```

### Pretrain Dataset

Map-style `Dataset` over numpy memmap. `__getitem__` returns a window of `max_seq_len` tokens. Shuffling via `DataLoader(shuffle=True)`.

### SFT Dataset

JSONL format: `{"prompt": "...", "completion": "..."}`. Tokenized on-the-fly. Loss masked on prompt tokens.

## Training

- Mixed precision via `torch.amp` (float16)
- Gradient accumulation for effective larger batches
- Cosine LR schedule with linear warmup
- Gradient clipping (default 1.0)

### Checkpointing

Saves: step, model/optimizer/scheduler state dicts, config, metrics, wandb_run_id.
Auto-save by interval + best model by val_loss.

### SFT / Fine-tuning

Same Trainer, different config and dataset. Loads pretrained checkpoint, lower LR (1e-5), early stopping on val_loss.

## Metrics and Experiment Comparison

### Logging

WandB with configurable metrics: val_loss, accuracy, perplexity (auto-derived from val_loss).

### Plotting (`scripts/plot_metrics.py`)

Pulls data from wandb API, renders via matplotlib.

```bash
python scripts/plot_metrics.py --run_id abc123 --metrics val_loss,accuracy --output plots/run1.png
```

### Comparison (`scripts/compare_runs.py`)

```bash
python scripts/compare_runs.py --runs abc123,def456 --metrics val_loss
```

Outputs: overlaid metric plots, final metrics table, config diff between runs.

## Dependencies

```
torch >= 2.0
tiktoken
wandb
numpy
matplotlib
datasets
pyyaml
```
