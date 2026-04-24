# nash-llm

Research LLM framework for pretraining and fine-tuning GPT-style transformers, inspired by nanoGPT.

This project targets single-NVIDIA-GPU CUDA training and uses `uv` for dependency management.

## Requirements

- Ubuntu 22.04 or similar Linux instance
- NVIDIA GPU with working drivers
- Python 3.10+
- Enough disk for datasets and checkpoints

Notes:

- Training is CUDA-oriented. Verify `nvidia-smi` works before starting.
- Default precision is `bf16`. If your GPU does not support bf16, use `--precision fp16`.
- This repo uses `uv`, not `pip`.
- The current pretraining configs are tuned around an H100-class GPU. In particular, the default `batch_size` and `grad_accum_steps` values assume H100-level memory/performance and will likely need to be reduced on smaller GPUs.

## Step-by-Step: Run on a GPU Instance

### 1. Connect to the instance

```bash
ssh ubuntu@YOUR_INSTANCE_IP
```

### 2. Install system packages

```bash
sudo apt update
sudo apt install -y git curl build-essential
```

### 3. Verify GPU access

```bash
nvidia-smi
```

If this fails, fix the NVIDIA driver/CUDA setup first.

### 4. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

### 5. Clone the repository

```bash
git clone YOUR_REPO_URL nash-llm
cd nash-llm
```

### 6. Install Python dependencies

```bash
uv sync --group dev
```

### 7. Optional: configure Weights & Biases

Training configs enable wandb by default.

```bash
uv run wandb login
```

If you do not want wandb, disable it at runtime:

```bash
--wandb_enabled false
```

### 8. Prepare a dataset

Smallest/easiest first run:

```bash
uv run python scripts/prepare_data.py --dataset tinystories_10M
```

This writes tokenized shards under:

```text
datasets/tokenized/tinystories_10M/
```

Available dataset prep commands:

```bash
uv run python scripts/prepare_data.py --dataset tinystories_10M
uv run python scripts/prepare_data.py --dataset openwebtext_100M
uv run python scripts/prepare_data.py --dataset fineweb_1B
uv run python scripts/prepare_data.py --dataset fineweb_2_5B
```

### 9. Start training

Recommended first real run:

```bash
uv run python scripts/train.py --config configs/pretrain_small.yaml
```

If your GPU does not support bf16:

```bash
uv run python scripts/train.py --config configs/pretrain_small.yaml --precision fp16
```

If you want to disable wandb:

```bash
uv run python scripts/train.py --config configs/pretrain_small.yaml --wandb_enabled false
```

### 10. Run a short smoke test first

Before a long training run, do a quick launch:

```bash
uv run python scripts/train.py \
  --config configs/pretrain_small.yaml \
  --max_steps 50 \
  --checkpoint_interval 25 \
  --eval_interval 10 \
  --wandb_enabled false
```

### 11. Check checkpoints

Training writes checkpoints to:

```text
checkpoints/
```

Typical outputs:

- `checkpoints/best.pt`
- `checkpoints/step_*.pt`

### 12. Resume training

```bash
uv run python scripts/train.py \
  --config configs/pretrain_small.yaml \
  --resume checkpoints/best.pt
```

### 13. Generate text from a checkpoint

```bash
uv run python scripts/generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "Once upon a time"
```

### 14. Evaluate a checkpoint

```bash
uv run python scripts/evaluate.py \
  --checkpoint checkpoints/best.pt \
  --data_dir datasets/tokenized/tinystories_10M
```

### 15. Run training in the background

```bash
nohup uv run python scripts/train.py \
  --config configs/pretrain_small.yaml \
  --wandb_enabled false \
  > train.log 2>&1 &
tail -f train.log
```

## Training Configs Present in This Repo

These config files currently exist:

- `configs/pretrain_small.yaml`
- `configs/pretrain_100m.yaml`
- `configs/pretrain_1b.yaml`
- `configs/pretrain_chinchilla.yaml`
- `configs/sft.yaml`

`configs/pretrain_debug.yaml` is referenced in some project notes, but it is not present in the repo right now.

## Useful Commands

### Install dependencies

```bash
uv sync --group dev
```

### Train

```bash
uv run python scripts/train.py --config configs/pretrain_small.yaml
uv run python scripts/train.py --config configs/pretrain_100m.yaml
uv run python scripts/train.py --config configs/pretrain_1b.yaml
uv run python scripts/train.py --config configs/pretrain_chinchilla.yaml
```

### CLI overrides

Short keys auto-resolve, so both simple and qualified overrides are supported:

```bash
uv run python scripts/train.py --config configs/pretrain_small.yaml --max_steps 50 --learning_rate 1e-3
uv run python scripts/train.py --config configs/pretrain_small.yaml --muon_lr 0.03
uv run python scripts/train.py --config configs/pretrain_small.yaml --n_kv_heads 4
```

`model.n_kv_heads` defaults to `model.n_heads`, which keeps standard multi-head attention. Set a smaller value such as `4` or `1` to enable grouped-query attention or multi-query attention.

### Generate

```bash
uv run python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "The"
```

### Evaluate

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --data_dir datasets/tokenized/tinystories_10M
```

### Plot / compare wandb runs

```bash
uv run python scripts/plot_metrics.py --run_id <wandb_run_id>
uv run python scripts/compare_runs.py --run_ids <id1> <id2>
```

## Project Layout

```text
nash_llm/
  config/        Config dataclasses and override loading
  data/          Tokenizer and dataset loaders
  eval/          Evaluation and text generation helpers
  model/         GPT model, attention, and layers
  optim/         Muon/TEON and SAGE optimizer setup
  training/      Trainer, scheduler, checkpoints
scripts/
  prepare_data.py
  train.py
  generate.py
  evaluate.py
configs/
  pretrain_small.yaml
  pretrain_100m.yaml
  pretrain_1b.yaml
  pretrain_chinchilla.yaml
  sft.yaml
```

## Notes

- Tokenizer: GPT-2 via `tiktoken` (`vocab_size=50257`)
- Tokenized datasets are written under `datasets/`
- Checkpoints are written under `checkpoints/`
- Mixed precision supports `bf16` and `fp16`
