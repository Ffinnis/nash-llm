"""Evaluate a checkpoint on validation data."""
import argparse
import math
import torch
from torch.utils.data import DataLoader
from nash_llm.config import load_config, NashConfig, ModelConfig
from nash_llm.model import GPT
from nash_llm.data.dataset import PretrainDataset
from nash_llm.training.checkpoint import load_checkpoint
from nash_llm.eval import compute_val_loss, compute_accuracy


def _resolve_max_batches(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_ckpt = torch.load(args.checkpoint, weights_only=False)
    if args.config:
        config = load_config(args.config)
    else:
        raw_config = raw_ckpt["config"]
        model_cfg = ModelConfig(**raw_config["model"])
        config = NashConfig(model=model_cfg)
        if "data" in raw_config:
            config.data = config.data.__class__(**{**vars(config.data), **raw_config["data"]})
        if "metrics" in raw_config:
            config.metrics = config.metrics.__class__(**{**vars(config.metrics), **raw_config["metrics"]})

    model = GPT(config.model).to(device)
    load_checkpoint(args.checkpoint, model)

    data_dir = args.data_dir or config.data.tokenized_dir
    val_dataset = PretrainDataset(data_dir, split="val", seq_len=config.model.max_seq_len)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    max_batches = _resolve_max_batches(args.max_batches)
    if max_batches is None:
        max_batches = _resolve_max_batches(config.metrics.final_eval_max_batches)

    val_loss = compute_val_loss(model, val_loader, max_batches=max_batches)
    accuracy = compute_accuracy(model, val_loader, max_batches=max_batches)
    perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")
    print(f"val_loss:    {val_loss:.4f}")
    print(f"accuracy:    {accuracy:.4f}")
    print(f"perplexity:  {perplexity:.2f}")

if __name__ == "__main__":
    main()
