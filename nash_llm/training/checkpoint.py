import os
import torch
import torch.nn as nn
from dataclasses import asdict
from typing import Any


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, config: Any, metrics: dict | None = None, wandb_run_id: str | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "metrics": metrics or {},
        "wandb_run_id": wandb_run_id,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
