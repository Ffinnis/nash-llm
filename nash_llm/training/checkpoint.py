import os
import torch
import torch.nn as nn
from dataclasses import asdict
from typing import Any


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer], step: int, config: Any, metrics: dict | None = None, wandb_run_id: str | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if isinstance(optimizer, list):
        opt_state = [opt.state_dict() for opt in optimizer]
    else:
        opt_state = optimizer.state_dict()
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt_state,
        "config": asdict(config),
        "metrics": metrics or {},
        "wandb_run_id": wandb_run_id,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = None) -> dict:
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        saved_state = checkpoint["optimizer_state_dict"]
        if isinstance(optimizer, list) and isinstance(saved_state, list):
            for opt, state in zip(optimizer, saved_state):
                opt.load_state_dict(state)
        elif not isinstance(optimizer, list) and not isinstance(saved_state, list):
            optimizer.load_state_dict(saved_state)
    return checkpoint
