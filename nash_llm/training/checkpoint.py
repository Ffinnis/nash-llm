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
        if isinstance(optimizer, list):
            if isinstance(saved_state, list):
                if len(optimizer) != len(saved_state):
                    raise ValueError(
                        f"Optimizer count mismatch when loading checkpoint: "
                        f"runtime has {len(optimizer)} optimizer(s), checkpoint has {len(saved_state)}."
                    )
                for opt, state in zip(optimizer, saved_state):
                    opt.load_state_dict(state)
            else:
                # Backward compatibility: old checkpoints stored a single optimizer state dict.
                if len(optimizer) == 1:
                    optimizer[0].load_state_dict(saved_state)
                else:
                    raise ValueError(
                        "Checkpoint contains a single optimizer state, but runtime expects "
                        f"{len(optimizer)} optimizers."
                    )
        elif not isinstance(saved_state, list):
            optimizer.load_state_dict(saved_state)
        elif len(saved_state) == 1:
            optimizer.load_state_dict(saved_state[0])
        else:
            raise ValueError(
                "Checkpoint contains multiple optimizer states, but runtime expects a single optimizer."
            )
    return checkpoint
