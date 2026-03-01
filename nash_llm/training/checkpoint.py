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


def _validate_model_compatibility(checkpoint: dict[str, Any], model: nn.Module) -> None:
    ckpt_cfg = checkpoint.get("config", {})
    if not isinstance(ckpt_cfg, dict):
        return
    ckpt_model_cfg = ckpt_cfg.get("model", {})
    if not isinstance(ckpt_model_cfg, dict):
        return

    runtime_cfg = getattr(model, "config", None)
    if runtime_cfg is None:
        return

    ckpt_norm_type = ckpt_model_cfg.get("norm_type")
    runtime_norm_type = getattr(runtime_cfg, "norm_type", None)
    if ckpt_norm_type is not None and runtime_norm_type is not None and ckpt_norm_type != runtime_norm_type:
        raise ValueError(
            "Checkpoint model.norm_type mismatch: "
            f"checkpoint={ckpt_norm_type}, runtime={runtime_norm_type}. "
            "Use a checkpoint created with the same model.norm_type."
        )

    ckpt_tie = ckpt_model_cfg.get("tie_embeddings")
    runtime_tie = getattr(runtime_cfg, "tie_embeddings", None)
    if ckpt_tie is not None and runtime_tie is not None and ckpt_tie != runtime_tie:
        raise ValueError(
            "Checkpoint model.tie_embeddings mismatch: "
            f"checkpoint={ckpt_tie}, runtime={runtime_tie}. "
            "Use a checkpoint created with the same model.tie_embeddings setting."
        )


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = None) -> dict:
    checkpoint = torch.load(path, weights_only=False)
    _validate_model_compatibility(checkpoint, model)
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
