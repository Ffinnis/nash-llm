import torch
import torch.nn as nn


def configure_optimizer(model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
