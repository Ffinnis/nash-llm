import torch
import torch.nn as nn
import inspect


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
    adamw_kwargs = {"lr": lr, "betas": betas}
    if torch.cuda.is_available() and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        adamw_kwargs["fused"] = True
    return torch.optim.AdamW(param_groups, **adamw_kwargs)
