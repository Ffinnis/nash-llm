import torch
import torch.nn as nn
import inspect

from nash_llm.optim.muon import Muon


def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    fused: bool | None = None,
) -> torch.optim.AdamW:
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
    can_use_fused = torch.cuda.is_available() and "fused" in inspect.signature(torch.optim.AdamW).parameters
    if fused is None:
        if can_use_fused:
            adamw_kwargs["fused"] = True
    elif fused:
        if not can_use_fused:
            raise ValueError("Fused AdamW requested but not supported in this runtime")
        adamw_kwargs["fused"] = True
    return torch.optim.AdamW(param_groups, **adamw_kwargs)


def configure_optimizers(
    model: nn.Module,
    optimizer_type: str,
    lr: float,
    weight_decay: float,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    ns_steps: int = 5,
    betas: tuple[float, float] = (0.9, 0.95),
    fused: bool | None = None,
) -> list[torch.optim.Optimizer]:
    """Create optimizer(s) based on optimizer_type.

    Returns a list of optimizers:
    - "adamw": [AdamW] (all params)
    - "muon": [Muon, AdamW] (MUON for 2D attn+MLP weights, AdamW for rest)
    - "teon": [Muon, AdamW] (TEON for Q/K/V stacking + MUON for out_proj+MLP, AdamW for rest)
    """
    if optimizer_type == "adamw":
        return [configure_optimizer(model, lr, weight_decay, betas, fused)]

    # Collect params by role
    muon_params: list[nn.Parameter] = []  # per-layer ortho (out_proj, MLP)
    teon_groups: list[list[nn.Parameter]] = []  # cross-layer stacking (Q/K/V)
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []

    # Identify which params go to Muon vs AdamW
    muon_param_ids: set[int] = set()

    # MUON target patterns: out_proj, fc1, fc2 weights (2D)
    muon_patterns = {"out_proj.weight", "fc1.weight", "fc2.weight"}
    # TEON target patterns: q_proj, k_proj, v_proj weights (2D)
    teon_patterns = {"q_proj.weight", "k_proj.weight", "v_proj.weight"}

    if optimizer_type == "teon":
        # Build TEON groups: stack K=2 consecutive blocks for each of Q/K/V
        teon_by_type: dict[str, list[nn.Parameter]] = {p: [] for p in teon_patterns}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            for pattern in teon_patterns:
                if pattern in name:
                    teon_by_type[pattern].append(param)
                    muon_param_ids.add(id(param))
                    break

        K = 2
        for pattern in teon_patterns:
            params = teon_by_type[pattern]
            num_groups = len(params) // K
            for i in range(num_groups):
                teon_groups.append(params[i * K : (i + 1) * K])
            # Remainder (odd layer count) goes to muon_params as per-layer
            remainder_start = num_groups * K
            for p in params[remainder_start:]:
                muon_params.append(p)

    # Collect MUON per-layer params (out_proj, fc1, fc2)
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in muon_param_ids:
            continue
        matched = False
        for pattern in muon_patterns:
            if pattern in name:
                muon_params.append(param)
                muon_param_ids.add(id(param))
                matched = True
                break
        if matched:
            continue
        # For "muon" mode (no teon), Q/K/V also go to muon per-layer
        if optimizer_type == "muon":
            for pattern in teon_patterns:
                if pattern in name:
                    muon_params.append(param)
                    muon_param_ids.add(id(param))
                    matched = True
                    break
        if not matched:
            if param.ndim < 2:
                adamw_no_decay.append(param)
            else:
                adamw_decay.append(param)

    # Build Muon optimizer
    muon_opt = Muon(
        muon_params=muon_params,
        teon_params=teon_groups,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=weight_decay,
        ns_steps=ns_steps,
    )

    # Build AdamW for remaining params
    adamw_groups = []
    if adamw_decay:
        adamw_groups.append({"params": adamw_decay, "weight_decay": weight_decay})
    if adamw_no_decay:
        adamw_groups.append({"params": adamw_no_decay, "weight_decay": 0.0})

    adamw_kwargs: dict = {"lr": lr, "betas": betas}
    can_use_fused = torch.cuda.is_available() and "fused" in inspect.signature(torch.optim.AdamW).parameters
    if fused is None:
        if can_use_fused:
            adamw_kwargs["fused"] = True
    elif fused:
        if not can_use_fused:
            raise ValueError("Fused AdamW requested but not supported in this runtime")
        adamw_kwargs["fused"] = True

    adamw_opt = torch.optim.AdamW(adamw_groups, **adamw_kwargs)

    return [muon_opt, adamw_opt]
