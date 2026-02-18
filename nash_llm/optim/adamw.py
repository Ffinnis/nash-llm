import torch
import torch.nn as nn
import inspect

from nash_llm.optim.muon import Muon


def configure_optimizers(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    ns_steps: int = 5,
    betas: tuple[float, float] = (0.9, 0.95),
    fused: bool | None = None,
) -> list[torch.optim.Optimizer]:
    """Create TEON+AdamW optimizer pair.

    Returns [Muon, AdamW]:
    - Muon: TEON cross-layer Q/K/V stacking (K=2) + per-layer ortho for out_proj, MLP
    - AdamW: embeddings, layer norms, biases, and remaining params
    """
    muon_params: list[nn.Parameter] = []  # per-layer ortho (out_proj, MLP)
    teon_groups: list[list[nn.Parameter]] = []  # cross-layer stacking (Q/K/V)
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []

    muon_param_ids: set[int] = set()

    muon_patterns = ("out_proj.weight", "fc1.weight", "fc2.weight")
    teon_patterns = ("q_proj.weight", "k_proj.weight", "v_proj.weight")
    router_patterns = ("router.weight", "router.bias")

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

    # Collect MUON per-layer params (out_proj, fc1, fc2) + remaining to AdamW
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in muon_param_ids:
            continue
        if any(pattern in name for pattern in router_patterns):
            adamw_no_decay.append(param)
            continue
        matched = False
        for pattern in muon_patterns:
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
