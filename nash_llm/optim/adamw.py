import torch
import torch.nn as nn
import inspect

from nash_llm.optim.muon import Muon


def _configure_muon_optimizers(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    muon_lr: float,
    muon_momentum: float,
    ns_steps: int,
    betas: tuple[float, float],
    fused: bool | None,
) -> list[torch.optim.Optimizer]:
    """Create TEON+AdamW optimizer pair (original path)."""
    muon_params: list[nn.Parameter] = []  # per-layer ortho (out_proj, MLP)
    teon_groups: list[list[nn.Parameter]] = []  # cross-layer stacking (Q/K/V)
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []

    muon_param_ids: set[int] = set()

    muon_patterns = ("out_proj.weight", "fc1.weight", "fc2.weight")
    teon_patterns = ("q_proj.weight", "k_proj.weight", "v_proj.weight")

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

    return [muon_opt] + _build_adamw(adamw_decay, adamw_no_decay, lr, weight_decay, betas, fused)


def _configure_taro_optimizers(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    muon_lr: float,
    muon_momentum: float,
    ns_steps: int,
    betas: tuple[float, float],
    fused: bool | None,
) -> list[torch.optim.Optimizer]:
    """Create TARO+AdamW optimizer pair.

    TARO groups all 2D matrix weights by type, each with K=2 cross-layer stacking
    and independent rotation matrices: Q, K, V, O, UP, DOWN.
    AdamW handles embeddings, norms, biases.
    """
    from nash_llm.optim.taro import Taro

    taro_param_ids: set[int] = set()
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []

    # Collect params by type — separate Q/K/V for independent rotation matrices
    group_patterns: dict[str, tuple[str, ...]] = {
        "q": ("q_proj.weight",),
        "k": ("k_proj.weight",),
        "v": ("v_proj.weight",),
        "o": ("out_proj.weight",),
        "up": ("fc1.weight",),
        "down": ("fc2.weight",),
    }

    # Gather params by group type
    params_by_type: dict[str, list[nn.Parameter]] = {k: [] for k in group_patterns}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = False
        for group_type, patterns in group_patterns.items():
            for pattern in patterns:
                if pattern in name:
                    params_by_type[group_type].append(param)
                    taro_param_ids.add(id(param))
                    matched = True
                    break
            if matched:
                break
        if not matched:
            if param.ndim < 2:
                adamw_no_decay.append(param)
            else:
                adamw_decay.append(param)

    # Build TARO groups with K=2 cross-layer stacking
    K = 2
    taro_groups: list[tuple[str, list[list[nn.Parameter]]]] = []

    for group_type in ("q", "k", "v", "o", "up", "down"):
        params = params_by_type[group_type]
        if not params:
            continue

        blocks: list[list[nn.Parameter]] = []
        # All groups: simple K=2 cross-layer stacking
        num_groups = len(params) // K
        for i in range(num_groups):
            blocks.append(params[i * K : (i + 1) * K])
        remainder_start = num_groups * K
        if remainder_start < len(params):
            blocks.append(params[remainder_start:])

        taro_groups.append((group_type, blocks))

    # Sinkhorn is O(mn) per iter vs Polar Express O(m²n) — use more iterations
    taro_opt = Taro(
        taro_groups=taro_groups,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=weight_decay,
        sinkhorn_iters=ns_steps * 5,
    )

    return [taro_opt] + _build_adamw(adamw_decay, adamw_no_decay, lr, weight_decay, betas, fused)


def _build_adamw(
    adamw_decay: list[nn.Parameter],
    adamw_no_decay: list[nn.Parameter],
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    fused: bool | None,
) -> list[torch.optim.Optimizer]:
    """Build AdamW optimizer for remaining params."""
    adamw_groups = []
    if adamw_decay:
        adamw_groups.append({"params": adamw_decay, "weight_decay": weight_decay})
    if adamw_no_decay:
        adamw_groups.append({"params": adamw_no_decay, "weight_decay": 0.0})

    if not adamw_groups:
        return []

    adamw_kwargs: dict = {"lr": lr, "betas": betas}
    can_use_fused = torch.cuda.is_available() and "fused" in inspect.signature(torch.optim.AdamW).parameters
    if fused is None:
        if can_use_fused:
            adamw_kwargs["fused"] = True
    elif fused:
        if not can_use_fused:
            raise ValueError("Fused AdamW requested but not supported in this runtime")
        adamw_kwargs["fused"] = True

    return [torch.optim.AdamW(adamw_groups, **adamw_kwargs)]


def configure_optimizers(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    ns_steps: int = 5,
    betas: tuple[float, float] = (0.9, 0.95),
    fused: bool | None = None,
    optimizer: str = "muon",
) -> list[torch.optim.Optimizer]:
    """Create optimizer pair.

    Returns [Muon/Taro, AdamW]:
    - optimizer="muon": TEON cross-layer Q/K/V stacking + per-layer ortho for out_proj, MLP
    - optimizer="taro": TARO adaptive rotation for all 2D weights by symmetry group
    - AdamW: embeddings, layer norms, biases, and remaining params
    """
    if optimizer == "taro":
        return _configure_taro_optimizers(
            model, lr, weight_decay, muon_lr, muon_momentum, ns_steps, betas, fused,
        )
    return _configure_muon_optimizers(
        model, lr, weight_decay, muon_lr, muon_momentum, ns_steps, betas, fused,
    )
