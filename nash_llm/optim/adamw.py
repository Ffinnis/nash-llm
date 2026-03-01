import inspect
import re

import torch
import torch.nn as nn

from nash_llm.optim.muon import Muon


_BLOCK_INDEX_RE = re.compile(r"^blocks\.(\d+)\.")


def _layer_index_from_name(name: str) -> int | None:
    match = _BLOCK_INDEX_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


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

    k = 2
    for pattern in teon_patterns:
        params = teon_by_type[pattern]
        num_groups = len(params) // k
        for i in range(num_groups):
            teon_groups.append(params[i * k : (i + 1) * k])
        remainder_start = num_groups * k
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
    taro_k: int,
    taro_sinkhorn_iters: int,
    taro_down_lr_mult: float,
) -> list[torch.optim.Optimizer]:
    """Create TARO+AdamW optimizer pair with symmetry-position grouping."""
    from nash_llm.optim.taro import Taro, TaroGroup, TaroParamRef

    if taro_k <= 0:
        raise ValueError(f"taro_k must be >= 1, got {taro_k}")
    if taro_sinkhorn_iters <= 0:
        raise ValueError(f"taro_sinkhorn_iters must be >= 1, got {taro_sinkhorn_iters}")

    q_by_layer: dict[int, nn.Parameter] = {}
    k_by_layer: dict[int, nn.Parameter] = {}
    v_by_layer: dict[int, nn.Parameter] = {}
    o_by_layer: dict[int, nn.Parameter] = {}
    up_by_layer: dict[int, nn.Parameter] = {}
    down_by_layer: dict[int, nn.Parameter] = {}

    taro_param_ids: set[int] = set()
    adamw_specs: list[tuple[nn.Parameter, float, float]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_idx = _layer_index_from_name(name)
        if layer_idx is not None:
            if name.endswith("attn.q_proj.weight"):
                q_by_layer[layer_idx] = param
                taro_param_ids.add(id(param))
                continue
            if name.endswith("attn.k_proj.weight"):
                k_by_layer[layer_idx] = param
                taro_param_ids.add(id(param))
                continue
            if name.endswith("attn.v_proj.weight"):
                v_by_layer[layer_idx] = param
                taro_param_ids.add(id(param))
                continue
            if name.endswith("attn.out_proj.weight"):
                o_by_layer[layer_idx] = param
                taro_param_ids.add(id(param))
                continue
            if name.endswith("ff.fc1.weight"):
                up_by_layer[layer_idx] = param
                taro_param_ids.add(id(param))
                continue
            if name.endswith("ff.fc2.weight"):
                down_by_layer[layer_idx] = param
                taro_param_ids.add(id(param))
                continue

        lr_mult = 1.0
        if name == "lm_head.weight":
            lr_mult = 0.5
        decay = weight_decay if param.ndim >= 2 else 0.0
        adamw_specs.append((param, lr_mult, decay))

    # Build TARO groups by symmetry position
    taro_groups: list[TaroGroup] = []
    layer_ids = sorted(set(q_by_layer) | set(k_by_layer) | set(v_by_layer) | set(o_by_layer) | set(up_by_layer) | set(down_by_layer))

    qkv_blocks: list[list[TaroParamRef]] = []
    o_up_blocks: list[list[TaroParamRef]] = []
    down_blocks: list[list[TaroParamRef]] = []

    for i in range(0, len(layer_ids), taro_k):
        chunk = layer_ids[i : i + taro_k]

        qkv_block: list[TaroParamRef] = []
        o_up_block: list[TaroParamRef] = []
        down_block: list[TaroParamRef] = []

        for layer_idx in chunk:
            if layer_idx in q_by_layer and layer_idx in k_by_layer and layer_idx in v_by_layer:
                qkv_block.extend([
                    TaroParamRef(q_by_layer[layer_idx], transpose_in_taro=False, lr_mult=1.0, group_name="qkv"),
                    TaroParamRef(k_by_layer[layer_idx], transpose_in_taro=False, lr_mult=1.0, group_name="qkv"),
                    TaroParamRef(v_by_layer[layer_idx], transpose_in_taro=False, lr_mult=1.0, group_name="qkv"),
                ])
            if layer_idx in o_by_layer:
                o_up_block.append(TaroParamRef(o_by_layer[layer_idx], transpose_in_taro=False, lr_mult=1.0, group_name="o_up"))
            if layer_idx in up_by_layer:
                # Canonical orientation aligns residual-stream dimension on rows.
                o_up_block.append(TaroParamRef(up_by_layer[layer_idx], transpose_in_taro=True, lr_mult=1.0, group_name="o_up"))
            if layer_idx in down_by_layer:
                down_block.append(
                    TaroParamRef(
                        down_by_layer[layer_idx],
                        transpose_in_taro=False,
                        lr_mult=taro_down_lr_mult,
                        group_name="down",
                    )
                )

        if qkv_block:
            qkv_blocks.append(qkv_block)
        if o_up_block:
            o_up_blocks.append(o_up_block)
        if down_block:
            down_blocks.append(down_block)

    if qkv_blocks:
        taro_groups.append(TaroGroup(group_name="qkv", blocks=qkv_blocks, lr_mult=1.0))
    if o_up_blocks:
        taro_groups.append(TaroGroup(group_name="o_up", blocks=o_up_blocks, lr_mult=1.0))
    if down_blocks:
        taro_groups.append(TaroGroup(group_name="down", blocks=down_blocks, lr_mult=taro_down_lr_mult))

    taro_opt = Taro(
        taro_groups=taro_groups,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=weight_decay,
        sinkhorn_iters=taro_sinkhorn_iters if taro_sinkhorn_iters > 0 else ns_steps,
    )

    return [taro_opt] + _build_adamw_with_lr_multipliers(adamw_specs, lr, betas, fused)


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


def _build_adamw_with_lr_multipliers(
    specs: list[tuple[nn.Parameter, float, float]],
    lr: float,
    betas: tuple[float, float],
    fused: bool | None,
) -> list[torch.optim.Optimizer]:
    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    for param, lr_mult, decay in specs:
        key = (lr_mult, decay)
        grouped.setdefault(key, []).append(param)

    adamw_groups = []
    for (lr_mult, decay), params in grouped.items():
        adamw_groups.append({
            "params": params,
            "weight_decay": decay,
            "lr": lr * lr_mult,
            "lr_mult": lr_mult,
        })

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
    taro_k: int = 2,
    taro_sinkhorn_iters: int = 5,
    taro_down_lr_mult: float = 0.5,
) -> list[torch.optim.Optimizer]:
    """Create optimizer pair.

    Returns [Muon/Taro, AdamW]:
    - optimizer="muon": TEON cross-layer Q/K/V stacking + per-layer ortho for out_proj, MLP
    - optimizer="taro": TARO adaptive rotation for symmetry-position groups
    - AdamW: embeddings, layer norms, biases, and remaining params
    """
    if optimizer == "taro":
        return _configure_taro_optimizers(
            model,
            lr,
            weight_decay,
            muon_lr,
            muon_momentum,
            ns_steps,
            betas,
            fused,
            taro_k=taro_k,
            taro_sinkhorn_iters=taro_sinkhorn_iters,
            taro_down_lr_mult=taro_down_lr_mult,
        )
    return _configure_muon_optimizers(
        model, lr, weight_decay, muon_lr, muon_momentum, ns_steps, betas, fused,
    )
