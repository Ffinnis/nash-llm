from __future__ import annotations

import torch
import torch.nn as nn
import inspect
from typing import TYPE_CHECKING

from nash_llm.optim.aro import ARO, LayerParamGroup

from typing import cast

if TYPE_CHECKING:
    from nash_llm.model import GPT
    from nash_llm.model.transformer import TransformerBlock


def configure_optimizers(
    model: GPT,
    lr: float,
    weight_decay: float,
    aro_lr: float = 0.02,
    aro_momentum: float = 0.95,
    sink_iters: int = 5,
    rms_target: float = 0.2,
    betas: tuple[float, float] = (0.9, 0.95),
    fused: bool | None = None,
) -> list[torch.optim.Optimizer]:
    """Create ARO+AdamW optimizer pair.

    Returns [ARO, AdamW]:
    - ARO: chain-coupled rotation sharing for all layer weight matrices
           (q/k/v_proj, out_proj, fc1, fc2 per layer)
    - AdamW: embeddings, layer norms, biases, and remaining params
    """
    layer_groups: list[LayerParamGroup] = []
    aro_param_ids: set[int] = set()

    # Build layer groups from model blocks
    for idx, module in enumerate(model.blocks):
        block = cast("TransformerBlock", module)
        consumers: list[nn.Parameter] = [
            block.attn.q_proj.weight,  # type: ignore[list-item]
            block.attn.k_proj.weight,  # type: ignore[list-item]
            block.attn.v_proj.weight,  # type: ignore[list-item]
            block.ff.fc1.weight,  # type: ignore[list-item]
        ]
        producers: list[nn.Parameter] = [
            block.attn.out_proj.weight,  # type: ignore[list-item]
            block.ff.fc2.weight,  # type: ignore[list-item]
        ]
        layer_groups.append(LayerParamGroup(idx, consumers, producers))
        for p in consumers + producers:
            aro_param_ids.add(id(p))

    # Everything else goes to AdamW
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in aro_param_ids:
            continue
        if param.ndim < 2:
            adamw_no_decay.append(param)
        else:
            adamw_decay.append(param)

    # Build ARO optimizer
    aro_opt = ARO(
        layer_groups=layer_groups,
        lr=aro_lr,
        momentum=aro_momentum,
        weight_decay=weight_decay,
        sink_iters=sink_iters,
        rms_target=rms_target,
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

    return [aro_opt, adamw_opt]
