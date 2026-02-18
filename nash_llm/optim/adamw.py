import torch
import torch.nn as nn

from nash_llm.optim.muon import Muon


class NesterovAdamW(torch.optim.AdamW):
    """AdamW with AdaPlus-style Nesterov look-ahead first moment."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group.get("maximize", False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("NesterovAdamW does not support sparse gradients")
                if maximize:
                    grad = -grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step = state["step"]
                if isinstance(step, torch.Tensor):
                    step_value = int(step.item()) + 1
                    state["step"] = torch.tensor(step_value, device=step.device, dtype=step.dtype)
                else:
                    step_value = int(step) + 1
                    state["step"] = step_value

                # Step 1: decoupled weight decay (AdamW-style).
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # Step 2: first moment EMA.
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Step 3: AdamW second moment EMA of g^2 (not AdaBelief/AdaPlus belief term).
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Step 4: Nesterov look-ahead momentum.
                m_bar = exp_avg * beta1 + grad * (1.0 - beta1)

                # Step 5: bias correction.
                bias_correction1 = 1.0 - beta1**step_value
                bias_correction2 = 1.0 - beta2**step_value
                m_hat = m_bar / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Step 6: AdamW denominator with Nesterov numerator.
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss


def _split_weight_decay_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return decay_params, no_decay_params


def _build_nesterov_adamw(
    param_groups: list[dict[str, object]],
    lr: float,
    betas: tuple[float, float],
    fused: bool | None,
) -> NesterovAdamW:
    if fused:
        raise ValueError("Fused AdamW is unsupported for NesterovAdamW")
    return NesterovAdamW(param_groups, lr=lr, betas=betas)


def _make_param_groups(
    decay_params: list[nn.Parameter],
    no_decay_params: list[nn.Parameter],
    weight_decay: float,
    drop_empty: bool = False,
) -> list[dict[str, object]]:
    param_groups: list[dict[str, object]] = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if drop_empty:
        param_groups = [group for group in param_groups if group["params"]]
    return param_groups


def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    fused: bool | None = None,
) -> torch.optim.AdamW:
    decay_params, no_decay_params = _split_weight_decay_params(model)
    param_groups = _make_param_groups(decay_params, no_decay_params, weight_decay)
    return _build_nesterov_adamw(param_groups, lr=lr, betas=betas, fused=fused)


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
    muon_patterns = ("out_proj.weight", "fc1.weight", "fc2.weight")
    # TEON target patterns: q_proj, k_proj, v_proj weights (2D)
    teon_patterns = ("q_proj.weight", "k_proj.weight", "v_proj.weight")

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
    adamw_groups = _make_param_groups(adamw_decay, adamw_no_decay, weight_decay, drop_empty=True)
    adamw_opt = _build_nesterov_adamw(adamw_groups, lr=lr, betas=betas, fused=fused)

    return [muon_opt, adamw_opt]
