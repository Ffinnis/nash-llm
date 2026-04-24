import torch
import torch.nn as nn
from torch.optim import Optimizer

from nash_llm.optim.muon import Muon


class Sage(Optimizer):
    """SAGE optimizer for embeddings and 1D parameters.

    This follows the paper's Lion-style sign direction with an O(d) adaptive
    damper for 2D embedding-like parameters and an elementwise variant for 1D
    tensors such as norms and biases.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

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

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("SAGE does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    if param.ndim > 1:
                        state["s_avg"] = torch.zeros_like(
                            param.mean(dim=0, keepdim=True),
                            memory_format=torch.preserve_format,
                        )
                    else:
                        state["s_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                s_avg = state["s_avg"]

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                grad_abs = grad.abs()
                if param.ndim > 1:
                    s_t = grad_abs.mean(dim=0, keepdim=True)
                else:
                    s_t = grad_abs

                s_avg.mul_(beta2).add_(s_t, alpha=1.0 - beta2)
                bias_correction2 = 1.0 - beta2 ** state["step"]
                s_avg_corrected = s_avg / bias_correction2

                s_rms = torch.sqrt(torch.mean(s_avg_corrected.square()))
                ema_damper = s_rms / (s_avg_corrected + eps)
                step_scale = torch.clamp(ema_damper, max=1.0)

                s_t_rms = torch.sqrt(torch.mean(s_t.square()))
                instant_damper = s_t_rms / (s_t + eps)
                final_scale = torch.minimum(step_scale, instant_damper)

                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                update.mul_(final_scale)
                param.add_(update, alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


def configure_optimizers(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    ns_steps: int = 5,
    sage_betas: tuple[float, float] = (0.9, 0.99),
    sage_eps: float = 1e-8,
) -> list[torch.optim.Optimizer]:
    """Create the optimizer pair used for training.

    Returns [Muon, Sage]:
    - Muon: TEON cross-layer Q/K/V stacking (K=2) + per-layer ortho for
      out_proj and MLP weights.
    - Sage: embeddings, normalization params, biases, and remaining params.
    """

    muon_params: list[nn.Parameter] = []
    teon_groups: list[list[nn.Parameter]] = []
    sage_decay: list[nn.Parameter] = []
    sage_no_decay: list[nn.Parameter] = []

    muon_param_ids: set[int] = set()

    muon_patterns = ("out_proj.weight", "fc1.weight", "fc_gate.weight", "fc2.weight")
    teon_patterns = ("q_proj.weight", "k_proj.weight", "v_proj.weight")

    teon_by_type: dict[str, list[nn.Parameter]] = {pattern: [] for pattern in teon_patterns}
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
        for index in range(num_groups):
            teon_groups.append(params[index * K : (index + 1) * K])
        remainder_start = num_groups * K
        for param in params[remainder_start:]:
            muon_params.append(param)

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
                sage_no_decay.append(param)
            else:
                sage_decay.append(param)

    muon_opt = Muon(
        muon_params=muon_params,
        teon_params=teon_groups,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=weight_decay,
        ns_steps=ns_steps,
    )

    sage_groups = []
    if sage_decay:
        sage_groups.append({"params": sage_decay, "weight_decay": weight_decay})
    if sage_no_decay:
        sage_groups.append({"params": sage_no_decay, "weight_decay": 0.0})

    sage_opt = Sage(sage_groups, lr=lr, betas=sage_betas, eps=sage_eps)

    return [muon_opt, sage_opt]
