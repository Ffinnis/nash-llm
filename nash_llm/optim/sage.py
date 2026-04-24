import torch
import torch.nn as nn
from torch.optim import Optimizer

from nash_llm.optim.muon import Muon

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is Linux/CUDA-only in most local dev envs.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _sage_update_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        scale_ptr,
        n_elements: tl.constexpr,
        n_cols: tl.constexpr,
        lr,
        weight_decay,
        beta1,
        beta2,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements

        param = tl.load(param_ptr + offsets, mask=mask).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
        scale = tl.load(scale_ptr + (offsets % n_cols), mask=mask).to(tl.float32)

        param = param * (1.0 - lr * weight_decay)
        direction = exp_avg * beta1 + grad * (1.0 - beta1)
        signed = tl.where(direction > 0.0, 1.0, tl.where(direction < 0.0, -1.0, 0.0))
        param = param - lr * signed * scale
        exp_avg = exp_avg * beta2 + grad * (1.0 - beta2)

        tl.store(param_ptr + offsets, param, mask=mask)
        tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
else:
    _sage_update_kernel = None


def _can_use_fused_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    final_scale: torch.Tensor,
) -> bool:
    return (
        _sage_update_kernel is not None
        and param.is_cuda
        and grad.is_cuda
        and exp_avg.is_cuda
        and final_scale.is_cuda
        and param.is_contiguous()
        and grad.is_contiguous()
        and exp_avg.is_contiguous()
        and final_scale.is_contiguous()
    )


def _fused_sage_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    final_scale: torch.Tensor,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
) -> None:
    assert _sage_update_kernel is not None
    n_elements = param.numel()
    n_cols = param.shape[-1] if param.ndim > 1 else n_elements
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _sage_update_kernel[grid](
        param,
        grad,
        exp_avg,
        final_scale,
        n_elements,
        n_cols,
        lr,
        weight_decay,
        beta1,
        beta2,
        block_size,
    )


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
        fused: bool = True,
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=fused)
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
            fused = group["fused"]

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

                if fused and _can_use_fused_update(param, grad, exp_avg, final_scale):
                    _fused_sage_update(param, grad, exp_avg, final_scale, lr, weight_decay, beta1, beta2)
                else:
                    if weight_decay != 0.0:
                        param.mul_(1.0 - lr * weight_decay)
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
    sage_fused: bool = True,
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

    sage_opt = Sage(sage_groups, lr=lr, betas=sage_betas, eps=sage_eps, fused=sage_fused)

    return [muon_opt, sage_opt]
