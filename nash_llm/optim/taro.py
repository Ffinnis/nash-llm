from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


def _sinkhorn_impl(x: Tensor, n_iters: int = 5) -> Tensor:
    """Alternating row/column L2 normalization."""
    y = x.bfloat16()
    for _ in range(n_iters):
        row_norms = y.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        y = y / row_norms
        col_norms = y.norm(dim=-2, keepdim=True).clamp(min=1e-7)
        y = y / col_norms
    return y


def _cholqr_impl(a: Tensor, eps: float = 1e-6) -> Tensor:
    """Project onto O(m) with shifted CholQR."""
    ata = a.mT @ a
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    ata = ata + eps * eye
    try:
        l = torch.linalg.cholesky(ata)
        x = torch.linalg.solve_triangular(l, a.mT, upper=False)
        q = x.mT
    except torch.linalg.LinAlgError:
        q, _ = torch.linalg.qr(a)
    return q


if torch.cuda.is_available():
    sinkhorn = torch.compile(_sinkhorn_impl)
    cholqr = torch.compile(_cholqr_impl)
else:
    sinkhorn = _sinkhorn_impl
    cholqr = _cholqr_impl


@dataclass(frozen=True)
class TaroParamRef:
    param: nn.Parameter
    transpose_in_taro: bool
    lr_mult: float
    group_name: str


@dataclass
class TaroGroup:
    group_name: str
    blocks: list[list[TaroParamRef]]
    lr_mult: float = 1.0


class Taro(Optimizer):
    """TARO optimizer: tensorized ARO-style rotation with Sinkhorn base map."""

    def __init__(
        self,
        taro_groups: list[TaroGroup],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        sinkhorn_iters: int = 5,
    ):
        all_params: list[nn.Parameter] = []
        for group in taro_groups:
            for block in group.blocks:
                for ref in block:
                    all_params.append(ref.param)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            sinkhorn_iters=sinkhorn_iters,
        )
        super().__init__(all_params, defaults)

        self._taro_groups = taro_groups
        self._rotations: dict[str, Tensor] = {}
        self._last_step_metrics: dict[str, float] = {}

        for group in taro_groups:
            first_ref = None
            for block in group.blocks:
                if block:
                    first_ref = block[0]
                    break
            if first_ref is None:
                continue
            shape = self._canonical_shape(first_ref)
            m = shape[0]
            self._rotations[group.group_name] = torch.eye(
                m,
                device=first_ref.param.device,
                dtype=torch.float32,
            )

    @staticmethod
    def _canonical_shape(ref: TaroParamRef) -> tuple[int, int]:
        if ref.transpose_in_taro:
            return ref.param.shape[1], ref.param.shape[0]
        return ref.param.shape[0], ref.param.shape[1]

    @staticmethod
    def _to_canonical(t: Tensor, ref: TaroParamRef) -> Tensor:
        return t.mT if ref.transpose_in_taro else t

    @staticmethod
    def _to_native(t: Tensor, ref: TaroParamRef) -> Tensor:
        return t.mT if ref.transpose_in_taro else t

    def _momentum_buffer(self, ref: TaroParamRef) -> Tensor:
        state = self.state[ref.param]
        if len(state) == 0:
            m, n = self._canonical_shape(ref)
            state["momentum_buffer"] = torch.zeros(
                (m, n),
                device=ref.param.device,
                dtype=ref.param.dtype,
            )
        return state["momentum_buffer"]

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        base_lr = self.param_groups[0].get("lr", self.defaults["lr"])
        mu = self.defaults["momentum"]
        wd = self.defaults["weight_decay"]
        n_iters = self.defaults["sinkhorn_iters"]
        self._last_step_metrics = {}

        for group in self._taro_groups:
            group_name = group.group_name
            if group_name not in self._rotations:
                continue
            r = self._rotations[group_name]

            # Momentum-first update
            for block in group.blocks:
                for ref in block:
                    grad = ref.param.grad
                    if grad is None:
                        continue
                    grad_canonical = self._to_canonical(grad, ref)
                    buf = self._momentum_buffer(ref)
                    if buf.shape != grad_canonical.shape:
                        buf = torch.zeros_like(grad_canonical)
                        self.state[ref.param]["momentum_buffer"] = buf
                    buf.mul_(mu).add_(grad_canonical, alpha=1.0 - mu)

            complete_blocks = [block for block in group.blocks if block and all(ref.param.grad is not None for ref in block)]
            incomplete_blocks = [block for block in group.blocks if block and not all(ref.param.grad is not None for ref in block)]

            orth_errs: list[float] = []
            rot_update_norms: list[float] = []
            effective_lrs: list[float] = []

            for block in complete_blocks:
                z = torch.cat([self.state[ref.param]["momentum_buffer"] for ref in block], dim=1)
                r_typed = r.to(dtype=z.dtype, device=z.device)
                y = r_typed.mT @ z
                s = sinkhorn(y.unsqueeze(0), n_iters).squeeze(0).to(dtype=z.dtype)

                zst = z @ s.mT
                r_new = cholqr(zst.float())
                r_new_typed = r_new.to(dtype=z.dtype, device=z.device)
                delta_z = r_new_typed @ s  # Use R_new for this step.

                r_old = r.to(dtype=r_new.dtype, device=r_new.device)
                rot_update_norms.append(float((r_new - r_old).norm().item()))
                eye = torch.eye(r_new.shape[-1], device=r_new.device, dtype=r_new.dtype)
                orth_err = (r_new @ r_new.mT - eye).norm() / eye.norm()
                orth_errs.append(float(orth_err.item()))

                # Apply updates slice-by-slice
                offset = 0
                for ref in block:
                    buf = self.state[ref.param]["momentum_buffer"]
                    width = buf.shape[1]
                    delta = delta_z[:, offset : offset + width]
                    offset += width

                    m_dim, n_dim = delta.shape
                    norm = delta.norm().clamp(min=1e-7)
                    delta = delta * ((m_dim * n_dim) ** 0.5 / norm)
                    delta_native = self._to_native(delta, ref)

                    lr_eff = base_lr * ref.lr_mult
                    effective_lrs.append(float(lr_eff))
                    if wd > 0:
                        ref.param.data.mul_(1.0 - lr_eff * wd)
                    ref.param.data.add_(delta_native.to(ref.param.dtype), alpha=-lr_eff)

                r = r_new.to(dtype=self._rotations[group_name].dtype, device=self._rotations[group_name].device)

            # Fallback path for partial gradient availability
            for block in incomplete_blocks:
                for ref in block:
                    if ref.param.grad is None:
                        continue
                    buf = self.state[ref.param]["momentum_buffer"]
                    r_typed = r.to(dtype=buf.dtype, device=buf.device)
                    y = r_typed.mT @ buf
                    s = sinkhorn(y.unsqueeze(0), n_iters).squeeze(0).to(dtype=buf.dtype)
                    zst = buf @ s.mT
                    r_new = cholqr(zst.float())
                    r_new_typed = r_new.to(dtype=buf.dtype, device=buf.device)
                    delta = r_new_typed @ s

                    m_dim, n_dim = delta.shape
                    norm = delta.norm().clamp(min=1e-7)
                    delta = delta * ((m_dim * n_dim) ** 0.5 / norm)
                    delta_native = self._to_native(delta, ref)

                    lr_eff = base_lr * ref.lr_mult
                    effective_lrs.append(float(lr_eff))
                    if wd > 0:
                        ref.param.data.mul_(1.0 - lr_eff * wd)
                    ref.param.data.add_(delta_native.to(ref.param.dtype), alpha=-lr_eff)

                    r_old = r.to(dtype=r_new.dtype, device=r_new.device)
                    rot_update_norms.append(float((r_new - r_old).norm().item()))
                    eye = torch.eye(r_new.shape[-1], device=r_new.device, dtype=r_new.dtype)
                    orth_err = (r_new @ r_new.mT - eye).norm() / eye.norm()
                    orth_errs.append(float(orth_err.item()))
                    r = r_new.to(dtype=self._rotations[group_name].dtype, device=self._rotations[group_name].device)

            self._rotations[group_name] = r
            if orth_errs:
                self._last_step_metrics[f"taro/{group_name}/orth_err"] = float(sum(orth_errs) / len(orth_errs))
            if rot_update_norms:
                self._last_step_metrics[f"taro/{group_name}/rotation_update_norm"] = float(sum(rot_update_norms) / len(rot_update_norms))
            if effective_lrs:
                self._last_step_metrics[f"taro/{group_name}/effective_lr"] = float(sum(effective_lrs) / len(effective_lrs))

        return loss

    def get_step_metrics(self) -> dict[str, float]:
        return dict(self._last_step_metrics)

    def state_dict(self):
        sd = super().state_dict()
        sd["taro_rotations"] = {k: v.cpu() for k, v in self._rotations.items()}
        return sd

    def load_state_dict(self, state_dict):
        rotations = state_dict.pop("taro_rotations", {})
        super().load_state_dict(state_dict)
        for key, value in rotations.items():
            if key in self._rotations:
                self._rotations[key] = value.to(device=self._rotations[key].device)
