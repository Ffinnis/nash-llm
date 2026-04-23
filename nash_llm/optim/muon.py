import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from itertools import repeat

# Pre-computed Polar Express coefficients for ℓ=1e-3, u=1, degree-5.
# From Amsel et al., "The Polar Express", 2025.
_POLAR_EXPRESS_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

# Safety factor 1/1.01 for all except last (already Newton-Schulz converged)
_POLAR_EXPRESS_COEFFS_SAFE = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for (a, b, c) in _POLAR_EXPRESS_COEFFS[:-1]
] + [_POLAR_EXPRESS_COEFFS[-1]]


def _polar_express_impl(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Polar Express: optimal polynomial approximation of polar(G) = UV^T.

    Uses pre-computed degree-5 adaptive coefficients with safety factor
    for bfloat16 stability. Transposes tall matrices to reduce FLOPs.
    """
    assert G.ndim >= 2
    X = G.bfloat16()

    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    # Normalize: X0 = M / (||M||_F + eps), safety factor folded into coeffs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    # Select coefficients, extending with Newton-Schulz for steps > 8
    hs = _POLAR_EXPRESS_COEFFS_SAFE[:steps] + list(
        repeat(_POLAR_EXPRESS_COEFFS_SAFE[-1], max(0, steps - len(_POLAR_EXPRESS_COEFFS_SAFE)))
    )

    for a, b, c in hs:
        A = X @ X.mT  # X X^T  (m x m)
        B = b * A + c * A @ A  # bA + cA^2
        X = a * X + B @ X  # aX + bX^3 + cX^5

    if transposed:
        X = X.mT

    return X


# torch.compile for GPU acceleration; falls back to eager on CPU
if torch.cuda.is_available():
    polar_express = torch.compile(_polar_express_impl)
else:
    polar_express = _polar_express_impl


def orthogonalize(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalization: M -> UV^T via Polar Express."""
    return polar_express(M, steps)


def _orthonormalize_columns(M: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(M.float(), mode="reduced")
    return Q.to(M.dtype)


def _dion_rank(shape: torch.Size, rank_fraction: float, rank_multiple_of: int) -> int:
    r = rank_fraction * min(shape)
    r = rank_multiple_of * math.ceil(r / rank_multiple_of)
    return max(1, min(r, *shape))


def _init_dion_state(
    param: nn.Parameter,
    state: dict,
    rank_fraction: float,
    rank_multiple_of: int,
) -> None:
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(param.data)

    m, n = param.shape
    is_transposed = m < n
    r = _dion_rank(param.shape, rank_fraction, rank_multiple_of)
    q_rows = m if is_transposed else n
    state["Q"] = torch.randn((q_rows, r), device=param.device, dtype=param.dtype)
    state["is_transposed"] = is_transposed


def _dion_update(
    param: nn.Parameter,
    grad: torch.Tensor,
    state: dict,
    lr: float,
    momentum: float,
    weight_decay: float,
    eps: float,
    power_iters: int,
) -> None:
    M = state["momentum_buffer"]
    Q = state["Q"]
    is_transposed = state["is_transposed"]

    M.add_(grad.to(M.dtype))
    B = M.mT if is_transposed else M

    P = None
    R = None
    for _ in range(power_iters):
        P = _orthonormalize_columns(B @ Q.to(B.dtype))
        R = B.mT @ P
        Q = R / (R.float().norm(dim=0, keepdim=True).add(eps).to(R.dtype))

    assert P is not None and R is not None

    if is_transposed:
        M.add_(R @ P.mT, alpha=-(1.0 - momentum))
    else:
        M.add_(P @ R.mT, alpha=-(1.0 - momentum))

    state["Q"] = Q.to(state["Q"].dtype)

    if weight_decay > 0:
        param.data.mul_(1.0 - lr * weight_decay)

    scale = (param.shape[0] / param.shape[1]) ** 0.5
    if is_transposed:
        update = Q @ P.mT
    else:
        update = P @ Q.mT
    param.data.add_(update.to(param.dtype), alpha=-lr * scale)


class TeonDion(Optimizer):
    """TEON + Dion optimizer.

    Handles two types of parameter groups:
    - muon_params: per-layer Dion low-rank updates (out_proj, MLP weights)
    - teon_params: cross-layer stacking of K consecutive blocks (Q/K/V)

    Math (Dion per param):
        M_t = M_{t-1} + G_t
        P_t, R_t ~= low_rank(M_t)
        M_t = M_t - (1 - mu) * P_t R_t^T
        Q_t = normalize_columns(R_t)
        W_t = W_{t-1} - lr * sqrt(m/n) * P_t Q_t^T

    Math (TEON per group of K params):
        Z = [M^(1) | M^(2) | ... | M^(K)]   (mode-1 matricization)
        Q = Ortho(Z)
        O^(k) = Q[:, k*n:(k+1)*n]            (split back)
        W^(k)_t = W^(k)_{t-1} - lr * sqrt(m/n) * O^(k)
    """

    def __init__(
        self,
        muon_params: list[nn.Parameter],
        teon_params: list[list[nn.Parameter]],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        dion_rank_fraction: float = 0.25,
        dion_rank_multiple_of: int = 1,
        dion_power_iters: int = 1,
        eps: float = 1e-8,
    ):
        if not 0 < dion_rank_fraction <= 1:
            raise ValueError(f"dion_rank_fraction must be in (0, 1], got {dion_rank_fraction}")
        if dion_rank_multiple_of <= 0:
            raise ValueError(f"dion_rank_multiple_of must be positive, got {dion_rank_multiple_of}")
        if dion_power_iters <= 0:
            raise ValueError(f"dion_power_iters must be positive, got {dion_power_iters}")

        # Flatten all params for Optimizer registration
        all_params: list[nn.Parameter] = []
        all_params.extend(muon_params)
        for group in teon_params:
            all_params.extend(group)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            dion_rank_fraction=dion_rank_fraction,
            dion_rank_multiple_of=dion_rank_multiple_of,
            dion_power_iters=dion_power_iters,
            eps=eps,
        )
        super().__init__(all_params, defaults)

        self._muon_params = muon_params
        self._dion_params = muon_params
        self._teon_groups = teon_params

        # Pre-group TEON groups by (m, n, K) for batched orthogonalization
        self._teon_shape_groups: dict[tuple[int, int, int], list[list[nn.Parameter]]] = {}
        for group in teon_params:
            if not group:
                continue
            m, n = group[0].shape
            K = len(group)
            key = (m, n, K)
            self._teon_shape_groups.setdefault(key, []).append(group)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lr = self.defaults["lr"]
        mu = self.defaults["momentum"]
        wd = self.defaults["weight_decay"]
        ns_steps = self.defaults["ns_steps"]
        dion_rank_fraction = self.defaults["dion_rank_fraction"]
        dion_rank_multiple_of = self.defaults["dion_rank_multiple_of"]
        dion_power_iters = self.defaults["dion_power_iters"]
        eps = self.defaults["eps"]

        # --- Dion: per-layer low-rank update with error feedback ---
        for param in self._dion_params:
            if param.grad is None:
                continue
            state = self.state[param]
            if len(state) == 0 or "Q" not in state:
                _init_dion_state(param, state, dion_rank_fraction, dion_rank_multiple_of)
            _dion_update(
                param=param,
                grad=param.grad,
                state=state,
                lr=lr,
                momentum=mu,
                weight_decay=wd,
                eps=eps,
                power_iters=dion_power_iters,
            )

        # --- TEON: batched cross-layer stacking ---
        # Phase 1: update all TEON momentum buffers
        for group in self._teon_groups:
            if not all(param.grad is not None for param in group):
                continue
            for param in group:
                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param.data)
                buf = state["momentum_buffer"]
                assert param.grad is not None
                buf.mul_(mu).add_(param.grad, alpha=1.0 - mu)

        # Phase 2: batched orthogonalization by (m, n, K)
        for (m, n, K), groups in self._teon_shape_groups.items():
            complete = []
            incomplete = []
            for group in groups:
                if all(p.grad is not None for p in group):
                    complete.append(group)
                else:
                    incomplete.append(group)

            # Fast path: batch all complete groups
            if complete:
                Z_list = []
                for group in complete:
                    momentums = [self.state[p]["momentum_buffer"] for p in group]
                    Z_list.append(torch.cat(momentums, dim=1))
                Z_batch = torch.stack(Z_list)
                Q_batch = orthogonalize(Z_batch, ns_steps)
                scale = (m / n) ** 0.5
                for gi, group in enumerate(complete):
                    ortho_slices = Q_batch[gi].split(n, dim=1)
                    for i, param in enumerate(group):
                        if wd > 0:
                            param.data.mul_(1.0 - lr * wd)
                        param.data.add_(ortho_slices[i], alpha=-lr * scale)

            # Fallback: per-param Dion for incomplete groups
            for group in incomplete:
                for param in group:
                    if param.grad is None:
                        continue
                    state = self.state[param]
                    if "Q" not in state:
                        _init_dion_state(param, state, dion_rank_fraction, dion_rank_multiple_of)
                    _dion_update(
                        param=param,
                        grad=param.grad,
                        state=state,
                        lr=lr,
                        momentum=mu,
                        weight_decay=wd,
                        eps=eps,
                        power_iters=dion_power_iters,
                    )

        return loss


Muon = TeonDion
