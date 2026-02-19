import torch
import torch.nn as nn
from torch.optim import Optimizer
from dataclasses import dataclass, field


@dataclass
class LayerParamGroup:
    """One transformer layer's weight matrices for chain-coupled ARO.

    consumer_params: weights that consume the residual stream (Q, K, V, fc1).
                     Transposed before ARO so d_model becomes row dimension.
    producer_params: weights that produce into the residual stream (out_proj, fc2).
                     NOT transposed — d_model is already the row dimension.
    """

    layer_idx: int
    consumer_params: list[nn.Parameter] = field(default_factory=list)
    producer_params: list[nn.Parameter] = field(default_factory=list)


def _sink_step_impl(X: torch.Tensor, L: int = 5) -> torch.Tensor:
    """Sinkhorn row/col normalization (L iterations).

    f_Sink: simultaneous row and column normalization.
    X ∝ Q(X)⁻¹ · X · R(X)⁻¹ where Q = diag(row norms), R = diag(col norms).
    """
    for _ in range(L):
        row_norms = X.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        col_norms = X.norm(dim=-2, keepdim=True).clamp(min=1e-8)
        X = X / row_norms / col_norms
    return X


# torch.compile for GPU acceleration; eager on CPU
if torch.cuda.is_available():
    sink_step = torch.compile(_sink_step_impl)
else:
    sink_step = _sink_step_impl


def shifted_cholesky_qr(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Shifted Cholesky QR: Q-factor via Cholesky factorization.

    P = A^T A + eps*I  (regularized Gram matrix)
    P = L L^T          (Cholesky)
    Q = A L^{-T}       (triangular solve)

    Falls back to standard QR on Cholesky failure.
    """
    try:
        P = A.T @ A
        P.diagonal().add_(eps)
        L = torch.linalg.cholesky(P)
        Q = torch.linalg.solve_triangular(L.T, A.T, upper=True).T
        if not torch.isfinite(Q).all():
            raise RuntimeError("SCQR produced non-finite values")
        return Q
    except (RuntimeError, torch.linalg.LinAlgError):  # type: ignore[attr-defined]
        Q, _ = torch.linalg.qr(A)
        return Q


def compute_aro_rotation(
    M_stack: torch.Tensor,
    R_prev: torch.Tensor,
    sink_iters: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute ARO rotation: R_t = QR(M · f_sink(R_{t-1}^T · M)^T).

    Args:
        M_stack: [d, N] stacked oriented momentum matrix
        R_prev: [d, d] previous rotation matrix
        sink_iters: Sinkhorn iterations for f_sink
        eps: SCQR regularization epsilon
    Returns:
        R_new: [d, d] new rotation matrix
    """
    M_rotated = R_prev.T @ M_stack  # [d, N]
    D = sink_step(M_rotated, L=sink_iters)  # [d, N]
    cross_gram = M_stack @ D.T  # [d, d]
    return shifted_cholesky_qr(cross_gram, eps=eps)


class ARO(Optimizer):
    """ARO-Sinkhorn optimizer with chain-coupled rotation sharing.

    For each transformer layer ℓ, maintains one rotation R_ℓ ∈ SO(d_model)
    shared across all weight matrices in the layer (Q, K, V, out_proj, fc1, fc2).

    Math (per layer ℓ):
        For each param p: M_p = β M_p + (1-β) G_p         (momentum)
        Orient: transpose consumers, keep producers
        M_stack = cat(oriented momentums, dim=1)             [d_model, N_ℓ]
        R_ℓ = SCQR(M_stack · f_sink(R_prev^T · M_stack)^T)  (rotation)
        For each param p:
            ΔW = R_ℓ · f_sink(R_ℓ^T · M_oriented)           (rotated update)
            ΔW = rms_target · ΔW / RMS(ΔW)                  (normalize)
            W -= lr · ΔW                                     (apply)
    """

    def __init__(
        self,
        layer_groups: list[LayerParamGroup],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        sink_iters: int = 5,
        rms_target: float = 0.2,
        scqr_eps: float = 1e-6,
    ):
        # Flatten all params for Optimizer registration
        all_params: list[nn.Parameter] = []
        for group in layer_groups:
            all_params.extend(group.consumer_params)
            all_params.extend(group.producer_params)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            sink_iters=sink_iters,
            rms_target=rms_target,
            scqr_eps=scqr_eps,
        )
        super().__init__(all_params, defaults)

        self._layer_groups = layer_groups

        # Sentinel param per layer — stores rotation in self.state
        self._layer_sentinels: list[nn.Parameter] = []
        for group in layer_groups:
            self._layer_sentinels.append(group.consumer_params[0])

        # Map param id -> is_consumer for orientation
        self._param_is_consumer: dict[int, bool] = {}
        for group in layer_groups:
            for p in group.consumer_params:
                self._param_is_consumer[id(p)] = True
            for p in group.producer_params:
                self._param_is_consumer[id(p)] = False

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Read from param_groups (updated by scheduler), not defaults
        pg = self.param_groups[0]
        lr = pg["lr"]
        mu = pg["momentum"]
        wd = pg["weight_decay"]
        sink_iters = pg["sink_iters"]
        rms_target = pg["rms_target"]
        scqr_eps = pg["scqr_eps"]

        for layer_idx, group in enumerate(self._layer_groups):
            sentinel = self._layer_sentinels[layer_idx]
            sentinel_state = self.state[sentinel]

            # Initialize layer state
            if len(sentinel_state) == 0:
                d_model = sentinel.shape[-1]  # consumer: [out, d_model], last dim is d_model
                sentinel_state["rotation"] = torch.eye(
                    d_model, device=sentinel.device, dtype=torch.float32
                )

            R_prev = sentinel_state["rotation"]

            all_params = [
                (p, True) for p in group.consumer_params
            ] + [
                (p, False) for p in group.producer_params
            ]

            # Phase 1: Update momentum buffers
            oriented_list: list[tuple[torch.Tensor, nn.Parameter, bool]] = []

            for param, is_consumer in all_params:
                if param.grad is None:
                    continue

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param.data)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(param.grad, alpha=1.0 - mu)

                # Orient momentum (Design 3): transpose consumers
                if is_consumer:
                    M_oriented = buf.T  # [out, d_model] -> [d_model, out]
                else:
                    M_oriented = buf  # [d_model, ...] stays as-is

                oriented_list.append((M_oriented, param, is_consumer))

            if not oriented_list:
                continue

            # Phase 2: Stack oriented momentums and compute rotation in float32
            M_list = [m.float() for m, _, _ in oriented_list]
            M_stack = torch.cat(M_list, dim=1)  # [d_model, N_ℓ]

            R_prev_dev = R_prev.to(M_stack.device)
            R_new = compute_aro_rotation(M_stack, R_prev_dev, sink_iters, scqr_eps)
            sentinel_state["rotation"] = R_new

            # Phase 3: Per-param rotated update
            for M_oriented, param, is_consumer in oriented_list:
                M_f32 = M_oriented.float()
                M_rot = R_new.T @ M_f32  # [d_model, n_cols]
                update = R_new @ sink_step(M_rot, L=sink_iters)  # [d_model, n_cols]

                # RMS normalization
                rows, cols = update.shape
                rms = update.norm() / (rows * cols) ** 0.5
                if rms > 1e-8:
                    update = update * (rms_target / rms)

                # Cast back to param dtype
                update = update.to(param.dtype)

                # De-orient (transpose back for consumers)
                if is_consumer:
                    update = update.T  # [d_model, out] -> [out, d_model]

                # Decoupled weight decay
                if wd > 0:
                    param.data.mul_(1.0 - lr * wd)

                param.data.add_(update, alpha=-lr)

        return loss
