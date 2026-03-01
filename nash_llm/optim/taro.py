import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


def _sinkhorn_impl(X: Tensor, n_iters: int = 5) -> Tensor:
    """Alternating row/col L2 normalization (Sinkhorn balancing).

    Operates in bfloat16 for speed.  Supports batched input (..., m, n).
    """
    Y = X.bfloat16()
    for _ in range(n_iters):
        # Row normalization
        row_norms = Y.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        Y = Y / row_norms
        # Column normalization
        col_norms = Y.norm(dim=-2, keepdim=True).clamp(min=1e-7)
        Y = Y / col_norms
    return Y


def _cholqr_impl(A: Tensor, eps: float = 1e-6) -> Tensor:
    """CholQR projection onto O(m): Q = A R^{-1} via Cholesky of A^T A.

    Standard CholQR: A^T A = L L^T, R = L^T, Q = A L^{-T}.
    Input: (..., m, m) square matrix.
    Returns: (..., m, m) orthogonal matrix.
    Falls back to torch.linalg.qr if Cholesky fails.
    """
    ATA = A.mT @ A
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    ATA = ATA + eps * eye

    try:
        L = torch.linalg.cholesky(ATA)  # A^T A = L L^T
        # Q = A L^{-T}: solve L X = A^T => X = L^{-1} A^T => X^T = A L^{-T} = Q
        X = torch.linalg.solve_triangular(L, A.mT, upper=False)
        Q = X.mT
    except torch.linalg.LinAlgError:
        Q, _ = torch.linalg.qr(A)
    return Q


# torch.compile for GPU acceleration
if torch.cuda.is_available():
    sinkhorn = torch.compile(_sinkhorn_impl)
    cholqr = torch.compile(_cholqr_impl)
else:
    sinkhorn = _sinkhorn_impl
    cholqr = _cholqr_impl


class Taro(Optimizer):
    """TARO optimizer: TEON tensorized gradients + ARO adaptive rotation.

    Uses Sinkhorn normalization instead of Polar Express, and maintains
    persistent rotation matrices per symmetry group.

    Args:
        taro_groups: list of (group_type, blocks) where group_type is a string
            identifier (e.g. "qkv", "o", "up", "down") and blocks is a list
            of param lists. Each inner list contains K params to stack.
        lr: learning rate
        momentum: momentum coefficient
        weight_decay: decoupled weight decay
        sinkhorn_iters: number of Sinkhorn normalization iterations
    """

    def __init__(
        self,
        taro_groups: list[tuple[str, list[list[nn.Parameter]]]],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        sinkhorn_iters: int = 5,
    ):
        # Flatten all params for Optimizer base class registration
        all_params: list[nn.Parameter] = []
        for _group_type, blocks in taro_groups:
            for block in blocks:
                all_params.extend(block)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            sinkhorn_iters=sinkhorn_iters,
        )
        super().__init__(all_params, defaults)

        self._taro_groups = taro_groups

        # Initialize rotation matrices to identity, one per group type.
        # Dimensions are inferred from first param in each group.
        self._rotations: dict[str, Tensor] = {}
        for group_type, blocks in taro_groups:
            if group_type in self._rotations or not blocks or not blocks[0]:
                continue
            m = blocks[0][0].shape[0]
            R = torch.eye(m, device=blocks[0][0].device, dtype=torch.float32)
            self._rotations[group_type] = R

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lr = self.defaults["lr"]
        mu = self.defaults["momentum"]
        wd = self.defaults["weight_decay"]
        n_iters = self.defaults["sinkhorn_iters"]

        for group_type, blocks in self._taro_groups:
            R = self._rotations[group_type]

            # Phase 1: update momentum buffers for all params in this group
            for block in blocks:
                for param in block:
                    if param.grad is None:
                        continue
                    state = self.state[param]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(param.data)
                    buf = state["momentum_buffer"]
                    buf.mul_(mu).add_(param.grad, alpha=1.0 - mu)

            # Phase 2: process all blocks of this group type
            # Collect complete blocks (all grads present)
            complete_blocks: list[list[nn.Parameter]] = []
            incomplete_blocks: list[list[nn.Parameter]] = []
            for block in blocks:
                if all(p.grad is not None for p in block):
                    complete_blocks.append(block)
                else:
                    incomplete_blocks.append(block)

            if complete_blocks:
                # Tensorize: stack momentums within each block, then batch across blocks
                Z_list = []
                for block in complete_blocks:
                    momentums = [self.state[p]["momentum_buffer"] for p in block]
                    Z_list.append(torch.cat(momentums, dim=1))  # (m, K*n)

                Z_batch = torch.stack(Z_list)  # (B, m, K*n)

                # Rotate: Y = R^T @ Z
                R_float = R.to(dtype=Z_batch.dtype, device=Z_batch.device)
                Y = R_float.mT @ Z_batch  # (B, m, K*n)

                # Sinkhorn normalize
                S = sinkhorn(Y, n_iters)  # (B, m, K*n) in bf16

                # Compute update: ΔZ = R_old @ S (consistent — S was computed with R_old)
                S_float = S.to(dtype=Z_batch.dtype)
                delta_Z = R_float @ S_float  # (B, m, K*n)

                # Update rotation for NEXT step: R_new = CholQR(mean(Z @ S^T))
                ZST = Z_batch @ S_float.mT  # (B, m, m)
                mean_ZST = ZST.mean(dim=0)  # (m, m)
                R_new = cholqr(mean_ZST.float())
                self._rotations[group_type] = R_new.to(dtype=R.dtype, device=R.device)

                # Split back and RMS normalize, then apply
                for bi, block in enumerate(complete_blocks):
                    n = block[0].shape[1]
                    slices = delta_Z[bi].split(n, dim=1)
                    for i, param in enumerate(block):
                        m_dim, n_dim = param.shape
                        delta_w = slices[i]
                        # Normalize to polar_express magnitude, then scale like Muon
                        norm = delta_w.norm().clamp(min=1e-7)
                        delta_w = delta_w * (min(m_dim, n_dim) ** 0.5 / norm)
                        scale = (m_dim / n_dim) ** 0.5
                        if wd > 0:
                            param.data.mul_(1.0 - lr * wd)
                        param.data.add_(delta_w.to(param.dtype), alpha=-lr * scale)

            # Fallback: process incomplete blocks per-param with Sinkhorn only
            for block in incomplete_blocks:
                for param in block:
                    if param.grad is None:
                        continue
                    buf = self.state[param]["momentum_buffer"]
                    s = sinkhorn(buf.unsqueeze(0), n_iters).squeeze(0)
                    m_dim, n_dim = param.shape
                    norm = s.norm().clamp(min=1e-7)
                    delta_w = s * (min(m_dim, n_dim) ** 0.5 / norm)
                    scale = (m_dim / n_dim) ** 0.5
                    if wd > 0:
                        param.data.mul_(1.0 - lr * wd)
                    param.data.add_(delta_w.to(param.dtype), alpha=-lr * scale)

        return loss

    def state_dict(self):
        sd = super().state_dict()
        sd["taro_rotations"] = {k: v.cpu() for k, v in self._rotations.items()}
        return sd

    def load_state_dict(self, state_dict):
        rotations = state_dict.pop("taro_rotations", {})
        super().load_state_dict(state_dict)
        for k, v in rotations.items():
            if k in self._rotations:
                self._rotations[k] = v.to(device=self._rotations[k].device)
