import torch
import pytest
import torch.nn as nn
from nash_llm.model import GPT
from nash_llm.config import ModelConfig
from nash_llm.optim.aro import (
    _sink_step_impl,
    shifted_cholesky_qr,
    compute_aro_rotation,
    ARO,
    LayerParamGroup,
)
from nash_llm.optim.adamw import configure_optimizers


class TestSinkStep:
    def test_preserves_shape(self):
        X = torch.randn(64, 128)
        result = _sink_step_impl(X, L=5)
        assert result.shape == X.shape

    def test_normalizes_rows_and_cols(self):
        """After Sinkhorn iterations, row and col norms should be approximately equal."""
        torch.manual_seed(42)
        X = torch.randn(64, 128)
        result = _sink_step_impl(X, L=10)

        row_norms = result.norm(dim=1)
        col_norms = result.norm(dim=0)

        # After convergence, row norms should be roughly equal to each other
        row_std = row_norms.std() / row_norms.mean()
        col_std = col_norms.std() / col_norms.mean()
        assert row_std < 0.1, f"Row norms not balanced: relative std = {row_std:.4f}"
        assert col_std < 0.1, f"Col norms not balanced: relative std = {col_std:.4f}"

    def test_idempotent(self):
        """Applying more iterations should converge."""
        torch.manual_seed(42)
        X = torch.randn(32, 64)
        result_5 = _sink_step_impl(X.clone(), L=5)
        result_10 = _sink_step_impl(X.clone(), L=10)
        result_20 = _sink_step_impl(X.clone(), L=20)

        # 10 and 20 iterations should be closer to each other than 5 and 10
        diff_5_10 = (result_5 - result_10).norm()
        diff_10_20 = (result_10 - result_20).norm()
        assert diff_10_20 < diff_5_10, "Should converge with more iterations"


class TestShiftedCholeskyQR:
    def test_produces_orthogonal_matrix(self):
        """Q^T @ Q should approximate identity."""
        torch.manual_seed(42)
        # Use a well-conditioned matrix (identity + noise) to avoid edge cases
        A = torch.eye(64) + 0.1 * torch.randn(64, 64)
        Q = shifted_cholesky_qr(A)

        product = Q.T @ Q
        eye = torch.eye(64)
        error = (product - eye).norm()
        assert error < 0.01, f"Orthogonality error {error:.6f}"

    def test_square_matrix(self):
        torch.manual_seed(42)
        A = torch.randn(128, 128)
        Q = shifted_cholesky_qr(A)
        assert Q.shape == (128, 128)

    def test_fallback_on_singular(self):
        """Near-singular matrix should fallback to standard QR without error."""
        A = torch.zeros(32, 32)
        A[0, 0] = 1.0  # rank-1
        Q = shifted_cholesky_qr(A)
        assert Q.shape == (32, 32)
        assert torch.isfinite(Q).all()


class TestComputeAroRotation:
    def test_rotation_is_orthogonal(self):
        """R^T @ R should approximate identity."""
        torch.manual_seed(42)
        # The cross-gram matrix M @ f_sink(M)^T is generally well-conditioned
        # because sink_step normalizes the input, improving conditioning
        M = torch.randn(32, 128)
        R_prev = torch.eye(32)
        R_new = compute_aro_rotation(M, R_prev, sink_iters=5)

        product = R_new.T @ R_new
        eye = torch.eye(32)
        error = (product - eye).norm()
        assert error < 0.1, f"Rotation orthogonality error {error:.6f}"

    def test_rotation_from_identity(self):
        """First step (R_prev=I) should produce valid rotation."""
        torch.manual_seed(42)
        M = torch.randn(32, 128)
        R_prev = torch.eye(32)
        R_new = compute_aro_rotation(M, R_prev, sink_iters=5)

        assert R_new.shape == (32, 32)
        assert torch.isfinite(R_new).all()

    def test_rotation_changes_across_steps(self):
        """Updating R with new momentum should produce different R."""
        torch.manual_seed(42)
        M1 = torch.randn(32, 128)
        M2 = torch.randn(32, 128)
        R_prev = torch.eye(32)

        R1 = compute_aro_rotation(M1, R_prev, sink_iters=5)
        R2 = compute_aro_rotation(M2, R1, sink_iters=5)

        assert not torch.allclose(R1, R2, atol=1e-3)


class TestAROOptimizer:
    def _make_layer_group(self):
        """Create a minimal layer group for testing."""
        torch.manual_seed(42)
        consumers = [
            nn.Parameter(torch.randn(64, 32)),   # q_proj-like
            nn.Parameter(torch.randn(64, 32)),   # k_proj-like
            nn.Parameter(torch.randn(64, 32)),   # v_proj-like
            nn.Parameter(torch.randn(128, 32)),  # fc1-like
        ]
        producers = [
            nn.Parameter(torch.randn(32, 64)),   # out_proj-like (d_model=32 is dim 0)
            nn.Parameter(torch.randn(32, 128)),  # fc2-like
        ]
        return LayerParamGroup(layer_idx=0, consumer_params=consumers, producer_params=producers)

    def test_step_updates_weights(self):
        """A single ARO step should change the parameters."""
        group = self._make_layer_group()
        all_params = group.consumer_params + group.producer_params
        original = [p.clone() for p in all_params]

        opt = ARO(
            layer_groups=[group],
            lr=0.02,
            momentum=0.95,
            sink_iters=5,
            rms_target=0.2,
        )

        for p in all_params:
            p.grad = torch.randn_like(p)

        opt.step()

        for orig, updated in zip(original, all_params):
            assert not torch.allclose(orig, updated), "Params should have changed"

    def test_momentum_accumulation(self):
        """Momentum buffer should accumulate across steps."""
        group = self._make_layer_group()
        param = group.consumer_params[0]

        opt = ARO(layer_groups=[group], lr=0.02, momentum=0.95, sink_iters=5)

        # Set grads for all params
        for p in group.consumer_params + group.producer_params:
            p.grad = torch.randn_like(p)

        opt.step()
        buf_after_1 = opt.state[param]["momentum_buffer"].clone()

        for p in group.consumer_params + group.producer_params:
            p.grad = torch.randn_like(p)

        opt.step()
        buf_after_2 = opt.state[param]["momentum_buffer"]

        assert not torch.allclose(buf_after_1, buf_after_2)

    def test_rotation_updates(self):
        """Rotation matrix should change across steps."""
        group = self._make_layer_group()

        opt = ARO(layer_groups=[group], lr=0.02, momentum=0.95, sink_iters=5)

        for p in group.consumer_params + group.producer_params:
            p.grad = torch.randn_like(p)

        opt.step()
        sentinel = group.consumer_params[0]
        R_after_1 = opt.state[sentinel]["rotation"].clone()

        for p in group.consumer_params + group.producer_params:
            p.grad = torch.randn_like(p)

        opt.step()
        R_after_2 = opt.state[sentinel]["rotation"]

        assert not torch.allclose(R_after_1, R_after_2, atol=1e-3)

    def test_rms_normalization(self):
        """Update RMS should be approximately rms_target."""
        torch.manual_seed(42)
        group = self._make_layer_group()
        rms_target = 0.2

        opt = ARO(layer_groups=[group], lr=1.0, momentum=0.0, sink_iters=5, rms_target=rms_target)

        for p in group.consumer_params + group.producer_params:
            p.grad = torch.randn_like(p)

        originals = {id(p): p.clone() for p in group.consumer_params + group.producer_params}
        opt.step()

        # Check that RMS of (original - updated) / lr is close to rms_target
        for p in group.consumer_params + group.producer_params:
            diff = originals[id(p)] - p.data
            rows, cols = diff.shape
            rms = diff.norm() / (rows * cols) ** 0.5
            # lr=1.0 and wd=0 so diff = lr * update, and RMS should be ~rms_target
            assert abs(rms.item() - rms_target) < 0.05, f"RMS {rms:.4f} not close to target {rms_target}"

    def test_weight_decay(self):
        """Weight decay should reduce parameter norms."""
        torch.manual_seed(42)
        group = self._make_layer_group()

        opt_wd = ARO(layer_groups=[group], lr=0.02, momentum=0.95, weight_decay=0.1, sink_iters=5)

        for p in group.consumer_params + group.producer_params:
            p.grad = torch.zeros_like(p)  # zero grad: only WD acts

        norms_before = [p.data.norm().item() for p in group.consumer_params + group.producer_params]
        opt_wd.step()
        norms_after = [p.data.norm().item() for p in group.consumer_params + group.producer_params]

        for before, after in zip(norms_before, norms_after):
            assert after < before, "Weight decay should reduce norms"

    def test_shapes_preserved(self):
        """Parameter shapes should not change after step."""
        group = self._make_layer_group()
        shapes_before = [(p.shape) for p in group.consumer_params + group.producer_params]

        opt = ARO(layer_groups=[group], lr=0.02, momentum=0.95, sink_iters=5)
        for p in group.consumer_params + group.producer_params:
            p.grad = torch.randn_like(p)
        opt.step()

        shapes_after = [(p.shape) for p in group.consumer_params + group.producer_params]
        assert shapes_before == shapes_after


class TestConfigureOptimizers:
    def setup_method(self):
        self.cfg = ModelConfig(
            n_layers=4, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
        )
        self.model = GPT(self.cfg)

    def test_returns_aro_and_adamw(self):
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1)
        assert len(opts) == 2
        assert isinstance(opts[0], ARO)
        assert isinstance(opts[1], torch.optim.AdamW)

    def test_all_params_assigned(self):
        """Every trainable param should belong to exactly one optimizer."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1)

        opt_param_ids = set()
        for opt in opts:
            for pg in opt.param_groups:
                for p in pg["params"]:
                    opt_param_ids.add(id(p))

        model_param_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        assert model_param_ids == opt_param_ids, "Not all params assigned to an optimizer"

    def test_no_param_in_both_optimizers(self):
        """No param should be in both ARO and AdamW."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1)

        aro_ids = set()
        for pg in opts[0].param_groups:
            for p in pg["params"]:
                aro_ids.add(id(p))

        adamw_ids = set()
        for pg in opts[1].param_groups:
            for p in pg["params"]:
                adamw_ids.add(id(p))

        overlap = aro_ids & adamw_ids
        assert len(overlap) == 0, f"Found {len(overlap)} params in both optimizers"

    def test_aro_layer_groups(self):
        """ARO should have 4 layer groups (one per layer), each with 6 params."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1)
        aro_opt = opts[0]
        assert isinstance(aro_opt, ARO)

        assert len(aro_opt._layer_groups) == 4  # n_layers=4
        for group in aro_opt._layer_groups:
            assert len(group.consumer_params) == 4  # q, k, v, fc1
            assert len(group.producer_params) == 2  # out_proj, fc2

    def test_aro_has_correct_params(self):
        """ARO should contain exactly the weight matrices of each layer."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1)
        aro_opt = opts[0]

        aro_param_ids = set()
        for group in aro_opt._layer_groups:
            for p in group.consumer_params + group.producer_params:
                aro_param_ids.add(id(p))

        expected_ids = set()
        for block in self.model.blocks:
            for proj in [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj, block.attn.out_proj]:
                expected_ids.add(id(proj.weight))
            expected_ids.add(id(block.ff.fc1.weight))
            expected_ids.add(id(block.ff.fc2.weight))

        assert aro_param_ids == expected_ids
