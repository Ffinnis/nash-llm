import torch
import pytest
import torch.nn as nn
from nash_llm.model import GPT
from nash_llm.config import ModelConfig
from nash_llm.optim.taro import _sinkhorn_impl, _cholqr_impl, Taro
from nash_llm.optim.adamw import configure_optimizers


class TestSinkhorn:
    def test_output_shape(self):
        torch.manual_seed(42)
        X = torch.randn(64, 128)
        result = _sinkhorn_impl(X, n_iters=5)
        assert result.shape == (64, 128)

    def test_row_norms_converge(self):
        """After Sinkhorn, row norms should be uniform (all equal)."""
        torch.manual_seed(42)
        X = torch.randn(32, 64)
        result = _sinkhorn_impl(X, n_iters=10).float()
        row_norms = result.norm(dim=-1)
        # All row norms should be nearly identical (Sinkhorn balances them)
        assert row_norms.std() < 0.01, f"Row norms should be uniform, std={row_norms.std():.4f}"
        # Column norms should also be uniform (last step is col normalization)
        col_norms = result.norm(dim=-2)
        assert col_norms.std() < 0.01, f"Col norms should be uniform, std={col_norms.std():.4f}"

    def test_batched_consistency(self):
        """Batched Sinkhorn should match independent calls."""
        torch.manual_seed(42)
        X1 = torch.randn(32, 64)
        X2 = torch.randn(32, 64)

        r1 = _sinkhorn_impl(X1, n_iters=5)
        r2 = _sinkhorn_impl(X2, n_iters=5)

        batch = torch.stack([X1, X2])
        rb = _sinkhorn_impl(batch, n_iters=5)

        assert torch.allclose(r1, rb[0], atol=1e-5)
        assert torch.allclose(r2, rb[1], atol=1e-5)

    def test_zero_input_safety(self):
        """Sinkhorn should not produce NaN on zero input."""
        X = torch.zeros(16, 32)
        result = _sinkhorn_impl(X, n_iters=5)
        assert not torch.isnan(result).any()


class TestCholQR:
    def test_orthogonality(self):
        """CholQR output should be approximately orthogonal."""
        torch.manual_seed(42)
        A = torch.randn(32, 32)
        Q = _cholqr_impl(A)
        product = Q @ Q.mT
        eye = torch.eye(32)
        error = (product - eye).norm()
        assert error < 0.5, f"Orthogonality error {error:.4f}"

    def test_batched(self):
        """CholQR should work on batched input."""
        torch.manual_seed(42)
        A = torch.randn(3, 16, 16)
        Q = _cholqr_impl(A)
        assert Q.shape == (3, 16, 16)
        for i in range(3):
            product = Q[i] @ Q[i].mT
            eye = torch.eye(16)
            error = (product - eye).norm()
            assert error < 0.5, f"Batch {i} orthogonality error {error:.4f}"

    def test_shape_preservation(self):
        torch.manual_seed(42)
        A = torch.randn(64, 64)
        Q = _cholqr_impl(A)
        assert Q.shape == (64, 64)


class TestTaroOptimizer:
    def setup_method(self):
        torch.manual_seed(42)
        self.params = [nn.Parameter(torch.randn(64, 128)) for _ in range(4)]

    def test_step_updates_weights(self):
        """A single Taro step should change the parameters."""
        original = [p.clone() for p in self.params[:2]]
        opt = Taro(
            taro_groups=[("test", [self.params[:2]])],
            lr=0.02,
            momentum=0.95,
            sinkhorn_iters=5,
        )
        for p in self.params[:2]:
            p.grad = torch.randn_like(p)
        opt.step()
        for orig, updated in zip(original, self.params[:2]):
            assert not torch.allclose(orig, updated), "Params should have changed"

    def test_rotation_updates_from_identity(self):
        """Rotation matrix should change from identity after a step."""
        opt = Taro(
            taro_groups=[("test", [self.params[:2]])],
            lr=0.02,
            momentum=0.95,
            sinkhorn_iters=5,
        )
        R_before = opt._rotations["test"].clone()
        assert torch.allclose(R_before, torch.eye(64))

        for p in self.params[:2]:
            p.grad = torch.randn_like(p)
        opt.step()

        R_after = opt._rotations["test"]
        assert not torch.allclose(R_after, torch.eye(64), atol=1e-3), "Rotation should change"

    def test_rotation_stays_orthogonal(self):
        """Rotation matrix should remain approximately orthogonal after steps."""
        opt = Taro(
            taro_groups=[("test", [self.params[:2]])],
            lr=0.02,
            momentum=0.95,
            sinkhorn_iters=5,
        )
        for step in range(5):
            for p in self.params[:2]:
                p.grad = torch.randn_like(p)
            opt.step()

        R = opt._rotations["test"].float()
        product = R @ R.mT
        eye = torch.eye(R.shape[0])
        error = (product - eye).norm() / eye.norm()
        assert error < 0.1, f"Rotation relative orthogonality error {error:.4f}"

    def test_differs_from_muon(self):
        """TARO should produce different updates than MUON."""
        from nash_llm.optim.muon import Muon

        torch.manual_seed(42)
        g1 = torch.randn(64, 128)
        g2 = torch.randn(64, 128)

        # TARO path
        p1_taro = nn.Parameter(torch.randn(64, 128))
        p2_taro = nn.Parameter(p1_taro.detach().clone())
        # Use different params for block but same initial weights
        p_taro_block = [nn.Parameter(p1_taro.detach().clone()), nn.Parameter(p1_taro.detach().clone())]
        opt_taro = Taro(
            taro_groups=[("test", [p_taro_block])],
            lr=0.02, momentum=0.0, sinkhorn_iters=5,
        )
        p_taro_block[0].grad = g1.clone()
        p_taro_block[1].grad = g2.clone()
        opt_taro.step()

        # MUON path
        p_muon_block = [nn.Parameter(p1_taro.detach().clone()), nn.Parameter(p1_taro.detach().clone())]
        opt_muon = Muon(
            muon_params=p_muon_block, teon_params=[],
            lr=0.02, momentum=0.0, ns_steps=5,
        )
        p_muon_block[0].grad = g1.clone()
        p_muon_block[1].grad = g2.clone()
        opt_muon.step()

        assert not torch.allclose(p_taro_block[0], p_muon_block[0], atol=1e-3), "TARO should differ from MUON"

    def test_shapes_preserved(self):
        """After TARO step, param shapes should be unchanged."""
        opt = Taro(
            taro_groups=[("test", [self.params[:2]])],
            lr=0.02, momentum=0.95, sinkhorn_iters=5,
        )
        for p in self.params[:2]:
            p.grad = torch.randn_like(p)
        opt.step()
        assert self.params[0].shape == (64, 128)
        assert self.params[1].shape == (64, 128)

    def test_state_dict_roundtrip(self):
        """state_dict/load_state_dict should preserve rotations."""
        opt = Taro(
            taro_groups=[("test", [self.params[:2]])],
            lr=0.02, momentum=0.95, sinkhorn_iters=5,
        )
        for p in self.params[:2]:
            p.grad = torch.randn_like(p)
        opt.step()

        sd = opt.state_dict()
        assert "taro_rotations" in sd
        assert "test" in sd["taro_rotations"]

        # Create a new optimizer and load
        opt2 = Taro(
            taro_groups=[("test", [self.params[:2]])],
            lr=0.02, momentum=0.95, sinkhorn_iters=5,
        )
        opt2.load_state_dict(sd)
        assert torch.allclose(opt._rotations["test"], opt2._rotations["test"])

    def test_multiple_group_types(self):
        """Optimizer should handle multiple group types independently."""
        torch.manual_seed(42)
        group_a = [nn.Parameter(torch.randn(32, 64)), nn.Parameter(torch.randn(32, 64))]
        group_b = [nn.Parameter(torch.randn(16, 32)), nn.Parameter(torch.randn(16, 32))]

        opt = Taro(
            taro_groups=[("a", [group_a]), ("b", [group_b])],
            lr=0.02, momentum=0.95, sinkhorn_iters=5,
        )
        assert "a" in opt._rotations
        assert "b" in opt._rotations
        assert opt._rotations["a"].shape == (32, 32)
        assert opt._rotations["b"].shape == (16, 16)

        for p in group_a + group_b:
            p.grad = torch.randn_like(p)
        opt.step()

        # Both rotations should have been updated
        assert not torch.allclose(opt._rotations["a"], torch.eye(32))
        assert not torch.allclose(opt._rotations["b"], torch.eye(16))


class TestConfigureTaroOptimizers:
    def setup_method(self):
        self.cfg = ModelConfig(
            n_layers=4, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
        )
        self.model = GPT(self.cfg)

    def test_returns_two_optimizers(self):
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1, optimizer="taro")
        assert len(opts) == 2
        assert isinstance(opts[0], Taro)
        assert isinstance(opts[1], torch.optim.AdamW)

    def test_all_params_assigned(self):
        """Every trainable param should belong to exactly one optimizer."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1, optimizer="taro")

        opt_param_ids = set()
        for opt in opts:
            for pg in opt.param_groups:
                for p in pg["params"]:
                    opt_param_ids.add(id(p))

        model_param_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        assert model_param_ids == opt_param_ids, "Not all params assigned to an optimizer"

    def test_no_param_overlap(self):
        """No param should be in both Taro and AdamW."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1, optimizer="taro")

        taro_ids = set()
        for pg in opts[0].param_groups:
            for p in pg["params"]:
                taro_ids.add(id(p))

        adamw_ids = set()
        for pg in opts[1].param_groups:
            for p in pg["params"]:
                adamw_ids.add(id(p))

        overlap = taro_ids & adamw_ids
        assert len(overlap) == 0, f"Found {len(overlap)} params in both optimizers"

    def test_qkv_grouped_by_symmetry(self):
        """QKV blocks should contain 6 params each (Q+K+V from 2 layers)."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1, optimizer="taro")
        taro_opt = opts[0]
        assert isinstance(taro_opt, Taro)

        qkv_groups = [(gt, blocks) for gt, blocks in taro_opt._taro_groups if gt == "qkv"]
        assert len(qkv_groups) == 1
        _, blocks = qkv_groups[0]

        # 4 layers, K=2, so 2 blocks of 6 params each
        assert len(blocks) == 2
        for block in blocks:
            assert len(block) == 6, f"QKV block should have 6 params, got {len(block)}"

    def test_all_matrix_weights_in_taro(self):
        """All 2D weight matrices (Q,K,V,O,fc1,fc2) should be in TARO."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1, optimizer="taro")
        taro_opt = opts[0]
        assert isinstance(taro_opt, Taro)

        taro_param_ids = set()
        for pg in taro_opt.param_groups:
            for p in pg["params"]:
                taro_param_ids.add(id(p))

        matrix_patterns = ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                           "out_proj.weight", "fc1.weight", "fc2.weight")
        for name, p in self.model.named_parameters():
            if any(pat in name for pat in matrix_patterns):
                assert id(p) in taro_param_ids, f"{name} should be in TARO"

    def test_muon_path_still_works(self):
        """Default optimizer="muon" should still work as before."""
        from nash_llm.optim.muon import Muon
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1, optimizer="muon")
        assert len(opts) == 2
        assert isinstance(opts[0], Muon)
        assert isinstance(opts[1], torch.optim.AdamW)


class TestTaroEndToEnd:
    def test_training_smoke(self):
        """Smoke test: TARO should reduce loss over a few steps on random data."""
        torch.manual_seed(42)
        cfg = ModelConfig(
            n_layers=2, n_heads=2, d_model=32, d_ff=128,
            vocab_size=100, max_seq_len=16, dropout=0.0,
        )
        model = GPT(cfg)
        opts = configure_optimizers(model, lr=3e-4, weight_decay=0.0, optimizer="taro", muon_lr=0.02)

        x = torch.randint(0, 100, (4, 16))
        y = torch.randint(0, 100, (4, 16))

        losses = []
        for _ in range(10):
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            for opt in opts:
                opt.step()
            losses.append(loss.item())

        # Loss should decrease (or at least not NaN)
        assert not any(torch.isnan(torch.tensor(l)) for l in losses), "Loss should not be NaN"
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
