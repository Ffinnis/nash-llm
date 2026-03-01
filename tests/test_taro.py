import torch
import pytest
import torch.nn as nn

from nash_llm.model import GPT
from nash_llm.config import ModelConfig
from nash_llm.optim.taro import _sinkhorn_impl, _cholqr_impl, Taro, TaroGroup, TaroParamRef
from nash_llm.optim.adamw import configure_optimizers


class TestSinkhorn:
    def test_output_shape(self):
        torch.manual_seed(42)
        x = torch.randn(64, 128)
        result = _sinkhorn_impl(x, n_iters=5)
        assert result.shape == (64, 128)

    def test_row_and_col_norms_converge(self):
        torch.manual_seed(42)
        x = torch.randn(32, 64)
        result = _sinkhorn_impl(x, n_iters=10).float()
        row_norms = result.norm(dim=-1)
        col_norms = result.norm(dim=-2)
        assert row_norms.std() < 0.01
        assert col_norms.std() < 0.01

    def test_zero_input_safety(self):
        x = torch.zeros(16, 32)
        result = _sinkhorn_impl(x, n_iters=5)
        assert not torch.isnan(result).any()


class TestCholQR:
    def test_orthogonality(self):
        torch.manual_seed(42)
        a = torch.randn(32, 32)
        q = _cholqr_impl(a)
        eye = torch.eye(32)
        err = (q @ q.mT - eye).norm()
        assert err < 0.5

    def test_batched(self):
        torch.manual_seed(42)
        a = torch.randn(3, 16, 16)
        q = _cholqr_impl(a)
        assert q.shape == (3, 16, 16)


class TestTaroOptimizer:
    def setup_method(self):
        torch.manual_seed(42)
        self.p1 = nn.Parameter(torch.randn(64, 128))
        self.p2 = nn.Parameter(torch.randn(64, 128))

    @staticmethod
    def _simple_group(p1: nn.Parameter, p2: nn.Parameter) -> list[TaroGroup]:
        return [
            TaroGroup(
                group_name="test",
                blocks=[[
                    TaroParamRef(p1, transpose_in_taro=False, lr_mult=1.0, group_name="test"),
                    TaroParamRef(p2, transpose_in_taro=False, lr_mult=1.0, group_name="test"),
                ]],
                lr_mult=1.0,
            ),
        ]

    def test_step_updates_weights(self):
        original1 = self.p1.clone()
        original2 = self.p2.clone()
        opt = Taro(taro_groups=self._simple_group(self.p1, self.p2), lr=0.02, momentum=0.95, sinkhorn_iters=5)
        self.p1.grad = torch.randn_like(self.p1)
        self.p2.grad = torch.randn_like(self.p2)
        opt.step()
        assert not torch.allclose(original1, self.p1)
        assert not torch.allclose(original2, self.p2)

    def test_rotation_updates_from_identity(self):
        opt = Taro(taro_groups=self._simple_group(self.p1, self.p2), lr=0.02, momentum=0.95, sinkhorn_iters=5)
        before = opt._rotations["test"].clone()
        assert torch.allclose(before, torch.eye(64))
        self.p1.grad = torch.randn_like(self.p1)
        self.p2.grad = torch.randn_like(self.p2)
        opt.step()
        after = opt._rotations["test"]
        assert not torch.allclose(after, torch.eye(64), atol=1e-3)

    def test_rotation_stays_orthogonal(self):
        opt = Taro(taro_groups=self._simple_group(self.p1, self.p2), lr=0.02, momentum=0.95, sinkhorn_iters=5)
        for _ in range(5):
            self.p1.grad = torch.randn_like(self.p1)
            self.p2.grad = torch.randn_like(self.p2)
            opt.step()
        r = opt._rotations["test"].float()
        eye = torch.eye(r.shape[0])
        err = (r @ r.mT - eye).norm() / eye.norm()
        assert err < 0.1

    def test_uses_r_new_for_delta_z(self, monkeypatch):
        import nash_llm.optim.taro as taro_mod

        p = nn.Parameter(torch.zeros(2, 2))
        group = [
            TaroGroup(
                group_name="g",
                blocks=[[TaroParamRef(p, transpose_in_taro=False, lr_mult=1.0, group_name="g")]],
                lr_mult=1.0,
            ),
        ]
        opt = Taro(taro_groups=group, lr=1.0, momentum=0.0, sinkhorn_iters=0)

        r_swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        monkeypatch.setattr(taro_mod, "cholqr", lambda a: r_swap.to(device=a.device))
        p.grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        opt.step()

        s = p.grad.to(torch.bfloat16).float()
        expected = r_swap @ s
        expected = expected * ((2 * 2) ** 0.5 / expected.norm())
        assert torch.allclose(p.data, -expected, atol=2e-2)

    def test_state_dict_roundtrip(self):
        opt = Taro(taro_groups=self._simple_group(self.p1, self.p2), lr=0.02, momentum=0.95, sinkhorn_iters=5)
        self.p1.grad = torch.randn_like(self.p1)
        self.p2.grad = torch.randn_like(self.p2)
        opt.step()
        sd = opt.state_dict()
        opt2 = Taro(taro_groups=self._simple_group(self.p1, self.p2), lr=0.02, momentum=0.95, sinkhorn_iters=5)
        opt2.load_state_dict(sd)
        assert torch.allclose(opt._rotations["test"], opt2._rotations["test"])


class TestConfigureTaroOptimizers:
    def setup_method(self):
        self.cfg = ModelConfig(
            n_layers=4, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
            tie_embeddings=False,
        )
        self.model = GPT(self.cfg)

    def _get_taro(self) -> Taro:
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1, optimizer="taro")
        assert isinstance(opts[0], Taro)
        return opts[0]

    def test_returns_two_optimizers(self):
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1, optimizer="taro")
        assert len(opts) == 2
        assert isinstance(opts[0], Taro)
        assert isinstance(opts[1], torch.optim.AdamW)

    def test_all_params_assigned(self):
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1, optimizer="taro")
        opt_param_ids = set()
        for opt in opts:
            for pg in opt.param_groups:
                for p in pg["params"]:
                    opt_param_ids.add(id(p))
        model_param_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        assert model_param_ids == opt_param_ids

    def test_no_param_overlap(self):
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1, optimizer="taro")
        taro_ids = {id(p) for pg in opts[0].param_groups for p in pg["params"]}
        adamw_ids = {id(p) for pg in opts[1].param_groups for p in pg["params"]}
        assert len(taro_ids & adamw_ids) == 0

    def test_qkv_grouped_by_symmetry(self):
        taro_opt = self._get_taro()
        qkv = [g for g in taro_opt._taro_groups if g.group_name == "qkv"]
        assert len(qkv) == 1
        blocks = qkv[0].blocks
        assert len(blocks) == 2
        for block in blocks:
            assert len(block) == 6
            assert all(not ref.transpose_in_taro for ref in block)

    def test_o_up_group_uses_transpose_metadata(self):
        taro_opt = self._get_taro()
        groups = [g for g in taro_opt._taro_groups if g.group_name == "o_up"]
        assert len(groups) == 1
        blocks = groups[0].blocks
        assert len(blocks) == 2
        for block in blocks:
            assert len(block) == 4
            names = [ref.group_name for ref in block]
            assert names == ["o_up", "o_up", "o_up", "o_up"]
            transposed = [ref.transpose_in_taro for ref in block]
            assert transposed == [False, True, False, True]

    def test_down_group_has_half_lr_multiplier(self):
        taro_opt = self._get_taro()
        down = [g for g in taro_opt._taro_groups if g.group_name == "down"]
        assert len(down) == 1
        assert down[0].lr_mult == 0.5
        assert all(ref.lr_mult == 0.5 for block in down[0].blocks for ref in block)

    def test_all_matrix_weights_in_taro(self):
        taro_opt = self._get_taro()
        taro_param_ids = {id(p) for pg in taro_opt.param_groups for p in pg["params"]}
        matrix_patterns = (
            "q_proj.weight", "k_proj.weight", "v_proj.weight",
            "out_proj.weight", "fc1.weight", "fc2.weight",
        )
        for name, p in self.model.named_parameters():
            if any(pat in name for pat in matrix_patterns):
                assert id(p) in taro_param_ids

    def test_muon_path_still_works(self):
        from nash_llm.optim.muon import Muon

        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1, optimizer="muon")
        assert len(opts) == 2
        assert isinstance(opts[0], Muon)
        assert isinstance(opts[1], torch.optim.AdamW)


class TestTaroEndToEnd:
    def test_training_smoke(self):
        torch.manual_seed(42)
        cfg = ModelConfig(
            n_layers=2, n_heads=2, d_model=32, d_ff=128,
            vocab_size=100, max_seq_len=16, dropout=0.0,
            tie_embeddings=False,
        )
        model = GPT(cfg)
        opts = configure_optimizers(
            model, lr=3e-4, weight_decay=0.0, optimizer="taro",
            muon_lr=0.02, taro_k=2, taro_sinkhorn_iters=5,
        )

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

        assert not any(torch.isnan(torch.tensor(v)) for v in losses)
        assert losses[-1] < losses[0]
