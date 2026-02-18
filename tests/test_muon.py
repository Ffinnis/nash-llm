import torch
import pytest
import torch.nn as nn
from nash_llm.model import GPT
from nash_llm.config import ModelConfig
from nash_llm.optim.muon import _polar_express_impl, orthogonalize, Muon
from nash_llm.optim.adamw import configure_optimizers


class TestPolarExpress:
    def test_approximates_polar_decomposition(self):
        """polar_express(M) should approximate UV^T from SVD."""
        torch.manual_seed(42)
        M = torch.randn(64, 128)
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        exact_polar = U @ Vt

        # Use 7 steps for tighter convergence; bf16 limits precision to ~0.1 relative error
        approx = _polar_express_impl(M, steps=7)  # bypass @torch.compile

        error = (exact_polar.float() - approx.float()).norm() / exact_polar.float().norm()
        assert error < 0.1, f"Relative error {error:.4f} too large"

    def test_tall_matrix_transpose(self):
        """Tall matrices (m > n) should be handled via transpose."""
        torch.manual_seed(42)
        M = torch.randn(128, 64)  # tall
        result = _polar_express_impl(M, steps=5)
        assert result.shape == (128, 64)

    def test_square_matrix(self):
        torch.manual_seed(42)
        M = torch.randn(64, 64)
        result = _polar_express_impl(M, steps=5)
        assert result.shape == (64, 64)

    def test_batched_matches_individual(self):
        """Batched 3D polar_express should match independent 2D calls."""
        torch.manual_seed(42)
        M1 = torch.randn(64, 128)
        M2 = torch.randn(64, 128)

        r1 = _polar_express_impl(M1, steps=5)
        r2 = _polar_express_impl(M2, steps=5)

        batch = torch.stack([M1, M2])
        rb = _polar_express_impl(batch, steps=5)

        assert torch.allclose(r1, rb[0], atol=1e-5)
        assert torch.allclose(r2, rb[1], atol=1e-5)

    def test_orthogonality_of_result(self):
        """Result should have orthonormal rows (for wide matrix)."""
        torch.manual_seed(42)
        M = torch.randn(32, 64)
        result = _polar_express_impl(M, steps=6).float()

        # Q @ Q^T should approximate I
        product = result @ result.T
        eye = torch.eye(32)
        error = (product - eye).norm()
        assert error < 0.5, f"Orthogonality error {error:.4f}"


class TestMuonOptimizer:
    def setup_method(self):
        torch.manual_seed(42)
        self.params = [nn.Parameter(torch.randn(64, 128)) for _ in range(3)]

    def test_muon_step_updates_weights(self):
        """A single Muon step should change the parameters."""
        original = [p.clone() for p in self.params]

        opt = Muon(
            muon_params=self.params,
            teon_params=[],
            lr=0.02,
            momentum=0.95,
            ns_steps=5,
        )

        # Simulate gradients
        for p in self.params:
            p.grad = torch.randn_like(p)

        opt.step()

        for orig, updated in zip(original, self.params):
            assert not torch.allclose(orig, updated), "Params should have changed"

    def test_muon_momentum_accumulation(self):
        """Momentum buffer should accumulate across steps."""
        opt = Muon(
            muon_params=self.params[:1],
            teon_params=[],
            lr=0.02,
            momentum=0.95,
            ns_steps=5,
        )

        self.params[0].grad = torch.randn_like(self.params[0])
        opt.step()
        buf_after_1 = opt.state[self.params[0]]["momentum_buffer"].clone()

        self.params[0].grad = torch.randn_like(self.params[0])
        opt.step()
        buf_after_2 = opt.state[self.params[0]]["momentum_buffer"]

        assert not torch.allclose(buf_after_1, buf_after_2)


class TestTEONStacking:
    def test_teon_stacking_updates_all_params(self):
        """TEON should update all K params in a group."""
        torch.manual_seed(42)
        p1 = nn.Parameter(torch.randn(64, 128))
        p2 = nn.Parameter(torch.randn(64, 128))
        original = [p1.clone(), p2.clone()]

        opt = Muon(
            muon_params=[],
            teon_params=[[p1, p2]],
            lr=0.02,
            momentum=0.95,
            ns_steps=5,
        )

        p1.grad = torch.randn_like(p1)
        p2.grad = torch.randn_like(p2)
        opt.step()

        assert not torch.allclose(original[0], p1)
        assert not torch.allclose(original[1], p2)

    def test_teon_stacked_ortho_differs_from_independent(self):
        """TEON stacking should produce different updates than per-layer MUON."""
        torch.manual_seed(42)
        p1 = nn.Parameter(torch.randn(64, 128))
        p2 = nn.Parameter(torch.randn(64, 128))
        g1 = torch.randn(64, 128)
        g2 = torch.randn(64, 128)

        # TEON path: stack and orthogonalize jointly
        p1_teon = nn.Parameter(p1.detach().clone())
        p2_teon = nn.Parameter(p2.detach().clone())
        opt_teon = Muon(muon_params=[], teon_params=[[p1_teon, p2_teon]], lr=0.02, momentum=0.0, ns_steps=5)
        p1_teon.grad = g1.clone()
        p2_teon.grad = g2.clone()
        opt_teon.step()

        # MUON path: orthogonalize independently
        p1_muon = nn.Parameter(p1.detach().clone())
        p2_muon = nn.Parameter(p2.detach().clone())
        opt_muon = Muon(muon_params=[p1_muon, p2_muon], teon_params=[], lr=0.02, momentum=0.0, ns_steps=5)
        p1_muon.grad = g1.clone()
        p2_muon.grad = g2.clone()
        opt_muon.step()

        # They should differ because stacking changes the orthogonalization
        assert not torch.allclose(p1_teon, p1_muon, atol=1e-3), "TEON should differ from MUON"

    def test_teon_group_shapes_preserved(self):
        """After TEON step, param shapes should be unchanged."""
        torch.manual_seed(42)
        p1 = nn.Parameter(torch.randn(32, 64))
        p2 = nn.Parameter(torch.randn(32, 64))

        opt = Muon(muon_params=[], teon_params=[[p1, p2]], lr=0.02, momentum=0.95, ns_steps=5)
        p1.grad = torch.randn_like(p1)
        p2.grad = torch.randn_like(p2)
        opt.step()

        assert p1.shape == (32, 64)
        assert p2.shape == (32, 64)


class TestConfigureOptimizers:
    def setup_method(self):
        self.cfg = ModelConfig(n_layers=4, n_heads=4, d_model=64, d_ff=256, vocab_size=100, max_seq_len=32, dropout=0.0)
        self.model = GPT(self.cfg)

    def test_returns_two_optimizers(self):
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1)
        assert len(opts) == 2
        assert isinstance(opts[0], Muon)
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
        """No param should be in both Muon and AdamW."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1)

        muon_ids = set()
        for pg in opts[0].param_groups:
            for p in pg["params"]:
                muon_ids.add(id(p))

        adamw_ids = set()
        for pg in opts[1].param_groups:
            for p in pg["params"]:
                adamw_ids.add(id(p))

        overlap = muon_ids & adamw_ids
        assert len(overlap) == 0, f"Found {len(overlap)} params in both optimizers"

    def test_teon_has_qkv_params(self):
        """TEON groups should contain q_proj, k_proj, v_proj weights."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1)
        muon_opt = opts[0]
        assert isinstance(muon_opt, Muon)

        teon_param_ids = set()
        for group in muon_opt._teon_groups:
            for p in group:
                teon_param_ids.add(id(p))

        qkv_param_ids = set()
        for name, p in self.model.named_parameters():
            if any(pat in name for pat in ["q_proj.weight", "k_proj.weight", "v_proj.weight"]):
                qkv_param_ids.add(id(p))

        assert qkv_param_ids == teon_param_ids, "TEON should contain exactly the Q/K/V params"

    def test_teon_groups_have_k2(self):
        """Each TEON group should have K=2 params."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1)
        muon_opt = opts[0]
        assert isinstance(muon_opt, Muon)

        for group in muon_opt._teon_groups:
            assert len(group) == 2, f"TEON group should have K=2, got {len(group)}"

    def test_teon_group_order_is_deterministic(self):
        """TEON groups must be in a stable Q->K->V order for state_dict compatibility."""
        opts = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1)
        muon_opt = opts[0]
        assert isinstance(muon_opt, Muon)
        name_by_id = {id(p): name for name, p in self.model.named_parameters()}

        group_names = []
        for group in muon_opt._teon_groups:
            group_names.append(tuple(name_by_id[id(p)] for p in group))

        expected = [
            ("blocks.0.attn.q_proj.weight", "blocks.1.attn.q_proj.weight"),
            ("blocks.2.attn.q_proj.weight", "blocks.3.attn.q_proj.weight"),
            ("blocks.0.attn.k_proj.weight", "blocks.1.attn.k_proj.weight"),
            ("blocks.2.attn.k_proj.weight", "blocks.3.attn.k_proj.weight"),
            ("blocks.0.attn.v_proj.weight", "blocks.1.attn.v_proj.weight"),
            ("blocks.2.attn.v_proj.weight", "blocks.3.attn.v_proj.weight"),
        ]
        assert group_names == expected


class TestAttentionSplitQKV:
    def test_output_shape_preserved(self):
        """Split QKV attention should produce same output shape."""
        from nash_llm.model.attention import MultiHeadAttention
        cfg = ModelConfig(d_model=64, n_heads=4, max_seq_len=32, dropout=0.0)
        attn = MultiHeadAttention(cfg)

        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_has_separate_projections(self):
        """Attention should have q_proj, k_proj, v_proj instead of qkv."""
        from nash_llm.model.attention import MultiHeadAttention
        cfg = ModelConfig(d_model=64, n_heads=4, max_seq_len=32, dropout=0.0)
        attn = MultiHeadAttention(cfg)

        assert hasattr(attn, "q_proj")
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "v_proj")
        assert not hasattr(attn, "qkv")

    def test_param_names_correct(self):
        """GPT model should have blocks.{i}.attn.q_proj.weight naming."""
        cfg = ModelConfig(n_layers=2, n_heads=4, d_model=64, d_ff=256, vocab_size=100, max_seq_len=32, dropout=0.0)
        model = GPT(cfg)

        param_names = {name for name, _ in model.named_parameters()}
        assert "blocks.0.attn.q_proj.weight" in param_names
        assert "blocks.0.attn.k_proj.weight" in param_names
        assert "blocks.0.attn.v_proj.weight" in param_names
        assert "blocks.0.attn.q_proj.bias" in param_names
        assert "blocks.0.attn.qkv.weight" not in param_names
