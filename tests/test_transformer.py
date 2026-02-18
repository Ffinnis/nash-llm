import torch
from nash_llm.model import GPT
from nash_llm.config import ModelConfig


class TestGPT:
    def setup_method(self):
        self.cfg = ModelConfig(
            n_layers=2, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
        )
        self.model = GPT(self.cfg)

    def test_logits_shape(self):
        x = torch.randint(0, 100, (2, 16))
        logits = self.model(x)
        assert logits.shape == (2, 16, 100)

    def test_loss_computation(self):
        x = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        logits, loss = self.model(x, targets)
        assert logits.shape == (2, 16, 100)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_weight_tying(self):
        assert self.model.token_emb.weight is self.model.lm_head.weight

    def test_parameter_count_small(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total > 0
        assert total < 1_000_000

    def test_generate_basic(self):
        self.model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        generated = self.model.generate(prompt, max_new_tokens=10)
        assert generated.shape == (1, 15)
        assert (generated[:, :5] == prompt).all()

    def test_forward_api_unchanged_with_moe(self):
        cfg = ModelConfig(
            n_layers=4,
            n_heads=4,
            d_model=64,
            d_ff=256,
            vocab_size=100,
            max_seq_len=32,
            dropout=0.0,
            moe_enabled=True,
            moe_num_experts=4,
            moe_top_k=2,
            moe_expert_d_ff=64,
            moe_start_layer=0,
            moe_layer_stride=1,
        )
        model = GPT(cfg)
        x = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))

        logits = model(x)
        assert logits.shape == (2, 16, 100)

        logits2, loss = model(x, targets)
        assert logits2.shape == (2, 16, 100)
        assert loss.ndim == 0

    def test_last_moe_metrics_populated(self):
        cfg = ModelConfig(
            n_layers=4,
            n_heads=4,
            d_model=64,
            d_ff=256,
            vocab_size=100,
            max_seq_len=32,
            dropout=0.0,
            moe_enabled=True,
            moe_num_experts=4,
            moe_top_k=2,
            moe_expert_d_ff=64,
            moe_start_layer=0,
            moe_layer_stride=1,
        )
        model = GPT(cfg)
        x = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        _, _ = model(x, targets)

        metrics = model.last_moe_metrics
        assert "aux_loss" in metrics
        assert "z_loss" in metrics
        assert "dropped_frac" in metrics
        assert "expert_entropy" in metrics
        assert metrics["aux_loss"] >= 0.0
        assert metrics["z_loss"] >= 0.0
        assert 0.0 <= metrics["dropped_frac"] <= 1.0
        assert 0.0 <= metrics["expert_entropy"] <= 1.0

        aux_loss, z_loss = model.get_moe_losses()
        assert aux_loss.ndim == 0
        assert z_loss.ndim == 0
