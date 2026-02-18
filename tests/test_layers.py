import torch
from nash_llm.model.layers import FeedForward, MoEFeedForward
from nash_llm.config import ModelConfig


class TestFeedForward:
    def setup_method(self):
        self.cfg = ModelConfig(d_model=64, d_ff=256, dropout=0.0)
        self.ff = FeedForward(self.cfg)

    def test_output_shape(self):
        x = torch.randn(2, 10, 64)
        out = self.ff(x)
        assert out.shape == (2, 10, 64)

    def test_parameter_count(self):
        params = sum(p.numel() for p in self.ff.parameters())
        # 64*256 + 256 + 256*64 + 64 = 33088
        assert params == 33088


class TestMoEFeedForward:
    def setup_method(self):
        self.cfg = ModelConfig(
            d_model=64,
            d_ff=256,
            dropout=0.0,
            moe_enabled=True,
            moe_num_experts=4,
            moe_top_k=2,
            moe_expert_d_ff=32,
            moe_capacity_factor=1.25,
        )
        self.moe = MoEFeedForward(self.cfg)

    def test_output_shape(self):
        x = torch.randn(2, 10, 64)
        out = self.moe(x)
        assert out.shape == (2, 10, 64)

    def test_aux_and_z_losses_are_finite(self):
        x = torch.randn(2, 8, 64)
        _ = self.moe(x)
        assert self.moe.last_aux_loss is not None
        assert self.moe.last_z_loss is not None
        assert torch.isfinite(self.moe.last_aux_loss)
        assert torch.isfinite(self.moe.last_z_loss)

    def test_capacity_drop_tracking(self):
        cfg = ModelConfig(
            d_model=32,
            d_ff=128,
            dropout=0.0,
            moe_enabled=True,
            moe_num_experts=2,
            moe_top_k=2,
            moe_expert_d_ff=16,
            moe_capacity_factor=0.25,
        )
        moe = MoEFeedForward(cfg)
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.bias.copy_(torch.tensor([10.0, -10.0]))

        x = torch.randn(4, 16, 32)
        _ = moe(x)
        dropped = moe.last_moe_metrics["dropped_frac"]
        assert 0.0 <= dropped <= 1.0
        assert dropped > 0.0
