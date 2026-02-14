import torch
from nash_llm.model.layers import FeedForward
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
