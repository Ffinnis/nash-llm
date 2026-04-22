import torch
import torch.nn as nn
from nash_llm.model.layers import FeedForward
from nash_llm.config import ModelConfig


class TestFeedForward:
    def setup_method(self):
        self.cfg = ModelConfig(d_model=64, d_ff=256, dropout=0.0)
        self.ff = FeedForward(self.cfg)

    def test_default_uses_relu2(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])
        expected = torch.square(torch.relu(x))
        assert torch.equal(self.ff.act(x), expected)

    def test_gelu_mode_uses_gelu(self):
        ff = FeedForward(ModelConfig(d_model=64, d_ff=256, dropout=0.0, activation="gelu"))
        assert isinstance(ff.act, nn.GELU)

    def test_output_shape(self):
        x = torch.randn(2, 10, 64)
        out = self.ff(x)
        assert out.shape == (2, 10, 64)

    def test_parameter_count(self):
        params = sum(p.numel() for p in self.ff.parameters())
        # 64*256 + 256 + 256*64 + 64 = 33088
        assert params == 33088
