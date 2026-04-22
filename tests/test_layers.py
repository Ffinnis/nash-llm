import torch
import torch.nn as nn
from nash_llm.model.layers import FeedForward
from nash_llm.config import ModelConfig


class TestFeedForward:
    def setup_method(self):
        self.cfg = ModelConfig(d_model=64, d_ff=256, dropout=0.0)
        self.ff = FeedForward(self.cfg)

    def test_default_uses_swiglu_with_two_thirds_hidden_dim(self):
        assert isinstance(self.ff.gate_act, nn.SiLU)
        assert self.ff.fc1.out_features == 170
        assert self.ff.fc_gate.out_features == 170
        assert self.ff.fc2.in_features == 170

    def test_gelu_mode_uses_gelu(self):
        ff = FeedForward(ModelConfig(d_model=64, d_ff=256, dropout=0.0, activation="gelu"))
        assert isinstance(ff.act, nn.GELU)
        assert ff.fc1.out_features == 256
        assert ff.fc_gate is None

    def test_swiglu_matches_reference_formula(self):
        ff = FeedForward(ModelConfig(d_model=4, d_ff=6, dropout=0.0, activation="swiglu"))
        with torch.no_grad():
            ff.fc1.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))
            ff.fc1.bias.zero_()
            ff.fc_gate.weight.copy_(torch.tensor([[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.5]]))
            ff.fc_gate.bias.zero_()
            ff.fc2.weight.copy_(torch.eye(4))
            ff.fc2.bias.zero_()
        x = torch.tensor([[[1.0, -2.0, 0.5, -0.25]]])
        expected = nn.functional.silu(ff.fc1(x)) * ff.fc_gate(x)
        out = ff(x)
        assert torch.allclose(out, expected)

    def test_output_shape(self):
        x = torch.randn(2, 10, 64)
        out = self.ff(x)
        assert out.shape == (2, 10, 64)

    def test_parameter_count(self):
        params = sum(p.numel() for p in self.ff.parameters())
        # SwiGLU keeps hidden width near 2/3 * d_ff to stay close to the
        # original FFN parameter budget.
        assert params == 33044
