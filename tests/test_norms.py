import torch

from nash_llm.model.norms import RMSNorm


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64, eps=1e-6)
        x = torch.randn(2, 10, 64)
        y = norm(x)
        assert y.shape == x.shape

    def test_has_learnable_weight(self):
        norm = RMSNorm(32, eps=1e-6)
        assert norm.weight.shape == (32,)
        assert norm.weight.requires_grad

    def test_rms_is_unit_when_weight_is_one(self):
        torch.manual_seed(0)
        norm = RMSNorm(16, eps=1e-6)
        with torch.no_grad():
            norm.weight.fill_(1.0)
        x = torch.randn(8, 16)
        y = norm(x)
        rms = torch.sqrt((y.pow(2)).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
