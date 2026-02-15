import torch
import pytest
from unittest.mock import patch
from nash_llm.model import GPT
from nash_llm.config import ModelConfig
from nash_llm.optim import configure_optimizer


class TestConfigureOptimizer:
    def setup_method(self):
        self.cfg = ModelConfig(n_layers=2, n_heads=4, d_model=64, d_ff=256, vocab_size=100, max_seq_len=32, dropout=0.0)
        self.model = GPT(self.cfg)

    def test_returns_adamw(self):
        opt = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1)
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups(self):
        opt = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1)
        assert len(opt.param_groups) == 2
        wds = {pg["weight_decay"] for pg in opt.param_groups}
        assert 0.0 in wds
        assert 0.1 in wds

    def test_no_decay_for_biases_and_layernorm(self):
        opt = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1)
        no_decay_group = [pg for pg in opt.param_groups if pg["weight_decay"] == 0.0][0]
        no_decay_count = sum(p.numel() for p in no_decay_group["params"])
        assert no_decay_count > 0

    @patch("torch.cuda.is_available", return_value=False)
    def test_fused_false_is_allowed_without_cuda(self, _mock_cuda):
        opt = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1, fused=False)
        assert isinstance(opt, torch.optim.AdamW)

    @patch("torch.cuda.is_available", return_value=False)
    def test_fused_true_without_cuda_raises(self, _mock_cuda):
        with pytest.raises(ValueError, match="Fused AdamW requested"):
            configure_optimizer(self.model, lr=3e-4, weight_decay=0.1, fused=True)
