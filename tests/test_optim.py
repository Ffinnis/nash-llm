import torch
import pytest
import torch.nn as nn
from nash_llm.model import GPT
from nash_llm.config import ModelConfig
from nash_llm.optim import configure_optimizer


class TestConfigureOptimizer:
    def setup_method(self):
        self.cfg = ModelConfig(n_layers=2, n_heads=4, d_model=64, d_ff=256, vocab_size=100, max_seq_len=32, dropout=0.0)
        self.model = GPT(self.cfg)

    def test_returns_nadam(self):
        opt = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1)
        assert isinstance(opt, torch.optim.NAdam)

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

    def test_decoupled_weight_decay_enabled(self):
        opt = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1)
        assert opt.defaults.get("decoupled_weight_decay") is True
