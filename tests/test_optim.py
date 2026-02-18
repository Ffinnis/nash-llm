import torch
import pytest
import torch.nn as nn
from nash_llm.model import GPT
from nash_llm.config import ModelConfig
from nash_llm.optim import configure_optimizer


class _SingleWeight(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([[value]], dtype=torch.float32))


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

    def test_fused_false_is_allowed(self):
        opt = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1, fused=False)
        assert isinstance(opt, torch.optim.AdamW)

    def test_fused_true_raises(self):
        with pytest.raises(ValueError, match="unsupported for NesterovAdamW"):
            configure_optimizer(self.model, lr=3e-4, weight_decay=0.1, fused=True)

    def test_nesterov_single_step_matches_manual_formula(self):
        model = _SingleWeight(1.0)
        lr = 1e-3
        betas = (0.9, 0.95)
        eps = 1e-8
        weight_decay = 0.1
        opt = configure_optimizer(model, lr=lr, weight_decay=weight_decay, betas=betas, fused=False)

        grad = torch.tensor([[0.2]], dtype=torch.float32)
        model.w.grad = grad.clone()
        initial = model.w.detach().clone()
        opt.step()

        beta1, beta2 = betas
        step = 1
        decayed = initial * (1.0 - lr * weight_decay)
        m = (1.0 - beta1) * grad
        v = (1.0 - beta2) * grad * grad
        m_bar = beta1 * m + (1.0 - beta1) * grad
        m_hat = m_bar / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        expected = decayed - lr * m_hat / (v_hat.sqrt() + eps)

        assert torch.allclose(model.w.detach(), expected, atol=1e-8)

    def test_exp_avg_and_exp_avg_sq_follow_adamw_ema(self):
        model = _SingleWeight(0.5)
        betas = (0.9, 0.95)
        opt = configure_optimizer(model, lr=1e-3, weight_decay=0.0, betas=betas, fused=False)
        beta1, beta2 = betas
        grads = [torch.tensor([[0.2]], dtype=torch.float32), torch.tensor([[-0.1]], dtype=torch.float32)]

        exp_avg = torch.zeros_like(model.w)
        exp_avg_sq = torch.zeros_like(model.w)
        belief_sq = torch.zeros_like(model.w)
        for grad in grads:
            model.w.grad = grad.clone()
            opt.step()
            exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
            exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
            belief_sq = beta2 * belief_sq + (1.0 - beta2) * (grad - exp_avg) * (grad - exp_avg)

        state = opt.state[model.w]
        assert torch.allclose(state["exp_avg"], exp_avg, atol=1e-8)
        assert torch.allclose(state["exp_avg_sq"], exp_avg_sq, atol=1e-8)
        assert not torch.allclose(state["exp_avg_sq"], belief_sq, atol=1e-8)
