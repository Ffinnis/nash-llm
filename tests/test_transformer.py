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
