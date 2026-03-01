import torch
from nash_llm.model import GPT
from nash_llm.config import ModelConfig
from nash_llm.model.norms import RMSNorm


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

    def test_uses_rmsnorm_when_configured(self):
        cfg = ModelConfig(
            n_layers=2, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
            norm_type="rmsnorm",
        )
        model = GPT(cfg)
        assert isinstance(model.blocks[0].ln1, RMSNorm)
        assert isinstance(model.blocks[0].ln2, RMSNorm)
        assert isinstance(model.ln_f, RMSNorm)

    def test_uses_layernorm_by_default(self):
        cfg = ModelConfig(
            n_layers=2, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
            norm_type="layernorm",
        )
        model = GPT(cfg)
        assert model.blocks[0].ln1.__class__.__name__ == "LayerNorm"
        assert model.blocks[0].ln2.__class__.__name__ == "LayerNorm"
        assert model.ln_f.__class__.__name__ == "LayerNorm"

    def test_tie_embeddings_true_shares_weights(self):
        cfg = ModelConfig(
            n_layers=2, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
            tie_embeddings=True,
        )
        model = GPT(cfg)
        assert model.token_emb.weight is model.lm_head.weight

    def test_tie_embeddings_false_unties_weights(self):
        cfg = ModelConfig(
            n_layers=2, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
            tie_embeddings=False,
        )
        model = GPT(cfg)
        assert model.token_emb.weight is not model.lm_head.weight
