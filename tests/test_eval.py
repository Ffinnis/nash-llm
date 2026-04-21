import torch
import numpy as np
import json
import pytest
from nash_llm.model import GPT
from nash_llm.config import ModelConfig
from nash_llm.data.dataset import PretrainDataset
from nash_llm.eval.evaluate import compute_val_loss, compute_accuracy
from nash_llm.eval.generate import generate_text


class TestEvaluate:
    def setup_method(self):
        self.cfg = ModelConfig(n_layers=2, n_heads=4, d_model=64, d_ff=256, vocab_size=100, max_seq_len=32, dropout=0.0)
        self.model = GPT(self.cfg)
        self.model.eval()

    def _make_val_loader(self, tmp_path, n_tokens=500, seq_len=32, batch_size=4):
        tokens = np.random.randint(0, 100, size=n_tokens, dtype=np.uint16)
        shard_path = tmp_path / "val_000.bin"
        tokens.tofile(str(shard_path))
        meta = {"vocab_size": 100, "total_tokens": n_tokens}
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        ds = PretrainDataset(str(tmp_path), split="val", seq_len=seq_len)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size)

    def test_val_loss_is_positive(self, tmp_path):
        loader = self._make_val_loader(tmp_path)
        loss = compute_val_loss(self.model, loader, max_batches=3)
        assert loss > 0

    def test_accuracy_between_0_and_1(self, tmp_path):
        loader = self._make_val_loader(tmp_path)
        acc = compute_accuracy(self.model, loader, max_batches=3)
        assert 0.0 <= acc <= 1.0

    def test_val_loss_weights_tokens_not_batches(self, tmp_path):
        loader = self._make_val_loader(tmp_path, n_tokens=330, seq_len=32, batch_size=4)

        losses = []
        token_counts = []
        device = next(self.model.parameters()).device
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = self.model(x, y)
            losses.append(loss.item())
            token_counts.append(y.numel())

        weighted = sum(loss * tokens for loss, tokens in zip(losses, token_counts)) / sum(token_counts)
        unweighted = sum(losses) / len(losses)
        computed = compute_val_loss(self.model, loader)

        assert computed == pytest.approx(weighted)
        assert abs(weighted - unweighted) > 1e-6


class TestGenerate:
    def test_generate_returns_string(self):
        cfg = ModelConfig(n_layers=2, n_heads=4, d_model=64, d_ff=256, vocab_size=50257, max_seq_len=32, dropout=0.0)
        model = GPT(cfg)
        model.eval()
        text = generate_text(model, prompt="Hello", max_new_tokens=10)
        assert isinstance(text, str)
        assert len(text) > len("Hello")
