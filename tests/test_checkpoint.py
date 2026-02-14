import torch
import os
from nash_llm.model import GPT
from nash_llm.config import ModelConfig, NashConfig
from nash_llm.optim import configure_optimizer
from nash_llm.training.checkpoint import save_checkpoint, load_checkpoint


class TestCheckpoint:
    def setup_method(self):
        self.cfg = ModelConfig(n_layers=2, n_heads=4, d_model=64, d_ff=256, vocab_size=100, max_seq_len=32, dropout=0.0)
        self.model = GPT(self.cfg)
        self.optimizer = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1)

    def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, self.model, self.optimizer, step=100, config=NashConfig(model=self.cfg))
        assert os.path.exists(path)

    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        x = torch.randint(0, 100, (1, 8))
        targets = torch.randint(0, 100, (1, 8))
        _, loss = self.model(x, targets)
        loss.backward()
        self.optimizer.step()

        save_checkpoint(path, self.model, self.optimizer, step=42, config=NashConfig(model=self.cfg), metrics={"val_loss": 3.5})

        model2 = GPT(self.cfg)
        optimizer2 = configure_optimizer(model2, lr=3e-4, weight_decay=0.1)
        ckpt = load_checkpoint(path, model2, optimizer2)

        assert ckpt["step"] == 42
        assert ckpt["metrics"]["val_loss"] == 3.5

        for p1, p2 in zip(self.model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
