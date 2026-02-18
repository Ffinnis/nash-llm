import torch
import os
import pytest
from nash_llm.model import GPT
from nash_llm.config import ModelConfig, NashConfig
from nash_llm.optim import configure_optimizer, configure_optimizers
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

    def test_load_legacy_single_state_into_single_optimizer_list(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        x = torch.randint(0, 100, (1, 8))
        targets = torch.randint(0, 100, (1, 8))
        _, loss = self.model(x, targets)
        loss.backward()
        self.optimizer.step()
        save_checkpoint(path, self.model, self.optimizer, step=10, config=NashConfig(model=self.cfg))

        model2 = GPT(self.cfg)
        optimizers = configure_optimizers(model2, "adamw", lr=3e-4, weight_decay=0.1)
        assert len(optimizers) == 1
        assert len(optimizers[0].state) == 0

        load_checkpoint(path, model2, optimizers)

        assert len(optimizers[0].state) > 0

    def test_load_single_state_into_multi_optimizer_raises(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, self.model, self.optimizer, step=10, config=NashConfig(model=self.cfg))

        model2 = GPT(self.cfg)
        optimizers = configure_optimizers(model2, "muon", lr=3e-4, weight_decay=0.1)
        assert len(optimizers) == 2

        with pytest.raises(ValueError, match="single optimizer state"):
            load_checkpoint(path, model2, optimizers)

    def test_load_vanilla_adamw_state_into_nesterov_adamw(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        model1 = GPT(self.cfg)
        decay_params = []
        no_decay_params = []
        for _, p in model1.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        vanilla_opt = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 0.1},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=3e-4,
            betas=(0.9, 0.95),
        )

        x = torch.randint(0, 100, (1, 8))
        targets = torch.randint(0, 100, (1, 8))
        _, loss = model1(x, targets)
        loss.backward()
        vanilla_opt.step()
        save_checkpoint(path, model1, vanilla_opt, step=7, config=NashConfig(model=self.cfg))

        model2 = GPT(self.cfg)
        nesterov_opt = configure_optimizer(model2, lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
        load_checkpoint(path, model2, nesterov_opt)
        assert len(nesterov_opt.state) > 0

        x2 = torch.randint(0, 100, (1, 8))
        targets2 = torch.randint(0, 100, (1, 8))
        _, loss2 = model2(x2, targets2)
        loss2.backward()
        nesterov_opt.step()
