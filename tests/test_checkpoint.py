import torch
import os
import pytest
from nash_llm.model import GPT
from nash_llm.config import ModelConfig, NashConfig
from nash_llm.optim import configure_optimizers
from nash_llm.training.checkpoint import save_checkpoint, load_checkpoint


class TestCheckpoint:
    def setup_method(self):
        self.cfg = ModelConfig(n_layers=2, n_heads=4, d_model=64, d_ff=256, vocab_size=100, max_seq_len=32, dropout=0.0)
        self.model = GPT(self.cfg)
        self.optimizers = configure_optimizers(self.model, lr=3e-4, weight_decay=0.1)

    def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, self.model, self.optimizers, step=100, config=NashConfig(model=self.cfg))
        assert os.path.exists(path)

    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        x = torch.randint(0, 100, (1, 8))
        targets = torch.randint(0, 100, (1, 8))
        _, loss = self.model(x, targets)
        loss.backward()
        for opt in self.optimizers:
            opt.step()

        save_checkpoint(path, self.model, self.optimizers, step=42, config=NashConfig(model=self.cfg), metrics={"val_loss": 3.5})

        model2 = GPT(self.cfg)
        optimizers2 = configure_optimizers(model2, lr=3e-4, weight_decay=0.1)
        ckpt = load_checkpoint(path, model2, optimizers2)

        assert ckpt["step"] == 42
        assert ckpt["metrics"]["val_loss"] == 3.5

        for p1, p2 in zip(self.model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_legacy_single_state_into_single_optimizer_list(self, tmp_path):
        """Legacy checkpoints with a single optimizer state dict can be loaded into a single-optimizer list."""
        path = str(tmp_path / "ckpt.pt")
        # Manually save a legacy-format checkpoint (single optimizer state dict, not a list)
        legacy_opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        x = torch.randint(0, 100, (1, 8))
        targets = torch.randint(0, 100, (1, 8))
        _, loss = self.model(x, targets)
        loss.backward()
        legacy_opt.step()

        torch.save({
            "step": 10,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": legacy_opt.state_dict(),
            "config": {},
            "metrics": {},
        }, path)

        model2 = GPT(self.cfg)
        single_opt = torch.optim.AdamW(model2.parameters(), lr=3e-4)
        load_checkpoint(path, model2, [single_opt])

        assert len(single_opt.state) > 0

    def test_load_single_state_into_multi_optimizer_raises(self, tmp_path):
        """Loading a single-optimizer checkpoint into a multi-optimizer runtime should raise."""
        path = str(tmp_path / "ckpt.pt")
        # Manually save a legacy-format checkpoint (single optimizer state dict)
        legacy_opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        torch.save({
            "step": 10,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": legacy_opt.state_dict(),
            "config": {},
            "metrics": {},
        }, path)

        model2 = GPT(self.cfg)
        optimizers = configure_optimizers(model2, lr=3e-4, weight_decay=0.1)
        assert len(optimizers) == 2

        with pytest.raises(ValueError, match="single optimizer state"):
            load_checkpoint(path, model2, optimizers)

    def test_roundtrip_with_moe_model(self, tmp_path):
        moe_cfg = ModelConfig(
            n_layers=4,
            n_heads=4,
            d_model=64,
            d_ff=256,
            vocab_size=100,
            max_seq_len=32,
            dropout=0.0,
            moe_enabled=True,
            moe_num_experts=4,
            moe_top_k=2,
            moe_expert_d_ff=64,
            moe_start_layer=0,
            moe_layer_stride=1,
        )
        model = GPT(moe_cfg)
        optimizers = configure_optimizers(model, lr=3e-4, weight_decay=0.1)

        path = str(tmp_path / "ckpt_moe.pt")
        x = torch.randint(0, 100, (1, 8))
        targets = torch.randint(0, 100, (1, 8))
        _, loss = model(x, targets)
        loss.backward()
        for opt in optimizers:
            opt.step()

        save_checkpoint(path, model, optimizers, step=7, config=NashConfig(model=moe_cfg))

        model2 = GPT(moe_cfg)
        optimizers2 = configure_optimizers(model2, lr=3e-4, weight_decay=0.1)
        ckpt = load_checkpoint(path, model2, optimizers2)

        assert ckpt["step"] == 7
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
