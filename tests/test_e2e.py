"""End-to-end smoke test: prepare fake data, train a tiny model, evaluate, generate."""
import numpy as np
import json
import torch
from nash_llm.config import NashConfig, ModelConfig, TrainConfig, DataConfig, MetricsConfig
from nash_llm.training import Trainer
from nash_llm.training.checkpoint import load_checkpoint
from nash_llm.model import GPT
from nash_llm.eval import compute_val_loss


class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        # 1. Create fake tokenized data
        n_tokens = 5000
        tokens = np.random.randint(0, 500, size=n_tokens, dtype=np.uint16)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train_000.bin").write_bytes(tokens[:4500].tobytes())
        (data_dir / "val_000.bin").write_bytes(tokens[4500:].tobytes())
        (data_dir / "meta.json").write_text(json.dumps({"vocab_size": 500, "total_tokens": n_tokens}))

        # 2. Train a tiny model
        cfg = NashConfig(
            model=ModelConfig(n_layers=2, n_heads=2, d_model=32, d_ff=128, vocab_size=500, max_seq_len=32, dropout=0.0),
            train=TrainConfig(batch_size=4, learning_rate=1e-3, max_steps=20, warmup_steps=5, eval_interval=10, checkpoint_interval=10, grad_accum_steps=1),
            data=DataConfig(tokenized_dir=str(data_dir)),
            metrics=MetricsConfig(wandb_enabled=False),
        )
        ckpt_dir = str(tmp_path / "checkpoints")
        trainer = Trainer(cfg, checkpoint_dir=ckpt_dir)
        history = trainer.train()
        assert len(history) == 20

        # 3. Load checkpoint and evaluate
        model = GPT(cfg.model)
        load_checkpoint(f"{ckpt_dir}/best.pt", model)
        model.eval()

        from nash_llm.data.dataset import PretrainDataset
        val_ds = PretrainDataset(str(data_dir), split="val", seq_len=32)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4)
        val_loss = compute_val_loss(model, val_loader, max_batches=3)
        assert val_loss > 0

        # 4. Generate (use model.generate directly since vocab_size=500, not 50257)
        prompt = torch.randint(0, 500, (1, 5))
        generated = model.generate(prompt, max_new_tokens=10)
        assert generated.shape[1] == 15
