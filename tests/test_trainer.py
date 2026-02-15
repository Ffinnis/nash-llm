import torch
import numpy as np
import json
import glob
from unittest.mock import patch
from nash_llm.training.trainer import Trainer
from nash_llm.config import NashConfig, ModelConfig, TrainConfig, DataConfig, MetricsConfig


def make_fake_data(tmp_path, n_tokens=2000, split="train"):
    tokens = np.random.randint(0, 100, size=n_tokens, dtype=np.uint16)
    shard_path = tmp_path / f"{split}_000.bin"
    tokens.tofile(str(shard_path))


class TestTrainer:
    def _make_config(self, tmp_path):
        make_fake_data(tmp_path, n_tokens=2000, split="train")
        make_fake_data(tmp_path, n_tokens=500, split="val")
        meta = {"vocab_size": 100, "total_tokens": 2000}
        (tmp_path / "meta.json").write_text(json.dumps(meta))

        return NashConfig(
            model=ModelConfig(n_layers=2, n_heads=4, d_model=64, d_ff=256, vocab_size=100, max_seq_len=32, dropout=0.0),
            train=TrainConfig(batch_size=4, learning_rate=3e-4, max_steps=10, warmup_steps=2, eval_interval=5, checkpoint_interval=5, grad_accum_steps=1),
            data=DataConfig(tokenized_dir=str(tmp_path)),
            metrics=MetricsConfig(wandb_enabled=False, log_interval=2),
        )

    def test_train_reduces_loss(self, tmp_path):
        cfg = self._make_config(tmp_path)
        ckpt_dir = str(tmp_path / "checkpoints")
        trainer = Trainer(cfg, checkpoint_dir=ckpt_dir)
        history = trainer.train()
        assert len(history) > 0
        assert all(not np.isnan(h["train_loss"]) for h in history)

    def test_checkpoint_saved(self, tmp_path):
        cfg = self._make_config(tmp_path)
        ckpt_dir = str(tmp_path / "checkpoints")
        trainer = Trainer(cfg, checkpoint_dir=ckpt_dir)
        trainer.train()
        ckpts = glob.glob(f"{ckpt_dir}/*.pt")
        assert len(ckpts) > 0

    def test_eval_runs(self, tmp_path):
        cfg = self._make_config(tmp_path)
        ckpt_dir = str(tmp_path / "checkpoints")
        trainer = Trainer(cfg, checkpoint_dir=ckpt_dir)
        history = trainer.train()
        eval_entries = [h for h in history if "val_loss" in h]
        assert len(eval_entries) > 0

    def test_progress_metrics_logged(self, tmp_path):
        cfg = self._make_config(tmp_path)
        ckpt_dir = str(tmp_path / "checkpoints")
        trainer = Trainer(cfg, checkpoint_dir=ckpt_dir)
        history = trainer.train()

        assert len(history) == cfg.train.max_steps
        assert all("tokens_per_sec" in h for h in history)
        assert all("step_current" in h for h in history)
        assert all("steps_remaining" in h for h in history)
        assert all("progress_pct" in h for h in history)
        assert all("progress_bar" in h for h in history)

        assert history[0]["step_current"] == 1
        assert history[-1]["steps_remaining"] == 0
        assert history[-1]["progress_pct"] == 100.0
        assert all(h["tokens_per_sec"] > 0 for h in history)

    def test_stops_by_max_tokens(self, tmp_path):
        cfg = self._make_config(tmp_path)
        cfg.train.max_steps = 100
        cfg.train.max_tokens = 500
        cfg.train.batch_size = 2
        cfg.train.grad_accum_steps = 1
        cfg.model.max_seq_len = 32
        ckpt_dir = str(tmp_path / "checkpoints")

        trainer = Trainer(cfg, checkpoint_dir=ckpt_dir)
        history = trainer.train()

        assert len(history) < cfg.train.max_steps
        assert history[-1]["tokens_seen"] >= cfg.train.max_tokens
        assert history[-1]["progress_pct"] == 100.0

    @patch("nash_llm.training.trainer.MetricsLogger")
    def test_passes_full_config_to_logger(self, mock_logger, tmp_path):
        cfg = self._make_config(tmp_path)
        Trainer(cfg, checkpoint_dir=str(tmp_path / "checkpoints"))

        assert mock_logger.call_count == 1
        _, kwargs = mock_logger.call_args
        assert "run_config" in kwargs
        assert kwargs["run_config"]["model"]["d_model"] == cfg.model.d_model
