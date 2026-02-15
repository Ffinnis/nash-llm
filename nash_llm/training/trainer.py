import os
import math
import time
import torch
from torch.utils.data import DataLoader
from typing import Any

from nash_llm.model import GPT
from nash_llm.optim import configure_optimizer
from nash_llm.data.dataset import PretrainDataset
from nash_llm.training.scheduler import CosineScheduler
from nash_llm.training.checkpoint import save_checkpoint
from nash_llm.metrics.logger import MetricsLogger
from nash_llm.eval.evaluate import compute_val_loss, compute_accuracy
from nash_llm.config import NashConfig


class Trainer:
    def __init__(self, config: NashConfig, checkpoint_dir: str = "checkpoints", resume_from: str | None = None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GPT(config.model).to(self.device)

        self.optimizer = configure_optimizer(
            self.model, lr=config.train.learning_rate, weight_decay=config.train.weight_decay,
        )

        min_lr = config.train.learning_rate / 10
        self.scheduler = CosineScheduler(
            max_lr=config.train.learning_rate, min_lr=min_lr,
            warmup_steps=config.train.warmup_steps, max_steps=config.train.max_steps,
        )

        self.train_dataset = PretrainDataset(config.data.tokenized_dir, split="train", seq_len=config.model.max_seq_len)
        self.val_dataset = PretrainDataset(config.data.tokenized_dir, split="val", seq_len=config.model.max_seq_len)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=0)

        self.logger = MetricsLogger(config.metrics, run_config=None)

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.start_step = 0
        self.best_val_loss = float("inf")

        if resume_from:
            self._resume(resume_from)

    def _resume(self, path: str):
        from nash_llm.training.checkpoint import load_checkpoint
        ckpt = load_checkpoint(path, self.model, self.optimizer)
        self.start_step = ckpt["step"]
        if ckpt.get("metrics", {}).get("val_loss"):
            self.best_val_loss = ckpt["metrics"]["val_loss"]

    def _set_lr(self, step: int):
        lr = self.scheduler.get_lr(step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    @staticmethod
    def _progress_bar(progress: float, width: int = 30) -> str:
        progress = max(0.0, min(1.0, progress))
        filled = int(progress * width)
        return f"[{'#' * filled}{'-' * (width - filled)}]"

    def _build_progress_metrics(self, step: int, tokens_per_sec: float) -> dict[str, Any]:
        total_steps = self.config.train.max_steps
        current_step = step + 1
        progress_ratio = (current_step / total_steps) if total_steps > 0 else 1.0
        steps_remaining = max(total_steps - current_step, 0)
        return {
            "step_current": current_step,
            "steps_total": total_steps,
            "steps_remaining": steps_remaining,
            "progress_pct": progress_ratio * 100.0,
            "tokens_per_sec": tokens_per_sec,
            "progress_bar": self._progress_bar(progress_ratio),
        }

    @staticmethod
    def _format_progress_log(metrics: dict[str, Any]) -> str:
        return (
            f"{metrics['progress_bar']} {metrics['progress_pct']:.2f}% | "
            f"step {metrics['step_current']}/{metrics['steps_total']} "
            f"(left {metrics['steps_remaining']}) | "
            f"loss {metrics['train_loss']:.4f} | "
            f"lr {metrics['lr']:.2e} | "
            f"tok/s {metrics['tokens_per_sec']:.0f}"
        )

    def train(self) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        cfg = self.config.train
        self.model.train()
        train_iter = iter(self.train_loader)

        for step in range(self.start_step, cfg.max_steps):
            step_started_at = time.perf_counter()
            lr = self._set_lr(step)

            self.optimizer.zero_grad()
            accum_loss = 0.0
            tokens_processed = 0

            for micro_step in range(cfg.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x, y = x.to(self.device), y.to(self.device)
                tokens_processed += int(x.numel())

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _, loss = self.model(x, y)
                    loss = loss / cfg.grad_accum_steps

                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            if cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            elapsed = max(time.perf_counter() - step_started_at, 1e-8)
            tokens_per_sec = tokens_processed / elapsed
            progress_metrics = self._build_progress_metrics(step, tokens_per_sec)

            record = {"step": step, "train_loss": accum_loss, "lr": lr, **progress_metrics}

            if step % self.config.metrics.log_interval == 0:
                train_metrics = {"train_loss": accum_loss, "lr": lr, **progress_metrics}
                self.logger.log(train_metrics, step=step)
                print(self._format_progress_log(train_metrics))

            if step > 0 and step % cfg.eval_interval == 0:
                self.model.eval()
                val_loss = compute_val_loss(self.model, self.val_loader, max_batches=20)
                accuracy = compute_accuracy(self.model, self.val_loader, max_batches=20)
                perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")

                eval_metrics = {"val_loss": val_loss, "accuracy": accuracy, "perplexity": perplexity}
                record.update(eval_metrics)
                self.logger.log(eval_metrics, step=step)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_checkpoint(
                        os.path.join(self.checkpoint_dir, "best.pt"),
                        self.model, self.optimizer, step=step, config=self.config, metrics=eval_metrics,
                    )

                self.model.train()

            if step > 0 and step % cfg.checkpoint_interval == 0:
                save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"step_{step}.pt"),
                    self.model, self.optimizer, step=step, config=self.config,
                )

            history.append(record)

        self.logger.finish()
        return history
