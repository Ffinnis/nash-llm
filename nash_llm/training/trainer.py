import os
import math
import time
from dataclasses import asdict
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
        self.tokens_per_micro_batch = config.train.batch_size * config.model.max_seq_len
        self.tokens_per_step = self.tokens_per_micro_batch * config.train.grad_accum_steps
        if config.train.max_tokens > 0:
            token_limited_steps = math.ceil(config.train.max_tokens / self.tokens_per_step)
            self.train_max_steps = min(config.train.max_steps, token_limited_steps)
        else:
            self.train_max_steps = config.train.max_steps

        self.model = GPT(config.model).to(self.device)

        self.optimizer = configure_optimizer(
            self.model, lr=config.train.learning_rate, weight_decay=config.train.weight_decay,
        )

        min_lr = config.train.learning_rate / 10
        scheduler_warmup_steps = min(config.train.warmup_steps, self.train_max_steps)
        self.scheduler = CosineScheduler(
            max_lr=config.train.learning_rate, min_lr=min_lr,
            warmup_steps=scheduler_warmup_steps, max_steps=self.train_max_steps,
        )

        self.train_dataset = PretrainDataset(config.data.tokenized_dir, split="train", seq_len=config.model.max_seq_len)
        self.val_dataset = PretrainDataset(config.data.tokenized_dir, split="val", seq_len=config.model.max_seq_len)
        loader_workers = max(int(config.data.num_workers), 0)
        self.pin_memory = self.device.type == "cuda"
        train_loader_kwargs: dict[str, Any] = {
            "batch_size": config.train.batch_size,
            "shuffle": True,
            "num_workers": loader_workers,
            "drop_last": True,
            "pin_memory": self.pin_memory,
        }
        val_loader_kwargs: dict[str, Any] = {
            "batch_size": config.train.batch_size,
            "shuffle": False,
            "num_workers": loader_workers,
            "pin_memory": self.pin_memory,
        }
        if loader_workers > 0:
            train_loader_kwargs["persistent_workers"] = True
            train_loader_kwargs["prefetch_factor"] = 2
            val_loader_kwargs["persistent_workers"] = True
            val_loader_kwargs["prefetch_factor"] = 2
        self.train_loader = DataLoader(self.train_dataset, **train_loader_kwargs)
        self.val_loader = DataLoader(self.val_dataset, **val_loader_kwargs)

        self.logger = MetricsLogger(config.metrics, run_config=asdict(config))

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

    def _build_progress_metrics(self, step: int, tokens_per_sec: float, total_tokens_processed: int) -> dict[str, Any]:
        total_steps = self.train_max_steps
        current_step = step + 1
        if self.config.train.max_tokens > 0:
            progress_ratio = min(total_tokens_processed / self.config.train.max_tokens, 1.0)
        else:
            progress_ratio = (current_step / total_steps) if total_steps > 0 else 1.0
        steps_remaining = max(total_steps - current_step, 0)
        metrics = {
            "step_current": current_step,
            "steps_total": total_steps,
            "steps_remaining": steps_remaining,
            "progress_pct": progress_ratio * 100.0,
            "tokens_per_sec": tokens_per_sec,
            "progress_bar": self._progress_bar(progress_ratio),
            "tokens_seen": total_tokens_processed,
        }
        if self.config.train.max_tokens > 0:
            metrics["tokens_target"] = self.config.train.max_tokens
        return metrics

    @staticmethod
    def _format_progress_log(metrics: dict[str, Any]) -> str:
        tokens_block = ""
        if "tokens_target" in metrics:
            tokens_block = f" | tok {metrics['tokens_seen']}/{metrics['tokens_target']}"
        return (
            f"{metrics['progress_bar']} {metrics['progress_pct']:.2f}% | "
            f"step {metrics['step_current']}/{metrics['steps_total']} "
            f"(left {metrics['steps_remaining']}) | "
            f"loss {metrics['train_loss']:.4f} | "
            f"lr {metrics['lr']:.2e} | "
            f"tok/s {metrics['tokens_per_sec']:.0f}"
            f"{tokens_block}"
        )

    def train(self) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        cfg = self.config.train
        self.model.train()
        train_iter = iter(self.train_loader)
        total_tokens_processed = self.start_step * self.tokens_per_step

        for step in range(self.start_step, self.train_max_steps):
            if cfg.max_tokens > 0 and total_tokens_processed >= cfg.max_tokens:
                break
            step_started_at = time.perf_counter()
            lr = self._set_lr(step)

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = torch.zeros((), device=self.device)
            tokens_processed = 0
            remaining_token_budget = cfg.max_tokens - total_tokens_processed if cfg.max_tokens > 0 else 0
            micro_steps_target = cfg.grad_accum_steps
            if cfg.max_tokens > 0:
                micro_steps_target = min(
                    cfg.grad_accum_steps,
                    max(1, math.ceil(remaining_token_budget / self.tokens_per_micro_batch)),
                )

            for micro_step in range(micro_steps_target):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x = x.to(self.device, non_blocking=self.pin_memory)
                y = y.to(self.device, non_blocking=self.pin_memory)
                tokens_processed += int(x.numel())

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _, loss = self.model(x, y)
                    loss = loss / micro_steps_target

                self.scaler.scale(loss).backward()
                accum_loss += loss.detach()

            if cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_tokens_processed += tokens_processed
            elapsed = max(time.perf_counter() - step_started_at, 1e-8)
            tokens_per_sec = tokens_processed / elapsed
            progress_metrics = self._build_progress_metrics(step, tokens_per_sec, total_tokens_processed)
            train_loss = float(accum_loss.item())

            record = {"step": step, "train_loss": train_loss, "lr": lr, **progress_metrics}

            if step % self.config.metrics.log_interval == 0:
                train_metrics = {"train_loss": train_loss, "lr": lr, **progress_metrics}
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
            if cfg.max_tokens > 0 and total_tokens_processed >= cfg.max_tokens:
                break

        self.logger.finish()
        return history
