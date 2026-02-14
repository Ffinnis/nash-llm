from typing import Any
import wandb
from nash_llm.config import MetricsConfig


class MetricsLogger:
    def __init__(self, config: MetricsConfig, run_name: str | None = None, run_config: dict | None = None):
        self.config = config
        self.history: list[dict[str, Any]] = []

        if config.wandb_enabled:
            wandb.init(project=config.wandb_project, name=run_name, config=run_config)

    def log(self, metrics: dict[str, Any], step: int):
        record = {"step": step, **metrics}
        self.history.append(record)

        if self.config.wandb_enabled:
            wandb.log(metrics, step=step)

    def finish(self):
        if self.config.wandb_enabled:
            wandb.finish()
