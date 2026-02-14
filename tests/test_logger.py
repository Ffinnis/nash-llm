from unittest.mock import patch, MagicMock
from nash_llm.metrics.logger import MetricsLogger
from nash_llm.config import MetricsConfig


class TestMetricsLogger:
    def test_disabled_mode(self):
        cfg = MetricsConfig(wandb_enabled=False)
        logger = MetricsLogger(cfg)
        logger.log({"val_loss": 3.5}, step=100)

    def test_history_tracking(self):
        cfg = MetricsConfig(wandb_enabled=False)
        logger = MetricsLogger(cfg)
        logger.log({"val_loss": 3.5, "accuracy": 0.4}, step=100)
        logger.log({"val_loss": 3.2, "accuracy": 0.5}, step=200)
        assert len(logger.history) == 2
        assert logger.history[1]["val_loss"] == 3.2

    @patch("nash_llm.metrics.logger.wandb")
    def test_wandb_init_called(self, mock_wandb):
        cfg = MetricsConfig(wandb_enabled=True, wandb_project="test-proj")
        logger = MetricsLogger(cfg, run_name="test-run")
        mock_wandb.init.assert_called_once_with(project="test-proj", name="test-run", config=None)

    @patch("nash_llm.metrics.logger.wandb")
    def test_wandb_log_called(self, mock_wandb):
        cfg = MetricsConfig(wandb_enabled=True)
        logger = MetricsLogger(cfg)
        logger.log({"val_loss": 3.5}, step=100)
        mock_wandb.log.assert_called_once()
