import pytest
import yaml
from nash_llm.config import ModelConfig, TrainConfig, DataConfig, MetricsConfig, NashConfig, load_config


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.n_layers == 12
        assert cfg.n_heads == 12
        assert cfg.d_model == 768
        assert cfg.d_ff == 3072
        assert cfg.vocab_size == 50257
        assert cfg.max_seq_len == 1024
        assert cfg.dropout == 0.1

    def test_custom_values(self):
        cfg = ModelConfig(n_layers=6, d_model=512)
        assert cfg.n_layers == 6
        assert cfg.d_model == 512
        assert cfg.n_heads == 12


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.batch_size == 64
        assert cfg.learning_rate == 3e-4
        assert cfg.weight_decay == 0.1
        assert cfg.max_steps == 100_000
        assert cfg.max_tokens == 0
        assert cfg.warmup_steps == 2000
        assert cfg.grad_clip == 1.0
        assert cfg.eval_interval == 500
        assert cfg.checkpoint_interval == 5000
        assert cfg.grad_accum_steps == 1

    def test_custom(self):
        cfg = TrainConfig(learning_rate=1e-4, batch_size=32)
        assert cfg.learning_rate == 1e-4
        assert cfg.batch_size == 32


class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.dataset == "openwebtext"
        assert cfg.dataset_size == "100M"
        assert cfg.tokenized_dir == "datasets/tokenized"


class TestMetricsConfig:
    def test_defaults(self):
        cfg = MetricsConfig()
        assert cfg.wandb_project == "nash-llm"
        assert cfg.log_interval == 10
        assert "val_loss" in cfg.metrics
        assert "accuracy" in cfg.metrics


class TestNashConfig:
    def test_contains_all_sub_configs(self):
        cfg = NashConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.train, TrainConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.metrics, MetricsConfig)


class TestLoadConfig:
    def test_load_from_yaml(self, tmp_path):
        yaml_content = {
            "model": {"n_layers": 6, "d_model": 512},
            "train": {"learning_rate": 1e-4},
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        cfg = load_config(str(yaml_path))
        assert cfg.model.n_layers == 6
        assert cfg.model.d_model == 512
        assert cfg.train.learning_rate == 1e-4
        assert cfg.model.n_heads == 12
        assert cfg.train.batch_size == 64

    def test_cli_overrides(self, tmp_path):
        yaml_content = {"model": {"n_layers": 6}}
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        overrides = {"model.n_layers": "12", "train.learning_rate": "1e-5"}
        cfg = load_config(str(yaml_path), overrides=overrides)
        assert cfg.model.n_layers == 12
        assert cfg.train.learning_rate == 1e-5

    def test_cli_overrides_without_yaml(self):
        overrides = {"model.n_layers": "6", "train.batch_size": "32"}
        cfg = load_config(config_path=None, overrides=overrides)
        assert cfg.model.n_layers == 6
        assert cfg.train.batch_size == 32
