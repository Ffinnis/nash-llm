import pytest
import yaml
from nash_llm.config import ModelConfig, TrainConfig, DataConfig, MetricsConfig, NashConfig, load_config


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.n_layers == 12
        assert cfg.n_heads == 12
        assert cfg.n_kv_heads == 12
        assert cfg.d_model == 768
        assert cfg.d_ff == 3072
        assert cfg.vocab_size == 50257
        assert cfg.max_seq_len == 1024
        assert cfg.dropout == 0.1
        assert cfg.activation == "swiglu"
        assert cfg.position_embedding == "rope"
        assert cfg.rope_base == 10_000.0

    def test_custom_values(self):
        cfg = ModelConfig(n_layers=6, d_model=512, activation="gelu", position_embedding="learned")
        assert cfg.n_layers == 6
        assert cfg.d_model == 512
        assert cfg.n_heads == 12
        assert cfg.n_kv_heads == 12
        assert cfg.activation == "gelu"
        assert cfg.position_embedding == "learned"

    def test_n_kv_heads_defaults_to_n_heads(self):
        cfg = ModelConfig(n_heads=8)
        assert cfg.n_kv_heads == 8

    def test_invalid_n_kv_heads_rejected_on_init(self):
        with pytest.raises(ValueError, match="must divide model.n_heads"):
            ModelConfig(n_heads=8, n_kv_heads=3)

    def test_invalid_activation_rejected_on_init(self):
        with pytest.raises(ValueError, match="Unsupported model.activation"):
            ModelConfig(activation="reglu")

    def test_invalid_position_embedding_rejected_on_init(self):
        with pytest.raises(ValueError, match="Unsupported model.position_embedding"):
            ModelConfig(position_embedding="sinusoidal")


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
        assert cfg.precision == "bf16"
        assert cfg.muon_lr == 0.02
        assert cfg.muon_momentum == 0.95
        assert cfg.ns_steps == 5
        assert cfg.sage_beta1 == 0.9
        assert cfg.sage_beta2 == 0.99
        assert cfg.sage_eps == 1e-8
        assert cfg.sage_fused is True

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
        assert cfg.eval_max_batches == 20
        assert cfg.final_eval_max_batches == 0
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
            "model": {
                "n_layers": 6,
                "d_model": 512,
                "activation": "gelu",
                "position_embedding": "learned",
            },
            "train": {"learning_rate": 1e-4, "precision": "fp16"},
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        cfg = load_config(str(yaml_path))
        assert cfg.model.n_layers == 6
        assert cfg.model.d_model == 512
        assert cfg.model.activation == "gelu"
        assert cfg.model.position_embedding == "learned"
        assert cfg.model.n_kv_heads == 12
        assert cfg.train.learning_rate == 1e-4
        assert cfg.train.precision == "fp16"
        assert cfg.model.n_heads == 12
        assert cfg.train.batch_size == 64

    def test_cli_overrides(self, tmp_path):
        yaml_content = {"model": {"n_layers": 6}}
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        overrides = {"model.n_layers": "12", "train.learning_rate": "1e-5", "train.precision": "fp16"}
        cfg = load_config(str(yaml_path), overrides=overrides)
        assert cfg.model.n_layers == 12
        assert cfg.train.learning_rate == 1e-5
        assert cfg.train.precision == "fp16"

    def test_cli_overrides_without_yaml(self):
        overrides = {
            "model.n_layers": "6",
            "train.batch_size": "32",
            "precision": "fp16",
            "activation": "gelu",
            "position_embedding": "learned",
        }
        cfg = load_config(config_path=None, overrides=overrides)
        assert cfg.model.n_layers == 6
        assert cfg.train.batch_size == 32
        assert cfg.train.precision == "fp16"
        assert cfg.model.activation == "gelu"
        assert cfg.model.position_embedding == "learned"

    def test_n_heads_override_keeps_default_n_kv_heads_in_sync(self):
        cfg = load_config(config_path=None, overrides={"n_heads": "8"})
        assert cfg.model.n_heads == 8
        assert cfg.model.n_kv_heads == 8

    def test_explicit_n_kv_heads_override_wins(self):
        cfg = load_config(config_path=None, overrides={"n_heads": "8", "n_kv_heads": "2"})
        assert cfg.model.n_heads == 8
        assert cfg.model.n_kv_heads == 2

    def test_invalid_activation_override_raises(self):
        with pytest.raises(ValueError, match="Unsupported model.activation"):
            load_config(config_path=None, overrides={"activation": "reglu"})

    def test_invalid_activation_yaml_raises(self, tmp_path):
        yaml_content = {"model": {"activation": "reglu"}}
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))
        with pytest.raises(ValueError, match="Unsupported model.activation"):
            load_config(str(yaml_path))

    def test_invalid_precision_override_raises(self):
        with pytest.raises(ValueError, match="Unsupported train.precision"):
            load_config(config_path=None, overrides={"precision": "fp32"})

    def test_invalid_precision_yaml_raises(self, tmp_path):
        yaml_content = {"train": {"precision": "fp32"}}
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))
        with pytest.raises(ValueError, match="Unsupported train.precision"):
            load_config(str(yaml_path))

    def test_invalid_position_embedding_override_raises(self):
        with pytest.raises(ValueError, match="Unsupported model.position_embedding"):
            load_config(config_path=None, overrides={"position_embedding": "sinusoidal"})

    def test_invalid_position_embedding_yaml_raises(self, tmp_path):
        yaml_content = {"model": {"position_embedding": "sinusoidal"}}
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))
        with pytest.raises(ValueError, match="Unsupported model.position_embedding"):
            load_config(str(yaml_path))

    def test_muon_lr_yaml(self, tmp_path):
        yaml_content = {"train": {"muon_lr": 0.03}}
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))
        cfg = load_config(str(yaml_path))
        assert cfg.train.muon_lr == 0.03

    def test_sage_hparams_yaml(self, tmp_path):
        yaml_content = {
            "train": {
                "sage_beta1": 0.85,
                "sage_beta2": 0.995,
                "sage_eps": 1e-7,
                "sage_fused": False,
            }
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))
        cfg = load_config(str(yaml_path))
        assert cfg.train.sage_beta1 == 0.85
        assert cfg.train.sage_beta2 == 0.995
        assert cfg.train.sage_eps == 1e-7
        assert cfg.train.sage_fused is False
