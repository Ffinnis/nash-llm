from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    vocab_size: int = 50257
    max_seq_len: int = 1024
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 100_000
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    eval_interval: int = 500
    checkpoint_interval: int = 5000
    grad_accum_steps: int = 1


@dataclass
class DataConfig:
    dataset: str = "openwebtext"
    dataset_size: str = "100M"
    tokenized_dir: str = "datasets/tokenized"
    num_workers: int = 4


@dataclass
class MetricsConfig:
    wandb_project: str = "nash-llm"
    wandb_enabled: bool = True
    log_interval: int = 10
    metrics: list[str] = field(default_factory=lambda: ["val_loss", "accuracy"])


@dataclass
class NashConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


def _apply_overrides(cfg: NashConfig, overrides: dict[str, str]) -> NashConfig:
    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(f"Override key must be 'section.field', got: {key}")
        section_name, field_name = parts
        section = getattr(cfg, section_name, None)
        if section is None:
            raise ValueError(f"Unknown config section: {section_name}")
        if not hasattr(section, field_name):
            raise ValueError(f"Unknown field {field_name} in {section_name}")
        field_type = type(getattr(section, field_name))
        if field_type == bool:
            parsed = value.lower() in ("true", "1", "yes")
        elif field_type == list:
            parsed = value.split(",")
        else:
            parsed = field_type(value)
        setattr(section, field_name, parsed)
    return cfg


def load_config(config_path: str | None = None, overrides: dict[str, str] | None = None) -> NashConfig:
    cfg = NashConfig()
    if config_path is not None:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        if "model" in raw:
            cfg.model = ModelConfig(**{**vars(cfg.model), **raw["model"]})
        if "train" in raw:
            cfg.train = TrainConfig(**{**vars(cfg.train), **raw["train"]})
        if "data" in raw:
            cfg.data = DataConfig(**{**vars(cfg.data), **raw["data"]})
        if "metrics" in raw:
            cfg.metrics = MetricsConfig(**{**vars(cfg.metrics), **raw["metrics"]})
    if overrides:
        _apply_overrides(cfg, overrides)
    return cfg
