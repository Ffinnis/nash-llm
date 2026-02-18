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
    moe_enabled: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_expert_d_ff: int = 1536
    moe_capacity_factor: float = 1.25
    moe_router_jitter: float = 0.01
    moe_start_layer: int = 6
    moe_layer_stride: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 100_000
    max_tokens: int = 0
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    eval_interval: int = 500
    checkpoint_interval: int = 5000
    grad_accum_steps: int = 1
    compile: bool = False
    precision: str = "bf16"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    ns_steps: int = 5
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 1e-4


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


def _resolve_short_key(cfg: NashConfig, key: str) -> tuple[str, str]:
    """Resolve a short key like 'max_steps' to 'train.max_steps' by searching all sections."""
    sections = {"model": cfg.model, "train": cfg.train, "data": cfg.data, "metrics": cfg.metrics}
    matches = [(name, key) for name, section in sections.items() if hasattr(section, key)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = [f"{m[0]}.{key}" for m in matches]
        raise ValueError(f"Ambiguous key '{key}', found in: {', '.join(options)}")
    raise ValueError(f"Unknown config field: {key}")


def _validate_train_precision(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Unsupported train.precision '{value}'. Expected one of: bf16, fp16")
    precision = value.lower()
    if precision not in {"fp16", "bf16"}:
        raise ValueError(f"Unsupported train.precision '{value}'. Expected one of: bf16, fp16")
    return precision


def _validate_model_moe_config(cfg: ModelConfig) -> None:
    if not cfg.moe_enabled:
        return
    if cfg.moe_top_k < 1 or cfg.moe_top_k > cfg.moe_num_experts:
        raise ValueError(
            f"Invalid model.moe_top_k={cfg.moe_top_k}. "
            f"Expected 1 <= moe_top_k <= moe_num_experts ({cfg.moe_num_experts})."
        )
    if cfg.moe_expert_d_ff <= 0:
        raise ValueError(f"Invalid model.moe_expert_d_ff={cfg.moe_expert_d_ff}. Expected > 0.")
    if cfg.moe_layer_stride < 1:
        raise ValueError(f"Invalid model.moe_layer_stride={cfg.moe_layer_stride}. Expected >= 1.")
    if cfg.moe_start_layer < 0 or cfg.moe_start_layer >= cfg.n_layers:
        raise ValueError(
            f"Invalid model.moe_start_layer={cfg.moe_start_layer}. "
            f"Expected 0 <= moe_start_layer < n_layers ({cfg.n_layers})."
        )


def _apply_overrides(cfg: NashConfig, overrides: dict[str, str]) -> NashConfig:
    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) == 1:
            section_name, field_name = _resolve_short_key(cfg, key)
        elif len(parts) == 2:
            section_name, field_name = parts
        else:
            raise ValueError(f"Override key must be 'field' or 'section.field', got: {key}")
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
        if section_name == "train" and field_name == "precision":
            parsed = _validate_train_precision(str(parsed))
        setattr(section, field_name, parsed)
    return cfg


def load_config(config_path: str | None = None, overrides: dict[str, str] | None = None) -> NashConfig:
    cfg = NashConfig()
    if config_path is not None:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        if "model" in raw:
            cfg.model = ModelConfig(**{**vars(cfg.model), **raw["model"]})
            _validate_model_moe_config(cfg.model)
        if "train" in raw:
            cfg.train = TrainConfig(**{**vars(cfg.train), **raw["train"]})
            cfg.train.precision = _validate_train_precision(cfg.train.precision)
        if "data" in raw:
            cfg.data = DataConfig(**{**vars(cfg.data), **raw["data"]})
        if "metrics" in raw:
            cfg.metrics = MetricsConfig(**{**vars(cfg.metrics), **raw["metrics"]})
    if overrides:
        _apply_overrides(cfg, overrides)
    _validate_model_moe_config(cfg.model)
    return cfg
