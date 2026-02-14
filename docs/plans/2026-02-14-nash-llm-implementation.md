# Nash-LLM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modular LLM research framework for pretraining and fine-tuning a 124M parameter GPT-2 style transformer.

**Architecture:** Modular Python package `nash_llm/` with sub-packages (model, optim, data, training, config, metrics, eval). Scripts in `scripts/`. YAML presets in `configs/`. TDD throughout — every component gets tested before integration.

**Tech Stack:** PyTorch >= 2.0, tiktoken, wandb, numpy, matplotlib, datasets (HF), pyyaml

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: all `__init__.py` files for package structure

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "nash-llm"
version = "0.1.0"
description = "Research LLM framework for pretraining and fine-tuning transformers"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "tiktoken",
    "wandb",
    "numpy",
    "matplotlib",
    "datasets",
    "pyyaml",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
datasets/
checkpoints/
*.bin
wandb/
plots/
.DS_Store
```

**Step 3: Create package structure**

Create all `__init__.py` files (empty):
- `nash_llm/__init__.py`
- `nash_llm/model/__init__.py`
- `nash_llm/optim/__init__.py`
- `nash_llm/data/__init__.py`
- `nash_llm/training/__init__.py`
- `nash_llm/config/__init__.py`
- `nash_llm/metrics/__init__.py`
- `nash_llm/eval/__init__.py`
- `tests/__init__.py`

Create empty directories:
- `scripts/`
- `configs/`

**Step 4: Install in dev mode and verify**

Run: `pip install -e ".[dev]"`
Expected: Installs successfully

Run: `python -c "import nash_llm; print('ok')"`
Expected: `ok`

**Step 5: Commit**

```bash
git add pyproject.toml .gitignore nash_llm/ tests/ scripts/ configs/
git commit -m "feat: project scaffolding with package structure"
```

---

### Task 2: Configuration System

**Files:**
- Create: `nash_llm/config/config.py`
- Modify: `nash_llm/config/__init__.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing tests**

```python
# tests/test_config.py
import pytest
import tempfile
import os
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
        assert cfg.n_heads == 12  # unchanged default


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.batch_size == 64
        assert cfg.learning_rate == 3e-4
        assert cfg.weight_decay == 0.1
        assert cfg.max_steps == 100_000
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
        # defaults preserved
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — cannot import from `nash_llm.config`

**Step 3: Implement config.py**

```python
# nash_llm/config/config.py
from dataclasses import dataclass, field, fields
from typing import Any
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
    """Apply dot-notation overrides like {'model.n_layers': '6'} to config."""
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
        # Cast to the field's type
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
    """Load config from YAML file, then apply CLI overrides."""
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
```

**Step 4: Update __init__.py**

```python
# nash_llm/config/__init__.py
from nash_llm.config.config import (
    ModelConfig,
    TrainConfig,
    DataConfig,
    MetricsConfig,
    NashConfig,
    load_config,
)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add nash_llm/config/ tests/test_config.py
git commit -m "feat: configuration system with YAML loading and CLI overrides"
```

---

### Task 3: Tokenizer Wrapper

**Files:**
- Create: `nash_llm/data/tokenizer.py`
- Modify: `nash_llm/data/__init__.py`
- Create: `tests/test_tokenizer.py`

**Step 1: Write the failing tests**

```python
# tests/test_tokenizer.py
from nash_llm.data.tokenizer import Tokenizer


class TestTokenizer:
    def setup_method(self):
        self.tok = Tokenizer()

    def test_encode_returns_list_of_ints(self):
        ids = self.tok.encode("Hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_decode_roundtrip(self):
        text = "The quick brown fox jumps over the lazy dog."
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_vocab_size(self):
        assert self.tok.vocab_size == 50257

    def test_eot_token(self):
        assert isinstance(self.tok.eot_token, int)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tokenizer.py -v`
Expected: FAIL — cannot import `Tokenizer`

**Step 3: Implement tokenizer.py**

```python
# nash_llm/data/tokenizer.py
import tiktoken


class Tokenizer:
    def __init__(self):
        self._enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot_token(self) -> int:
        return self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)
```

**Step 4: Update __init__.py**

```python
# nash_llm/data/__init__.py
from nash_llm.data.tokenizer import Tokenizer
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tokenizer.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add nash_llm/data/tokenizer.py nash_llm/data/__init__.py tests/test_tokenizer.py
git commit -m "feat: tiktoken GPT-2 tokenizer wrapper"
```

---

### Task 4: Model — Layers (FeedForward, LayerNorm helpers)

**Files:**
- Create: `nash_llm/model/layers.py`
- Create: `tests/test_layers.py`

**Step 1: Write the failing tests**

```python
# tests/test_layers.py
import torch
import pytest
from nash_llm.model.layers import FeedForward
from nash_llm.config import ModelConfig


class TestFeedForward:
    def setup_method(self):
        self.cfg = ModelConfig(d_model=64, d_ff=256, dropout=0.0)
        self.ff = FeedForward(self.cfg)

    def test_output_shape(self):
        x = torch.randn(2, 10, 64)  # batch=2, seq=10, d_model=64
        out = self.ff(x)
        assert out.shape == (2, 10, 64)

    def test_parameter_count(self):
        # d_model->d_ff weight+bias + d_ff->d_model weight+bias
        params = sum(p.numel() for p in self.ff.parameters())
        # 64*256 + 256 + 256*64 + 64 = 16384 + 256 + 16384 + 64 = 33088
        assert params == 33088
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layers.py -v`
Expected: FAIL

**Step 3: Implement layers.py**

```python
# nash_llm/model/layers.py
import torch
import torch.nn as nn
from nash_llm.config import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_layers.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add nash_llm/model/layers.py tests/test_layers.py
git commit -m "feat: FeedForward layer with GELU activation"
```

---

### Task 5: Model — Multi-Head Attention

**Files:**
- Create: `nash_llm/model/attention.py`
- Create: `tests/test_attention.py`

**Step 1: Write the failing tests**

```python
# tests/test_attention.py
import torch
from nash_llm.model.attention import MultiHeadAttention
from nash_llm.config import ModelConfig


class TestMultiHeadAttention:
    def setup_method(self):
        self.cfg = ModelConfig(d_model=64, n_heads=4, max_seq_len=32, dropout=0.0)
        self.attn = MultiHeadAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(2, 16, 64)  # batch=2, seq=16, d_model=64
        out = self.attn(x)
        assert out.shape == (2, 16, 64)

    def test_causal_mask(self):
        """Future tokens should not attend to past tokens."""
        torch.manual_seed(42)
        self.attn.eval()
        x = torch.randn(1, 8, 64)
        out_full = self.attn(x)

        # Changing token 7 should NOT affect output at position 3
        x_modified = x.clone()
        x_modified[0, 7, :] = torch.randn(64)
        out_modified = self.attn(x_modified)

        # Positions 0-6 should be identical
        assert torch.allclose(out_full[0, :7], out_modified[0, :7], atol=1e-6)

    def test_head_dim_consistency(self):
        # d_model must be divisible by n_heads
        bad_cfg = ModelConfig(d_model=65, n_heads=4)
        with pytest.raises(ValueError):
            MultiHeadAttention(bad_cfg)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_attention.py -v`
Expected: FAIL

**Step 3: Implement attention.py**

```python
# nash_llm/model/attention.py
import math
import torch
import torch.nn as nn
from nash_llm.config import ModelConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})")

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask — register as buffer (not a parameter)
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V in one matmul
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_attention.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add nash_llm/model/attention.py tests/test_attention.py
git commit -m "feat: multi-head causal attention"
```

---

### Task 6: Model — Transformer (full GPT model)

**Files:**
- Create: `nash_llm/model/transformer.py`
- Modify: `nash_llm/model/__init__.py`
- Create: `tests/test_transformer.py`

**Step 1: Write the failing tests**

```python
# tests/test_transformer.py
import torch
from nash_llm.model import GPT
from nash_llm.config import ModelConfig


class TestGPT:
    def setup_method(self):
        self.cfg = ModelConfig(
            n_layers=2, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
        )
        self.model = GPT(self.cfg)

    def test_logits_shape(self):
        x = torch.randint(0, 100, (2, 16))  # batch=2, seq=16
        logits = self.model(x)
        assert logits.shape == (2, 16, 100)  # (B, T, vocab_size)

    def test_loss_computation(self):
        x = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        logits, loss = self.model(x, targets)
        assert logits.shape == (2, 16, 100)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_weight_tying(self):
        """Embedding and lm_head should share weights."""
        assert self.model.token_emb.weight is self.model.lm_head.weight

    def test_parameter_count_small(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total > 0
        # Rough check: tiny model should be well under 1M params
        assert total < 1_000_000

    def test_generate_basic(self):
        self.model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        generated = self.model.generate(prompt, max_new_tokens=10)
        assert generated.shape == (1, 15)  # 5 prompt + 10 generated
        assert (generated[:, :5] == prompt).all()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_transformer.py -v`
Expected: FAIL

**Step 3: Implement transformer.py**

```python
# nash_llm/model/transformer.py
import torch
import torch.nn as nn
from nash_llm.config import ModelConfig
from nash_llm.model.attention import MultiHeadAttention
from nash_llm.model.layers import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        tok_emb = self.token_emb(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=-1)
        return idx
```

**Step 4: Update __init__.py**

```python
# nash_llm/model/__init__.py
from nash_llm.model.transformer import GPT, TransformerBlock
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_transformer.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add nash_llm/model/ tests/test_transformer.py
git commit -m "feat: GPT transformer model with weight tying and generation"
```

---

### Task 7: Optimizer — AdamW with Weight Decay Splitting

**Files:**
- Create: `nash_llm/optim/adamw.py`
- Modify: `nash_llm/optim/__init__.py`
- Create: `tests/test_optim.py`

**Step 1: Write the failing tests**

```python
# tests/test_optim.py
import torch
from nash_llm.model import GPT
from nash_llm.config import ModelConfig
from nash_llm.optim import configure_optimizer


class TestConfigureOptimizer:
    def setup_method(self):
        self.cfg = ModelConfig(
            n_layers=2, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
        )
        self.model = GPT(self.cfg)

    def test_returns_adamw(self):
        opt = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1)
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups(self):
        opt = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1)
        assert len(opt.param_groups) == 2
        wds = {pg["weight_decay"] for pg in opt.param_groups}
        assert 0.0 in wds
        assert 0.1 in wds

    def test_no_decay_for_biases_and_layernorm(self):
        opt = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1)
        # Group with wd=0 should contain biases and layernorm weights
        no_decay_group = [pg for pg in opt.param_groups if pg["weight_decay"] == 0.0][0]
        no_decay_count = sum(p.numel() for p in no_decay_group["params"])
        assert no_decay_count > 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optim.py -v`
Expected: FAIL

**Step 3: Implement adamw.py**

```python
# nash_llm/optim/adamw.py
import torch
import torch.nn as nn


def configure_optimizer(model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW:
    """Create AdamW optimizer with proper weight decay splitting.

    Weight decay is applied only to 2D parameters (weight matrices),
    not to biases, LayerNorm weights, or embeddings.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
```

**Step 4: Update __init__.py**

```python
# nash_llm/optim/__init__.py
from nash_llm.optim.adamw import configure_optimizer
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_optim.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add nash_llm/optim/ tests/test_optim.py
git commit -m "feat: AdamW optimizer with weight decay splitting"
```

---

### Task 8: Training — LR Scheduler

**Files:**
- Create: `nash_llm/training/scheduler.py`
- Create: `tests/test_scheduler.py`

**Step 1: Write the failing tests**

```python
# tests/test_scheduler.py
import torch
from nash_llm.training.scheduler import CosineScheduler


class TestCosineScheduler:
    def test_warmup_phase(self):
        """LR should increase linearly during warmup."""
        scheduler = CosineScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=100, max_steps=1000)
        lr_0 = scheduler.get_lr(0)
        lr_50 = scheduler.get_lr(50)
        lr_100 = scheduler.get_lr(100)
        assert lr_0 < lr_50 < lr_100
        assert abs(lr_100 - 1e-3) < 1e-8

    def test_cosine_decay(self):
        """LR should decrease after warmup."""
        scheduler = CosineScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=100, max_steps=1000)
        lr_100 = scheduler.get_lr(100)
        lr_500 = scheduler.get_lr(500)
        lr_999 = scheduler.get_lr(999)
        assert lr_100 > lr_500 > lr_999

    def test_min_lr_at_end(self):
        scheduler = CosineScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=100, max_steps=1000)
        lr_end = scheduler.get_lr(1000)
        assert abs(lr_end - 1e-5) < 1e-8

    def test_beyond_max_steps(self):
        """LR should stay at min_lr after max_steps."""
        scheduler = CosineScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=100, max_steps=1000)
        lr = scheduler.get_lr(2000)
        assert abs(lr - 1e-5) < 1e-8
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scheduler.py -v`
Expected: FAIL

**Step 3: Implement scheduler.py**

```python
# nash_llm/training/scheduler.py
import math


class CosineScheduler:
    def __init__(self, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        # Warmup phase: linear increase
        if step < self.warmup_steps:
            return self.max_lr * (step / self.warmup_steps)

        # Beyond max_steps: stay at min_lr
        if step >= self.max_steps:
            return self.min_lr

        # Cosine decay
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scheduler.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add nash_llm/training/scheduler.py tests/test_scheduler.py
git commit -m "feat: cosine LR scheduler with linear warmup"
```

---

### Task 9: Training — Checkpoint Save/Load

**Files:**
- Create: `nash_llm/training/checkpoint.py`
- Create: `tests/test_checkpoint.py`

**Step 1: Write the failing tests**

```python
# tests/test_checkpoint.py
import torch
import os
from nash_llm.model import GPT
from nash_llm.config import ModelConfig, NashConfig
from nash_llm.optim import configure_optimizer
from nash_llm.training.checkpoint import save_checkpoint, load_checkpoint


class TestCheckpoint:
    def setup_method(self):
        self.cfg = ModelConfig(
            n_layers=2, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
        )
        self.model = GPT(self.cfg)
        self.optimizer = configure_optimizer(self.model, lr=3e-4, weight_decay=0.1)

    def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, self.model, self.optimizer, step=100, config=NashConfig(model=self.cfg))
        assert os.path.exists(path)

    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        # Do a forward pass to change optimizer state
        x = torch.randint(0, 100, (1, 8))
        targets = torch.randint(0, 100, (1, 8))
        _, loss = self.model(x, targets)
        loss.backward()
        self.optimizer.step()

        save_checkpoint(path, self.model, self.optimizer, step=42, config=NashConfig(model=self.cfg), metrics={"val_loss": 3.5})

        # Load into fresh model
        model2 = GPT(self.cfg)
        optimizer2 = configure_optimizer(model2, lr=3e-4, weight_decay=0.1)
        ckpt = load_checkpoint(path, model2, optimizer2)

        assert ckpt["step"] == 42
        assert ckpt["metrics"]["val_loss"] == 3.5

        # Verify model weights match
        for p1, p2 in zip(self.model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_checkpoint.py -v`
Expected: FAIL

**Step 3: Implement checkpoint.py**

```python
# nash_llm/training/checkpoint.py
import os
import torch
import torch.nn as nn
from dataclasses import asdict
from typing import Any


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Any,
    metrics: dict | None = None,
    wandb_run_id: str | None = None,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "metrics": metrics or {},
        "wandb_run_id": wandb_run_id,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_checkpoint.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add nash_llm/training/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: checkpoint save/load with full state"
```

---

### Task 10: Metrics — WandB Logger

**Files:**
- Create: `nash_llm/metrics/logger.py`
- Modify: `nash_llm/metrics/__init__.py`
- Create: `tests/test_logger.py`

**Step 1: Write the failing tests**

```python
# tests/test_logger.py
from unittest.mock import patch, MagicMock
from nash_llm.metrics.logger import MetricsLogger
from nash_llm.config import MetricsConfig


class TestMetricsLogger:
    def test_disabled_mode(self):
        """When wandb disabled, log() should not crash."""
        cfg = MetricsConfig(wandb_enabled=False)
        logger = MetricsLogger(cfg)
        # Should not raise
        logger.log({"val_loss": 3.5}, step=100)

    def test_log_filters_configured_metrics(self):
        """Only metrics listed in config should be logged."""
        cfg = MetricsConfig(wandb_enabled=False, metrics=["val_loss"])
        logger = MetricsLogger(cfg)
        logger.log({"val_loss": 3.5, "some_other": 99}, step=1)
        assert "val_loss" in logger.history[0]
        # train_loss and other step-level metrics should pass through
        # only eval metrics are filtered

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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_logger.py -v`
Expected: FAIL

**Step 3: Implement logger.py**

```python
# nash_llm/metrics/logger.py
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
```

**Step 4: Update __init__.py**

```python
# nash_llm/metrics/__init__.py
from nash_llm.metrics.logger import MetricsLogger
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_logger.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add nash_llm/metrics/ tests/test_logger.py
git commit -m "feat: wandb metrics logger with local history"
```

---

### Task 11: Data — Pretrain Dataset

**Files:**
- Create: `nash_llm/data/dataset.py`
- Modify: `nash_llm/data/__init__.py`
- Create: `tests/test_dataset.py`

**Step 1: Write the failing tests**

```python
# tests/test_dataset.py
import numpy as np
import json
import torch
from nash_llm.data.dataset import PretrainDataset


class TestPretrainDataset:
    def setup_method(self, tmp_path=None):
        pass

    def _make_dataset(self, tmp_path, n_tokens=1000, seq_len=64):
        """Create a fake tokenized shard for testing."""
        tokens = np.random.randint(0, 50257, size=n_tokens, dtype=np.uint16)
        shard_path = tmp_path / "train_000.bin"
        tokens.tofile(str(shard_path))

        meta = {"vocab_size": 50257, "total_tokens": n_tokens}
        (tmp_path / "meta.json").write_text(json.dumps(meta))

        return PretrainDataset(str(tmp_path), split="train", seq_len=seq_len)

    def test_len(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_tokens=1000, seq_len=64)
        assert len(ds) == 1000 // 64

    def test_getitem_shapes(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_tokens=1000, seq_len=64)
        x, y = ds[0]
        assert x.shape == (64,)
        assert y.shape == (64,)
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64

    def test_targets_shifted(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_tokens=1000, seq_len=64)
        x, y = ds[0]
        # y should be x shifted by 1
        # We can check the underlying data
        tokens = np.fromfile(str(tmp_path / "train_000.bin"), dtype=np.uint16)
        assert x[0].item() == tokens[0]
        assert y[0].item() == tokens[1]

    def test_dataloader_compatible(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_tokens=1000, seq_len=64)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, 64)
        assert batch_y.shape == (4, 64)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL

**Step 3: Implement dataset.py**

```python
# nash_llm/data/dataset.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class PretrainDataset(Dataset):
    def __init__(self, data_dir: str, split: str, seq_len: int):
        self.seq_len = seq_len
        data_path = Path(data_dir)

        # Find all shards for this split
        shard_files = sorted(data_path.glob(f"{split}_*.bin"))
        if not shard_files:
            raise FileNotFoundError(f"No {split} shards found in {data_dir}")

        # Memory-map all shards and concatenate
        arrays = []
        for shard in shard_files:
            arr = np.memmap(str(shard), dtype=np.uint16, mode="r")
            arrays.append(arr)

        self.data = np.concatenate(arrays)
        self.n_tokens = len(self.data)

    def __len__(self) -> int:
        return self.n_tokens // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = idx * self.seq_len
        x = self.data[offset : offset + self.seq_len].astype(np.int64)
        y = self.data[offset + 1 : offset + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
```

**Step 4: Update __init__.py**

```python
# nash_llm/data/__init__.py
from nash_llm.data.tokenizer import Tokenizer
from nash_llm.data.dataset import PretrainDataset
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add nash_llm/data/ tests/test_dataset.py
git commit -m "feat: memory-mapped pretrain dataset"
```

---

### Task 12: Data — SFT Dataset

**Files:**
- Create: `nash_llm/data/sft_dataset.py`
- Modify: `nash_llm/data/__init__.py`
- Create: `tests/test_sft_dataset.py`

**Step 1: Write the failing tests**

```python
# tests/test_sft_dataset.py
import json
import torch
from nash_llm.data.sft_dataset import SFTDataset


class TestSFTDataset:
    def _make_dataset(self, tmp_path, max_seq_len=128):
        data = [
            {"prompt": "What is 2+2?", "completion": "4"},
            {"prompt": "Hello", "completion": "Hi there, how can I help?"},
        ]
        path = tmp_path / "sft_data.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return SFTDataset(str(path), max_seq_len=max_seq_len)

    def test_len(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        assert len(ds) == 2

    def test_getitem_returns_three_tensors(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        input_ids, targets, loss_mask = ds[0]
        assert input_ids.dtype == torch.int64
        assert targets.dtype == torch.int64
        assert loss_mask.dtype == torch.bool

    def test_loss_mask_masks_prompt(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        input_ids, targets, loss_mask = ds[0]
        # First tokens (prompt) should be masked (False)
        # Completion tokens should be unmasked (True)
        assert not loss_mask[0].item()  # prompt start is masked
        assert loss_mask.any()  # at least some completion tokens unmasked

    def test_padding_to_max_seq_len(self, tmp_path):
        ds = self._make_dataset(tmp_path, max_seq_len=128)
        input_ids, targets, loss_mask = ds[0]
        assert input_ids.shape == (128,)
        assert targets.shape == (128,)
        assert loss_mask.shape == (128,)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sft_dataset.py -v`
Expected: FAIL

**Step 3: Implement sft_dataset.py**

```python
# nash_llm/data/sft_dataset.py
import json
import torch
from torch.utils.data import Dataset
from nash_llm.data.tokenizer import Tokenizer


class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer()
        self.samples = []

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        prompt_ids = self.tokenizer.encode(item["prompt"])
        completion_ids = self.tokenizer.encode(item["completion"])
        eot = [self.tokenizer.eot_token]

        # prompt + completion + eot
        full_ids = prompt_ids + completion_ids + eot
        full_ids = full_ids[: self.max_seq_len]

        prompt_len = min(len(prompt_ids), self.max_seq_len)
        seq_len = len(full_ids)

        # Build loss mask: False for prompt, True for completion
        loss_mask = [False] * prompt_len + [True] * (seq_len - prompt_len)

        # Pad to max_seq_len
        pad_len = self.max_seq_len - seq_len
        input_ids = full_ids + [0] * pad_len
        # Targets: shifted by 1
        targets = full_ids[1:] + [0] * (pad_len + 1)
        targets = targets[: self.max_seq_len]
        loss_mask = loss_mask + [False] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.int64),
            torch.tensor(targets, dtype=torch.int64),
            torch.tensor(loss_mask, dtype=torch.bool),
        )
```

**Step 4: Update __init__.py**

```python
# nash_llm/data/__init__.py
from nash_llm.data.tokenizer import Tokenizer
from nash_llm.data.dataset import PretrainDataset
from nash_llm.data.sft_dataset import SFTDataset
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_sft_dataset.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add nash_llm/data/ tests/test_sft_dataset.py
git commit -m "feat: SFT dataset with prompt masking"
```

---

### Task 13: Eval — Evaluate and Generate

**Files:**
- Create: `nash_llm/eval/evaluate.py`
- Create: `nash_llm/eval/generate.py`
- Modify: `nash_llm/eval/__init__.py`
- Create: `tests/test_eval.py`

**Step 1: Write the failing tests**

```python
# tests/test_eval.py
import torch
import numpy as np
import json
from nash_llm.model import GPT
from nash_llm.config import ModelConfig
from nash_llm.data.dataset import PretrainDataset
from nash_llm.eval.evaluate import compute_val_loss, compute_accuracy
from nash_llm.eval.generate import generate_text


class TestEvaluate:
    def setup_method(self):
        self.cfg = ModelConfig(
            n_layers=2, n_heads=4, d_model=64, d_ff=256,
            vocab_size=100, max_seq_len=32, dropout=0.0,
        )
        self.model = GPT(self.cfg)
        self.model.eval()

    def _make_val_loader(self, tmp_path, n_tokens=500, seq_len=32, batch_size=4):
        tokens = np.random.randint(0, 100, size=n_tokens, dtype=np.uint16)
        shard_path = tmp_path / "val_000.bin"
        tokens.tofile(str(shard_path))
        meta = {"vocab_size": 100, "total_tokens": n_tokens}
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        ds = PretrainDataset(str(tmp_path), split="val", seq_len=seq_len)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size)

    def test_val_loss_is_positive(self, tmp_path):
        loader = self._make_val_loader(tmp_path)
        loss = compute_val_loss(self.model, loader, max_batches=3)
        assert loss > 0

    def test_accuracy_between_0_and_1(self, tmp_path):
        loader = self._make_val_loader(tmp_path)
        acc = compute_accuracy(self.model, loader, max_batches=3)
        assert 0.0 <= acc <= 1.0


class TestGenerate:
    def test_generate_returns_string(self):
        cfg = ModelConfig(
            n_layers=2, n_heads=4, d_model=64, d_ff=256,
            vocab_size=50257, max_seq_len=32, dropout=0.0,
        )
        model = GPT(cfg)
        model.eval()
        text = generate_text(model, prompt="Hello", max_new_tokens=10)
        assert isinstance(text, str)
        assert len(text) > len("Hello")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py -v`
Expected: FAIL

**Step 3: Implement evaluate.py**

```python
# nash_llm/eval/evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_val_loss(model: nn.Module, val_loader: DataLoader, max_batches: int | None = None) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for i, (x, y) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def compute_accuracy(model: nn.Module, val_loader: DataLoader, max_batches: int | None = None) -> float:
    model.eval()
    correct = 0
    total = 0

    for i, (x, y) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
        logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / total if total > 0 else 0.0
```

**Step 4: Implement generate.py**

```python
# nash_llm/eval/generate.py
import torch
import torch.nn as nn
from nash_llm.data.tokenizer import Tokenizer


def generate_text(
    model: nn.Module,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    tokenizer = Tokenizer()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(generated[0].tolist())
```

**Step 5: Update __init__.py**

```python
# nash_llm/eval/__init__.py
from nash_llm.eval.evaluate import compute_val_loss, compute_accuracy
from nash_llm.eval.generate import generate_text
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_eval.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add nash_llm/eval/ tests/test_eval.py
git commit -m "feat: evaluation (val_loss, accuracy) and text generation"
```

---

### Task 14: Training — Trainer

**Files:**
- Create: `nash_llm/training/trainer.py`
- Modify: `nash_llm/training/__init__.py`
- Create: `tests/test_trainer.py`

This is the largest single component. It integrates model, optimizer, scheduler, checkpoint, metrics.

**Step 1: Write the failing tests**

```python
# tests/test_trainer.py
import torch
import numpy as np
import json
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
            model=ModelConfig(
                n_layers=2, n_heads=4, d_model=64, d_ff=256,
                vocab_size=100, max_seq_len=32, dropout=0.0,
            ),
            train=TrainConfig(
                batch_size=4, learning_rate=3e-4, max_steps=10,
                warmup_steps=2, eval_interval=5, checkpoint_interval=5,
                grad_accum_steps=1,
            ),
            data=DataConfig(tokenized_dir=str(tmp_path)),
            metrics=MetricsConfig(wandb_enabled=False, log_interval=2),
        )

    def test_train_reduces_loss(self, tmp_path):
        cfg = self._make_config(tmp_path)
        ckpt_dir = str(tmp_path / "checkpoints")
        trainer = Trainer(cfg, checkpoint_dir=ckpt_dir)
        history = trainer.train()
        # Loss should decrease over 10 steps (at least not be NaN)
        assert len(history) > 0
        assert all(not np.isnan(h["train_loss"]) for h in history)

    def test_checkpoint_saved(self, tmp_path):
        cfg = self._make_config(tmp_path)
        ckpt_dir = str(tmp_path / "checkpoints")
        trainer = Trainer(cfg, checkpoint_dir=ckpt_dir)
        trainer.train()
        # checkpoint_interval=5, max_steps=10 => should have at least 1 checkpoint
        import glob
        ckpts = glob.glob(f"{ckpt_dir}/*.pt")
        assert len(ckpts) > 0

    def test_eval_runs(self, tmp_path):
        cfg = self._make_config(tmp_path)
        ckpt_dir = str(tmp_path / "checkpoints")
        trainer = Trainer(cfg, checkpoint_dir=ckpt_dir)
        history = trainer.train()
        # eval_interval=5 => should have eval metrics at step 5
        eval_entries = [h for h in history if "val_loss" in h]
        assert len(eval_entries) > 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trainer.py -v`
Expected: FAIL

**Step 3: Implement trainer.py**

```python
# nash_llm/training/trainer.py
import os
import math
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

        # Model
        self.model = GPT(config.model).to(self.device)

        # Optimizer
        self.optimizer = configure_optimizer(
            self.model,
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        # Scheduler
        min_lr = config.train.learning_rate / 10
        self.scheduler = CosineScheduler(
            max_lr=config.train.learning_rate,
            min_lr=min_lr,
            warmup_steps=config.train.warmup_steps,
            max_steps=config.train.max_steps,
        )

        # Data
        self.train_dataset = PretrainDataset(
            config.data.tokenized_dir, split="train", seq_len=config.model.max_seq_len,
        )
        self.val_dataset = PretrainDataset(
            config.data.tokenized_dir, split="val", seq_len=config.model.max_seq_len,
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.train.batch_size,
            shuffle=True, num_workers=0, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.train.batch_size,
            shuffle=False, num_workers=0,
        )

        # Metrics
        self.logger = MetricsLogger(config.metrics, run_config=None)

        # AMP scaler
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # State
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

    def train(self) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        cfg = self.config.train
        self.model.train()
        train_iter = iter(self.train_loader)

        for step in range(self.start_step, cfg.max_steps):
            lr = self._set_lr(step)

            # Gradient accumulation
            self.optimizer.zero_grad()
            accum_loss = 0.0

            for micro_step in range(cfg.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x, y = x.to(self.device), y.to(self.device)

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

            record = {"step": step, "train_loss": accum_loss, "lr": lr}

            # Logging
            if step % self.config.metrics.log_interval == 0:
                self.logger.log({"train_loss": accum_loss, "lr": lr}, step=step)

            # Eval
            if step > 0 and step % cfg.eval_interval == 0:
                self.model.eval()
                val_loss = compute_val_loss(self.model, self.val_loader, max_batches=20)
                accuracy = compute_accuracy(self.model, self.val_loader, max_batches=20)
                perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")

                eval_metrics = {"val_loss": val_loss, "accuracy": accuracy, "perplexity": perplexity}
                record.update(eval_metrics)
                self.logger.log(eval_metrics, step=step)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_checkpoint(
                        os.path.join(self.checkpoint_dir, "best.pt"),
                        self.model, self.optimizer, step=step,
                        config=self.config, metrics=eval_metrics,
                    )

                self.model.train()

            # Periodic checkpoint
            if step > 0 and step % cfg.checkpoint_interval == 0:
                save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"step_{step}.pt"),
                    self.model, self.optimizer, step=step,
                    config=self.config,
                )

            history.append(record)

        self.logger.finish()
        return history
```

**Step 4: Update __init__.py**

```python
# nash_llm/training/__init__.py
from nash_llm.training.trainer import Trainer
from nash_llm.training.scheduler import CosineScheduler
from nash_llm.training.checkpoint import save_checkpoint, load_checkpoint
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_trainer.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add nash_llm/training/ tests/test_trainer.py
git commit -m "feat: training loop with AMP, gradient accumulation, eval, checkpointing"
```

---

### Task 15: Scripts — Data Preparation

**Files:**
- Create: `scripts/prepare_data.py`

**Step 1: Implement prepare_data.py**

```python
# scripts/prepare_data.py
"""Download and tokenize a dataset into binary shards for pretraining."""
import argparse
import json
import os
import numpy as np
from pathlib import Path

from nash_llm.data.tokenizer import Tokenizer


DATASET_CONFIGS = {
    "tinystories_10M": {
        "hf_path": "roneneldan/TinyStories",
        "split": "train",
        "max_tokens": 10_000_000,
    },
    "openwebtext_100M": {
        "hf_path": "Skylion007/openwebtext",
        "split": "train",
        "max_tokens": 100_000_000,
    },
    "fineweb_1B": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "max_tokens": 1_000_000_000,
    },
}

SHARD_SIZE = 100_000_000  # 100M tokens per shard


def tokenize_dataset(dataset_key: str, output_dir: str, val_ratio: float = 0.01):
    from datasets import load_dataset

    config = DATASET_CONFIGS[dataset_key]
    tokenizer = Tokenizer()
    out_path = Path(output_dir) / dataset_key
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {config['hf_path']}")
    load_kwargs = {"path": config["hf_path"], "split": config["split"], "streaming": True}
    if "name" in config:
        load_kwargs["name"] = config["name"]
    ds = load_dataset(**load_kwargs)

    max_tokens = config["max_tokens"]
    all_tokens = []
    total = 0

    print(f"Tokenizing up to {max_tokens:,} tokens...")
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        tokens = tokenizer.encode(text)
        tokens.append(tokenizer.eot_token)
        all_tokens.extend(tokens)
        total += len(tokens)

        if total % 1_000_000 < len(tokens):
            print(f"  {total:,} tokens...")

        if total >= max_tokens:
            break

    all_tokens = all_tokens[:max_tokens]
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens):,}")

    # Split train/val
    val_size = int(len(all_tokens) * val_ratio)
    train_tokens = all_tokens[:-val_size] if val_size > 0 else all_tokens
    val_tokens = all_tokens[-val_size:] if val_size > 0 else np.array([], dtype=np.uint16)

    # Write shards
    def write_shards(tokens, prefix):
        for i in range(0, len(tokens), SHARD_SIZE):
            shard = tokens[i : i + SHARD_SIZE]
            shard_path = out_path / f"{prefix}_{i // SHARD_SIZE:03d}.bin"
            shard.tofile(str(shard_path))
            print(f"  Wrote {shard_path} ({len(shard):,} tokens)")

    print("Writing train shards...")
    write_shards(train_tokens, "train")
    print("Writing val shards...")
    write_shards(val_tokens, "val")

    # Write metadata
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "total_tokens": len(all_tokens),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "val_ratio": val_ratio,
    }
    (out_path / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Output: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare tokenized dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to prepare")
    parser.add_argument("--output", type=str, default="datasets/tokenized",
                        help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.01,
                        help="Fraction of data for validation")
    args = parser.parse_args()
    tokenize_dataset(args.dataset, args.output, args.val_ratio)


if __name__ == "__main__":
    main()
```

**Step 2: Verify it parses args**

Run: `python scripts/prepare_data.py --help`
Expected: Shows help with `--dataset`, `--output`, `--val_ratio` flags

**Step 3: Commit**

```bash
git add scripts/prepare_data.py
git commit -m "feat: data preparation script with HuggingFace download and tokenization"
```

---

### Task 16: Scripts — Train Entry Point

**Files:**
- Create: `scripts/train.py`

**Step 1: Implement train.py**

```python
# scripts/train.py
"""Main training entry point with config + CLI overrides."""
import argparse
import sys

from nash_llm.config import load_config
from nash_llm.training import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Nash-LLM")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Parse known args, collect the rest as overrides
    args, unknown = parser.parse_known_args()

    # Parse overrides: --model.n_layers 6 --train.learning_rate 1e-4
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            i += 1

    return args, overrides


def main():
    args, overrides = parse_args()
    config = load_config(args.config, overrides=overrides if overrides else None)

    print(f"Model: {config.model}")
    print(f"Train: {config.train}")
    print(f"Data:  {config.data}")

    trainer = Trainer(config, checkpoint_dir=args.checkpoint_dir, resume_from=args.resume)

    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Total parameters: {total_params:,}")

    history = trainer.train()
    print(f"Training complete. {len(history)} steps.")


if __name__ == "__main__":
    main()
```

**Step 2: Verify it parses args**

Run: `python scripts/train.py --help`
Expected: Shows help

**Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat: training entry point with CLI config overrides"
```

---

### Task 17: Scripts — Evaluate and Generate

**Files:**
- Create: `scripts/evaluate.py`
- Create: `scripts/generate.py`

**Step 1: Implement scripts/evaluate.py**

```python
# scripts/evaluate.py
"""Evaluate a checkpoint on validation data."""
import argparse
import math
import torch
from torch.utils.data import DataLoader

from nash_llm.config import load_config
from nash_llm.model import GPT
from nash_llm.data.dataset import PretrainDataset
from nash_llm.training.checkpoint import load_checkpoint
from nash_llm.eval import compute_val_loss, compute_accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Config YAML (uses checkpoint config if omitted)")
    parser.add_argument("--data_dir", type=str, default=None, help="Tokenized data directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint to get config
    raw_ckpt = torch.load(args.checkpoint, weights_only=False)
    if args.config:
        config = load_config(args.config)
    else:
        from nash_llm.config import NashConfig, ModelConfig
        model_cfg = ModelConfig(**raw_ckpt["config"]["model"])
        config = NashConfig(model=model_cfg)

    model = GPT(config.model).to(device)
    load_checkpoint(args.checkpoint, model)

    data_dir = args.data_dir or config.data.tokenized_dir
    val_dataset = PretrainDataset(data_dir, split="val", seq_len=config.model.max_seq_len)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    val_loss = compute_val_loss(model, val_loader, max_batches=args.max_batches)
    accuracy = compute_accuracy(model, val_loader, max_batches=args.max_batches)
    perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")

    print(f"val_loss:    {val_loss:.4f}")
    print(f"accuracy:    {accuracy:.4f}")
    print(f"perplexity:  {perplexity:.2f}")


if __name__ == "__main__":
    main()
```

**Step 2: Implement scripts/generate.py**

```python
# scripts/generate.py
"""Generate text from a trained checkpoint."""
import argparse
import torch

from nash_llm.config import load_config, NashConfig, ModelConfig
from nash_llm.model import GPT
from nash_llm.training.checkpoint import load_checkpoint
from nash_llm.eval import generate_text


def main():
    parser = argparse.ArgumentParser(description="Generate text")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_ckpt = torch.load(args.checkpoint, weights_only=False)
    model_cfg = ModelConfig(**raw_ckpt["config"]["model"])
    model = GPT(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    text = generate_text(model, prompt=args.prompt, max_new_tokens=args.max_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    print(text)


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add scripts/evaluate.py scripts/generate.py
git commit -m "feat: evaluate and generate scripts"
```

---

### Task 18: Scripts — Plot Metrics and Compare Runs

**Files:**
- Create: `scripts/plot_metrics.py`
- Create: `scripts/compare_runs.py`

**Step 1: Implement scripts/plot_metrics.py**

```python
# scripts/plot_metrics.py
"""Plot training metrics from a wandb run."""
import argparse
import matplotlib.pyplot as plt
import wandb


def main():
    parser = argparse.ArgumentParser(description="Plot metrics from wandb run")
    parser.add_argument("--run_id", type=str, required=True, help="wandb run ID")
    parser.add_argument("--project", type=str, default="nash-llm")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--metrics", type=str, default="val_loss", help="Comma-separated metric names")
    parser.add_argument("--output", type=str, default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    api = wandb.Api()
    run_path = f"{args.entity + '/' if args.entity else ''}{args.project}/{args.run_id}"
    run = api.run(run_path)
    history = run.scan_history()

    metrics = [m.strip() for m in args.metrics.split(",")]
    data = {m: {"steps": [], "values": []} for m in metrics}

    for row in history:
        step = row.get("_step", 0)
        for m in metrics:
            if m in row and row[m] is not None:
                data[m]["steps"].append(step)
                data[m]["values"].append(row[m])

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), squeeze=False)
    for i, m in enumerate(metrics):
        ax = axes[i][0]
        ax.plot(data[m]["steps"], data[m]["values"])
        ax.set_xlabel("Step")
        ax.set_ylabel(m)
        ax.set_title(f"{m} — {run.name}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
```

**Step 2: Implement scripts/compare_runs.py**

```python
# scripts/compare_runs.py
"""Compare multiple wandb training runs."""
import argparse
import matplotlib.pyplot as plt
import wandb


def main():
    parser = argparse.ArgumentParser(description="Compare wandb runs")
    parser.add_argument("--runs", type=str, required=True, help="Comma-separated run IDs")
    parser.add_argument("--project", type=str, default="nash-llm")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--metrics", type=str, default="val_loss", help="Comma-separated metric names")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    api = wandb.Api()
    run_ids = [r.strip() for r in args.runs.split(",")]
    metrics = [m.strip() for m in args.metrics.split(",")]

    runs_data = {}
    configs = {}

    for run_id in run_ids:
        run_path = f"{args.entity + '/' if args.entity else ''}{args.project}/{run_id}"
        run = api.run(run_path)
        runs_data[run_id] = {"name": run.name, "metrics": {m: {"steps": [], "values": []} for m in metrics}}
        configs[run_id] = run.config

        for row in run.scan_history():
            step = row.get("_step", 0)
            for m in metrics:
                if m in row and row[m] is not None:
                    runs_data[run_id]["metrics"][m]["steps"].append(step)
                    runs_data[run_id]["metrics"][m]["values"].append(row[m])

    # Plot overlaid metrics
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), squeeze=False)
    for i, m in enumerate(metrics):
        ax = axes[i][0]
        for run_id in run_ids:
            d = runs_data[run_id]["metrics"][m]
            label = runs_data[run_id]["name"] or run_id
            ax.plot(d["steps"], d["values"], label=label)
        ax.set_xlabel("Step")
        ax.set_ylabel(m)
        ax.set_title(m)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()

    # Print final metrics table
    print(f"\n{'Run':<30} ", end="")
    for m in metrics:
        print(f"{m:<15} ", end="")
    print()
    print("-" * (30 + 16 * len(metrics)))

    for run_id in run_ids:
        name = runs_data[run_id]["name"] or run_id
        print(f"{name:<30} ", end="")
        for m in metrics:
            vals = runs_data[run_id]["metrics"][m]["values"]
            final = f"{vals[-1]:.4f}" if vals else "N/A"
            print(f"{final:<15} ", end="")
        print()

    # Config diff
    if len(run_ids) == 2:
        print(f"\nConfig diff between {run_ids[0]} and {run_ids[1]}:")
        c1, c2 = configs[run_ids[0]], configs[run_ids[1]]
        all_keys = set(list(c1.keys()) + list(c2.keys()))
        for key in sorted(all_keys):
            v1 = c1.get(key)
            v2 = c2.get(key)
            if v1 != v2:
                print(f"  {key}: {v1} -> {v2}")


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add scripts/plot_metrics.py scripts/compare_runs.py
git commit -m "feat: metric plotting and run comparison scripts"
```

---

### Task 19: YAML Config Presets

**Files:**
- Create: `configs/pretrain_100m.yaml`
- Create: `configs/pretrain_small.yaml`
- Create: `configs/sft.yaml`

**Step 1: Create pretrain_100m.yaml**

```yaml
# configs/pretrain_100m.yaml
model:
  n_layers: 12
  n_heads: 12
  d_model: 768
  d_ff: 3072
  vocab_size: 50257
  max_seq_len: 1024
  dropout: 0.1

train:
  batch_size: 64
  learning_rate: 3.0e-4
  weight_decay: 0.1
  max_steps: 100000
  warmup_steps: 2000
  grad_clip: 1.0
  eval_interval: 500
  checkpoint_interval: 5000
  grad_accum_steps: 4

data:
  dataset: openwebtext
  dataset_size: 100M
  tokenized_dir: datasets/tokenized/openwebtext_100M

metrics:
  wandb_project: nash-llm
  wandb_enabled: true
  log_interval: 10
  metrics:
    - val_loss
    - accuracy
```

**Step 2: Create pretrain_small.yaml**

```yaml
# configs/pretrain_small.yaml
# Small model for debugging and testing
model:
  n_layers: 4
  n_heads: 4
  d_model: 256
  d_ff: 1024
  vocab_size: 50257
  max_seq_len: 256
  dropout: 0.1

train:
  batch_size: 16
  learning_rate: 1.0e-3
  weight_decay: 0.1
  max_steps: 1000
  warmup_steps: 100
  grad_clip: 1.0
  eval_interval: 100
  checkpoint_interval: 500
  grad_accum_steps: 1

data:
  dataset: tinystories
  dataset_size: 10M
  tokenized_dir: datasets/tokenized/tinystories_10M

metrics:
  wandb_project: nash-llm
  wandb_enabled: true
  log_interval: 10
  metrics:
    - val_loss
    - accuracy
```

**Step 3: Create sft.yaml**

```yaml
# configs/sft.yaml
model:
  n_layers: 12
  n_heads: 12
  d_model: 768
  d_ff: 3072
  vocab_size: 50257
  max_seq_len: 1024
  dropout: 0.1

train:
  batch_size: 8
  learning_rate: 1.0e-5
  weight_decay: 0.01
  max_steps: 5000
  warmup_steps: 200
  grad_clip: 1.0
  eval_interval: 100
  checkpoint_interval: 1000
  grad_accum_steps: 2

metrics:
  wandb_project: nash-llm
  wandb_enabled: true
  log_interval: 10
  metrics:
    - val_loss
    - accuracy
```

**Step 4: Commit**

```bash
git add configs/
git commit -m "feat: YAML config presets for pretrain 100M, small debug, and SFT"
```

---

### Task 20: Integration Test — End-to-End Smoke Test

**Files:**
- Create: `tests/test_e2e.py`

**Step 1: Write the smoke test**

```python
# tests/test_e2e.py
"""End-to-end smoke test: prepare fake data, train a tiny model, evaluate, generate."""
import numpy as np
import json
import torch
from nash_llm.config import NashConfig, ModelConfig, TrainConfig, DataConfig, MetricsConfig
from nash_llm.training import Trainer
from nash_llm.training.checkpoint import load_checkpoint
from nash_llm.model import GPT
from nash_llm.eval import compute_val_loss, generate_text


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

        # 4. Generate text (just verify it doesn't crash with small vocab)
        # Use model.generate directly since generate_text uses tiktoken
        prompt = torch.randint(0, 500, (1, 5))
        generated = model.generate(prompt, max_new_tokens=10)
        assert generated.shape[1] == 15
```

**Step 2: Run the test**

Run: `pytest tests/test_e2e.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end smoke test for full training pipeline"
```

---

### Task 21: Run All Tests

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Final commit if any fixes needed**

Fix any issues, then:
```bash
git add -A && git commit -m "fix: test suite cleanup"
```
