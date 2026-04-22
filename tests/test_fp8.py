from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from nash_llm.model import GPT
from nash_llm.training.fp8 import apply_fp8_training, should_convert_linear_to_fp8
from nash_llm.config import ModelConfig


class TestFP8ConversionFilter:
    def test_hidden_transformer_linears_are_eligible(self):
        assert should_convert_linear_to_fp8(nn.Linear(64, 64), "blocks.0.attn.q_proj")
        assert should_convert_linear_to_fp8(nn.Linear(64, 64), "blocks.0.attn.out_proj")
        assert should_convert_linear_to_fp8(nn.Linear(64, 64), "blocks.0.ff.fc1")
        assert should_convert_linear_to_fp8(nn.Linear(64, 64), "blocks.0.ff.fc_gate")
        assert should_convert_linear_to_fp8(nn.Linear(64, 64), "blocks.0.ff.fc2")

    def test_lm_head_and_non_linear_modules_are_excluded(self):
        assert not should_convert_linear_to_fp8(nn.Linear(64, 64), "lm_head")
        assert not should_convert_linear_to_fp8(nn.Embedding(64, 64), "token_emb")

    def test_non_aligned_linears_are_excluded(self):
        assert not should_convert_linear_to_fp8(nn.Linear(64, 63), "blocks.0.ff.fc2")
        assert not should_convert_linear_to_fp8(nn.Linear(63, 64), "blocks.0.attn.q_proj")


class TestApplyFP8Training:
    def test_requires_supported_torch_version(self):
        model = GPT(ModelConfig(n_layers=1, n_heads=4, d_model=64, d_ff=256, vocab_size=256, max_seq_len=32))
        with patch("nash_llm.training.fp8._torch_version_tuple", return_value=(2, 10, 0)):
            with pytest.raises(RuntimeError, match="torch>=2.11"):
                apply_fp8_training(model, torch.device("cuda"))

    def test_requires_supported_gpu_capability(self):
        model = GPT(ModelConfig(n_layers=1, n_heads=4, d_model=64, d_ff=256, vocab_size=256, max_seq_len=32))
        with patch("nash_llm.training.fp8._torch_version_tuple", return_value=(2, 11, 0)):
            with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
                with pytest.raises(RuntimeError, match="Hopper, Ada, or newer"):
                    apply_fp8_training(model, torch.device("cuda"))

    def test_converts_supported_hidden_linears_only(self):
        model = GPT(ModelConfig(n_layers=1, n_heads=4, d_model=64, d_ff=256, vocab_size=256, max_seq_len=32))

        converted = []

        def fake_convert(module, *, module_filter_fn, config):
            for fqn, child in module.named_modules():
                if fqn and module_filter_fn(child, fqn):
                    converted.append(fqn)
            return module

        fake_float8_config = SimpleNamespace(from_recipe_name=lambda _: "tensorwise-config")
        fake_float8_module = SimpleNamespace(
            Float8LinearConfig=fake_float8_config,
            convert_to_float8_training=fake_convert,
        )
        fake_torchao_package = SimpleNamespace(float8=fake_float8_module)

        with patch("nash_llm.training.fp8._torch_version_tuple", return_value=(2, 11, 0)):
            with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
                with patch.dict(
                    "sys.modules",
                    {"torchao": fake_torchao_package, "torchao.float8": fake_float8_module},
                ):
                    apply_fp8_training(model, torch.device("cuda"))

        assert "blocks.0.attn.q_proj" in converted
        assert "blocks.0.attn.k_proj" in converted
        assert "blocks.0.attn.v_proj" in converted
        assert "blocks.0.attn.out_proj" in converted
        assert "blocks.0.ff.fc1" in converted
        assert "blocks.0.ff.fc2" in converted
        assert "lm_head" not in converted
