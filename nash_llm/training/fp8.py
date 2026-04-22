import torch
import torch.nn as nn


_FP8_LINEAR_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc_gate",
    "fc2",
)


def _torch_version_tuple() -> tuple[int, int, int]:
    core = torch.__version__.split("+", 1)[0]
    parts = core.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return major, minor, patch


def _torch_supports_fp8() -> bool:
    return _torch_version_tuple() >= (2, 11, 0)


def _cuda_supports_fp8(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return major > 8 or (major == 8 and minor >= 9)


def should_convert_linear_to_fp8(module: nn.Module, fqn: str) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if fqn == "lm_head":
        return False
    if fqn.rsplit(".", 1)[-1] not in _FP8_LINEAR_SUFFIXES:
        return False
    return module.in_features % 16 == 0 and module.out_features % 16 == 0


def apply_fp8_training(model: nn.Module, device: torch.device) -> nn.Module:
    if not _torch_supports_fp8():
        raise RuntimeError(
            "train.precision=fp8 requires torch>=2.11. "
            f"Found torch {torch.__version__}."
        )
    if not _cuda_supports_fp8(device):
        raise RuntimeError(
            "train.precision=fp8 requires a CUDA GPU with native FP8 support "
            "(Hopper, Ada, or newer)."
        )
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "train.precision=fp8 requires the torchao package. "
            "Install project dependencies with `uv sync` on the target machine."
        ) from exc

    config = Float8LinearConfig.from_recipe_name("tensorwise")
    convert_to_float8_training(
        model,
        config=config,
        module_filter_fn=should_convert_linear_to_fp8,
    )
    return model
