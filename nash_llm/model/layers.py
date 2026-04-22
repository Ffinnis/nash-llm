import math
import torch
import torch.nn as nn
from nash_llm.config import ModelConfig


class ReLU2(nn.Module):
    """Squared ReLU activation from Primer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(torch.relu(x))


def _build_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu2":
        return ReLU2()
    raise ValueError(f"Unsupported model.activation '{name}'. Expected one of: gelu, relu2, swiglu")


def _swiglu_hidden_dim(d_ff: int) -> int:
    # GLU variants use three matrices instead of two; shrink hidden width to
    # roughly 2/3 of the original FFN width to keep params/compute comparable.
    return max(1, math.floor((2 * d_ff) / 3))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.activation_name = config.activation
        hidden_dim = config.d_ff if config.activation != "swiglu" else _swiglu_hidden_dim(config.d_ff)
        self.fc1 = nn.Linear(config.d_model, hidden_dim)
        self.fc_gate = nn.Linear(config.d_model, hidden_dim) if config.activation == "swiglu" else None
        self.fc2 = nn.Linear(hidden_dim, config.d_model)
        self.act = _build_activation(config.activation) if config.activation != "swiglu" else None
        self.gate_act = nn.SiLU() if config.activation == "swiglu" else None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "swiglu":
            assert self.fc_gate is not None
            assert self.gate_act is not None
            hidden = self.gate_act(self.fc1(x)) * self.fc_gate(x)
            return self.dropout(self.fc2(hidden))
        assert self.act is not None
        return self.dropout(self.fc2(self.act(self.fc1(x))))
