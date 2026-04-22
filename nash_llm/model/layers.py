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
    raise ValueError(f"Unsupported model.activation '{name}'. Expected one of: gelu, relu2")


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.act = _build_activation(config.activation)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))
