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
