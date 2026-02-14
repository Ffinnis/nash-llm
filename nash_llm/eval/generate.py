import torch
import torch.nn as nn
from nash_llm.data.tokenizer import Tokenizer


def generate_text(model: nn.Module, prompt: str, max_new_tokens: int = 100, temperature: float = 1.0, top_k: int | None = None) -> str:
    tokenizer = Tokenizer()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(generated[0].tolist())
