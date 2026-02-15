import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_val_loss(model: nn.Module, val_loader: DataLoader, max_batches: int | None = None) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    device = next(model.parameters()).device

    for i, (x, y) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def compute_accuracy(model: nn.Module, val_loader: DataLoader, max_batches: int | None = None) -> float:
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device

    for i, (x, y) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / total if total > 0 else 0.0
