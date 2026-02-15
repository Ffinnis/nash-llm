import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_val_loss(
    model: nn.Module, val_loader: DataLoader, max_batches: int | None = None,
    amp_dtype: torch.dtype | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    device = next(model.parameters()).device
    use_amp = amp_dtype is not None and device.type == "cuda"

    for i, (x, y) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def compute_accuracy(
    model: nn.Module, val_loader: DataLoader, max_batches: int | None = None,
    amp_dtype: torch.dtype | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    use_amp = amp_dtype is not None and device.type == "cuda"

    for i, (x, y) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / total if total > 0 else 0.0
