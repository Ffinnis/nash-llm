"""Backward-compatibility shim for the old module path."""

from nash_llm.optim.sage import Sage, configure_optimizers

__all__ = ["Sage", "configure_optimizers"]
