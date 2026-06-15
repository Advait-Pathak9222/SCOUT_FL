"""Federated server: global-model lifecycle + evaluation.

The server owns the global model and does broadcast / apply-aggregated-update /
evaluate. It does NOT decide client selection — the runner passes in the
selected indices (selection stays modular). This keeps SCOUT-FL and the
baselines interchangeable at the call site.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from scout_fl.fl.models import get_flat_params, set_flat_params

_LOSS_SUM = nn.CrossEntropyLoss(reduction="sum")


class FLServer:
    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.device = device
        self.model = model.to(device)

    def global_flat(self) -> np.ndarray:
        return get_flat_params(self.model).cpu().numpy()

    def set_global(self, flat) -> None:
        set_flat_params(self.model, np.asarray(flat, dtype=np.float32))

    def apply_aggregated_update(self, base_global: np.ndarray, agg_update: np.ndarray) -> None:
        """global <- base_global + aggregated_update (FedAvg of deltas)."""
        self.set_global(np.asarray(base_global, dtype=np.float32) + np.asarray(agg_update, dtype=np.float32))

    @torch.no_grad()
    def evaluate(self, x_test: torch.Tensor, y_test: torch.Tensor,
                 batch_size: int = 512) -> tuple[float, float]:
        """Return (test_loss, test_accuracy)."""
        self.model.eval()
        n = len(y_test)
        total_loss, correct = 0.0, 0
        for i in range(0, n, batch_size):
            xb = x_test[i:i + batch_size].to(self.device)
            yb = y_test[i:i + batch_size].to(self.device)
            out = self.model(xb)
            total_loss += _LOSS_SUM(out, yb).item()
            correct += (out.argmax(1) == yb).sum().item()
        return total_loss / n, correct / n
