"""Client-side federated computation: cheap probe (for selection) + local SGD.

* ``probe_loss_and_embedding`` — one (few) mini-batch forward/backward on the
  CURRENT global model, returning the local loss and the flattened gradient.
  The gradient is the per-client *embedding* fed to the learning utility
  (f_learn) for selection — real, not a placeholder, and cheap (no full train).
* ``local_train`` — full local SGD for the SELECTED clients; returns the model
  update vector (delta = trained - global), local loss, and sample count.

The caller resets the shared model to the global state before each client (so a
single model instance is reused — no K model copies).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scout_fl.fl.models import get_flat_grad, get_flat_params

_LOSS = nn.CrossEntropyLoss()


def probe_loss_and_embedding(model, dataset, *, batch_size: int = 64,
                             device: str = "cpu", max_batches: int = 1):
    """Return (mean_loss, flat_gradient) from a probe pass on the global model."""
    model.train()
    model.zero_grad(set_to_none=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total, n, batches = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = _LOSS(model(xb), yb)
        loss.backward()
        total += loss.item() * len(yb)
        n += len(yb)
        batches += 1
        if batches >= max_batches:
            break
    grad = get_flat_grad(model).cpu().numpy()
    model.zero_grad(set_to_none=True)
    return total / max(n, 1), grad


def local_train(model, dataset, *, epochs: int = 1, lr: float = 0.05,
                batch_size: int = 64, optimizer: str = "sgd", momentum: float = 0.9,
                device: str = "cpu", max_steps: int | None = None) -> dict:
    """Train ``model`` (already set to the global state) locally; return the update.

    ``max_steps=1`` performs a single SGD step (the FedSGD update rule); ``None``
    runs the full ``epochs`` of mini-batch SGD (the FedAvg update rule).
    """
    init = get_flat_params(model).clone()
    model.train()
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total, n, steps = 0.0, 0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = _LOSS(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        if max_steps is not None and steps >= max_steps:
            break
    update = (get_flat_params(model) - init).cpu().numpy()
    return {"update": update, "loss": total / max(n, 1), "num_samples": len(dataset)}
