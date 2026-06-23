"""Scaffolded adapters for the external wireless-sensing datasets in the campaign's
"scaffold" tier: **DeepMIMO**, **DeepSense 6G**, and **RadarScenes**.

These datasets are large and license-gated, so they are not bundled. Each adapter
follows a simple drop-in contract and falls back to a clearly-labelled synthetic
generator when the real data is absent — so the full campaign runs end-to-end now
and swaps to real data later with no code change.

Two roles an external source can play:

1. **FL classification task** (`load_external_classification`) — e.g. DeepSense-6G
   beam prediction, RadarScenes object classification. Drop a NumPy archive at
   ``<root>/<name>/<name>.npz`` with arrays ``x_train, y_train, x_test, y_test``
   (x: (N, F) or (N, C, H, W); y: int64 labels). Returns an ``FLDataset``.

2. **Wireless / sensing simulator inputs** (`load_channel_realizations`,
   `load_sensing_geometry`) — e.g. DeepMIMO channel gains, real client/target
   geometry. Drop ``<root>/<name>/channels.npy`` (per-client gains) or
   ``geometry.npz`` (clients, targets). Returns arrays ready for ``build_scenario``.
   (Wiring into ``build_scenario`` is left as a documented hook — see README.)
"""
from __future__ import annotations

import os

import numpy as np
import torch

from scout_fl.fl.datasets import FLDataset

# default shape/class counts for each source's synthetic stand-in
_EXTERNAL_SPEC = {
    "deepmimo": {"features": 256, "classes": 64, "note": "mmWave beam index prediction"},
    "deepsense6g": {"features": 512, "classes": 64, "note": "multi-modal beam prediction"},
    "deepsense": {"features": 512, "classes": 64, "note": "multi-modal beam prediction"},
    "radarscenes": {"features": 128, "classes": 6, "note": "radar point-cloud object class"},
}


def _npz_path(name: str, root: str) -> str:
    return os.path.join(root, name, f"{name}.npz")


def _synthetic_classification(name: str, seed: int = 0) -> FLDataset:
    spec = _EXTERNAL_SPEC[name]
    f, c = spec["features"], spec["classes"]
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((c, f)) * 1.5

    def make(n):
        y = rng.integers(0, c, size=n).astype(np.int64)
        x = (centers[y] + rng.standard_normal((n, f))).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    xtr, ytr = make(4000)
    xte, yte = make(1000)
    print(f"[{name}] real data not found; using synthetic fallback "
          f"({spec['note']}, F={f}, C={c}). Drop '{name}.npz' under the data root to use real data.")
    return FLDataset(x_train=xtr, y_train=ytr, x_test=xte, y_test=yte,
                     num_classes=c, input_shape=(f,))


def load_external_classification(name: str, root: str = "data") -> FLDataset:
    """Load a DeepMIMO/DeepSense-6G/RadarScenes FL classification task (synthetic fallback)."""
    key = name.lower()
    if key not in _EXTERNAL_SPEC:
        raise ValueError(f"unknown external dataset {name!r}; supported: {list(_EXTERNAL_SPEC)}")
    path = _npz_path(key, root)
    if os.path.exists(path):
        d = np.load(path)
        return FLDataset(
            x_train=torch.as_tensor(d["x_train"], dtype=torch.float32),
            y_train=torch.as_tensor(d["y_train"], dtype=torch.long),
            x_test=torch.as_tensor(d["x_test"], dtype=torch.float32),
            y_test=torch.as_tensor(d["y_test"], dtype=torch.long),
            num_classes=int(d["y_train"].max()) + 1,
            input_shape=tuple(np.asarray(d["x_train"]).shape[1:]),
        )
    return _synthetic_classification(key)


def load_channel_realizations(name: str, num_clients: int, rng, root: str = "data") -> np.ndarray:
    """Per-client comm channel gains |h_k|^2 from a real source (DeepMIMO), synthetic fallback.

    Real contract: ``<root>/<name>/channels.npy`` of shape (>=num_clients,) or (T, >=num_clients).
    """
    path = os.path.join(root, name.lower(), "channels.npy")
    if os.path.exists(path):
        gains = np.load(path)
        gains = gains.reshape(-1, gains.shape[-1])[0] if gains.ndim > 1 else gains
        if len(gains) >= num_clients:
            return np.asarray(gains[:num_clients], dtype=float)
        print(f"[{name}] channels.npy has too few clients ({len(gains)}<{num_clients}); padding synthetically.")
    # synthetic fallback: log-normal shadowing (matches the analytic channel model's scale)
    return rng.lognormal(mean=0.0, sigma=0.5, size=num_clients)


def load_sensing_geometry(name: str, num_clients: int, num_targets: int, rng,
                          area: float = 100.0, root: str = "data"):
    """Real (client, target) positions from a sensing dataset (RadarScenes/DeepSense), synthetic fallback.

    Real contract: ``<root>/<name>/geometry.npz`` with ``clients`` (K,2) and ``targets`` (M,2).
    """
    path = os.path.join(root, name.lower(), "geometry.npz")
    if os.path.exists(path):
        d = np.load(path)
        return np.asarray(d["clients"], dtype=float)[:num_clients], \
            np.asarray(d["targets"], dtype=float)[:num_targets]
    clients = rng.uniform(0, area, (num_clients, 2))
    targets = rng.uniform(0, area, (num_targets, 2))
    return clients, targets
