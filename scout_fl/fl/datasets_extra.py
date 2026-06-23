"""UCI HAR (Human Activity Recognition Using Smartphones) — the tabular wireless-
sensing-adjacent FL task in the campaign's "ready now" tier.

561 hand-engineered inertial features, 6 activities (walking, walking-up,
walking-down, sitting, standing, laying), 30 subjects. The natural federated
split is per-subject; here the runner applies its spatial/Dirichlet partition.

Source: https://archive.ics.uci.edu/dataset/240 ("UCI HAR Dataset.zip"). Place
the unzipped ``UCI HAR Dataset/`` folder under ``<root>/`` (or let ``download=True``
fetch it). If the data is absent and download fails/disabled, a clearly-labelled
synthetic Gaussian-cluster fallback is returned so the pipeline still runs end-to-end.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from scout_fl.fl.datasets import FLDataset

_HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
_HAR_DIR = "UCI HAR Dataset"
_NUM_FEATURES = 561
_NUM_CLASSES = 6


def _read_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)


def _load_from_disk(base: str):
    """Read the standard UCI HAR train/test split from an unzipped folder."""
    x_train = _read_matrix(os.path.join(base, "train", "X_train.txt"))
    y_train = _read_matrix(os.path.join(base, "train", "y_train.txt")).astype(np.int64) - 1
    x_test = _read_matrix(os.path.join(base, "test", "X_test.txt"))
    y_test = _read_matrix(os.path.join(base, "test", "y_test.txt")).astype(np.int64) - 1
    return x_train, y_train, x_test, y_test


def _maybe_download(root: str) -> str | None:
    """Return the path to an unzipped UCI HAR Dataset folder, downloading if needed."""
    for cand in (os.path.join(root, _HAR_DIR), os.path.join(root, "UCI_HAR", _HAR_DIR)):
        if os.path.isdir(cand):
            return cand
    try:
        import urllib.request
        import zipfile

        os.makedirs(root, exist_ok=True)
        zpath = os.path.join(root, "uci_har.zip")
        if not os.path.exists(zpath):
            urllib.request.urlretrieve(_HAR_URL, zpath)
        with zipfile.ZipFile(zpath) as zf:                    # outer zip contains inner "UCI HAR Dataset.zip"
            zf.extractall(root)
        inner = os.path.join(root, "UCI HAR Dataset.zip")
        if os.path.exists(inner):
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(root)
        cand = os.path.join(root, _HAR_DIR)
        return cand if os.path.isdir(cand) else None
    except Exception as exc:                                  # network/parse failure -> fallback
        print(f"[uci_har] download failed ({exc}); using synthetic fallback.")
        return None


def _synthetic(seed: int = 0):
    """Labelled Gaussian clusters with the UCI HAR shape — keeps the pipeline runnable offline."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((_NUM_CLASSES, _NUM_FEATURES)) * 2.0

    def make(n):
        y = rng.integers(0, _NUM_CLASSES, size=n)
        x = centers[y] + rng.standard_normal((n, _NUM_FEATURES))
        return x.astype(np.float32), y.astype(np.int64)

    xtr, ytr = make(3000)
    xte, yte = make(800)
    return xtr, ytr, xte, yte


def load_uci_har(root: str = "data", download: bool = True) -> FLDataset:
    """Load UCI HAR as standardized feature tensors (synthetic fallback if unavailable)."""
    base = _maybe_download(root) if download else (
        os.path.join(root, _HAR_DIR) if os.path.isdir(os.path.join(root, _HAR_DIR)) else None)
    if base is not None:
        x_train, y_train, x_test, y_test = _load_from_disk(base)
    else:
        if download:
            print("[uci_har] dataset not found; using synthetic fallback "
                  "(place 'UCI HAR Dataset/' under the data root for the real data).")
        x_train, y_train, x_test, y_test = _synthetic()

    mu, sd = x_train.mean(0, keepdims=True), x_train.std(0, keepdims=True) + 1e-6
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd
    return FLDataset(
        x_train=torch.from_numpy(x_train), y_train=torch.from_numpy(y_train),
        x_test=torch.from_numpy(x_test), y_test=torch.from_numpy(y_test),
        num_classes=_NUM_CLASSES, input_shape=(_NUM_FEATURES,),
    )
