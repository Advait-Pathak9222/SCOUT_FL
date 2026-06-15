"""Information-matrix functionals: log-det (D-optimal) and CRB / trace-inverse
(A-optimal), both batched over a leading stack of targets.

- ``logdet_spd``  -> the D-optimal criterion used as SCOUT-FL's *objective*
  (monotone submodular; see objectives/sensing_utility.py).
- ``crb_trace``   -> the A-optimal criterion (Cramer-Rao bound on position),
  used as a *constraint and evaluation metric* (only weakly submodular, so it
  is deliberately NOT the optimization objective).
"""
from __future__ import annotations

import numpy as np


def logdet_spd(mat: np.ndarray) -> np.ndarray:
    """Stable log-determinant of an SPD matrix stack ``(..., d, d)`` -> ``(...)``.

    Uses ``slogdet`` (robust, batched). For SPD inputs the sign is +1.
    """
    sign, logdet = np.linalg.slogdet(mat)
    return logdet


def crb_trace(mat: np.ndarray, reg: float = 0.0) -> np.ndarray:
    """Trace of the inverse FIM (A-optimal CRB) for an SPD stack ``(..., d, d)``.

    ``reg`` adds optional Tikhonov jitter for conditioning. Returns ``(...)``.
    """
    if reg:
        mat = mat + reg * np.eye(mat.shape[-1])
    inv = np.linalg.inv(mat)
    return np.trace(inv, axis1=-2, axis2=-1)


def accumulate(prior: np.ndarray, fim_stack: np.ndarray, idx) -> np.ndarray:
    """Accumulated FIM ``J_0 + sum_{k in idx} J_k`` for a target stack.

    Parameters
    ----------
    prior : (M, d, d)
    fim_stack : (K, M, d, d)
    idx : iterable of client indices
    """
    acc = np.array(prior, dtype=float, copy=True)
    idx = list(idx)
    if idx:
        acc = acc + fim_stack[idx].sum(axis=0)
    return acc
