"""Federated data partitioning across clients: IID and Dirichlet label-skew.

* ``partition_iid``       — shuffle and split into balanced shards.
* ``partition_dirichlet`` — the standard label-distribution-skew non-IID scheme
  (Hsu et al. 2019): for each class, draw a Dirichlet(alpha) split across clients.
  Small ``alpha`` => high skew (each client sees few classes); large ``alpha``
  => near-IID. A ``min_size`` guard re-draws until every client is non-trivial.

``partition_report`` summarizes shard sizes + per-client class histogram and a
``mean_top_class_fraction`` non-IID-severity scalar (≈1/num_classes for IID,
→1 for extreme skew). NumPy-only; torch is not required here.

NOTE: a third **spatial** scheme (partition by client region, coupling non-IID
to the ISAC coverage map) is a planned SCOUT-FL-specific extension; for now we
ship IID + Dirichlet as requested.
"""
from __future__ import annotations

import numpy as np


def partition_iid(labels, num_clients: int, rng: np.random.Generator):
    """Balanced random shards -> list of sorted index arrays."""
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    return [np.sort(part) for part in np.array_split(idx, num_clients)]


def partition_dirichlet(labels, num_clients: int, alpha: float,
                        rng: np.random.Generator, min_size: int = 1,
                        max_tries: int = 100):
    """Dirichlet label-skew partition -> list of sorted index arrays."""
    labels = np.asarray(labels)
    n_classes = int(labels.max()) + 1
    buckets = None
    for _ in range(max_tries):
        buckets = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(props)[:-1] * len(idx_c)).astype(int)
            for k, part in enumerate(np.split(idx_c, cuts)):
                buckets[k].extend(part.tolist())
        if min(len(b) for b in buckets) >= min_size:
            break
    return [np.sort(np.array(b, dtype=int)) for b in buckets]


def partition(labels, num_clients: int, scheme: str = "iid", alpha: float = 0.5,
              rng: np.random.Generator | None = None, **kwargs):
    """Dispatch to a partitioning scheme: ``iid`` | ``dirichlet``."""
    rng = rng if rng is not None else np.random.default_rng(0)
    if scheme == "iid":
        return partition_iid(labels, num_clients, rng)
    if scheme == "dirichlet":
        return partition_dirichlet(labels, num_clients, alpha, rng, **kwargs)
    raise ValueError(f"unknown partition scheme {scheme!r} (use 'iid' or 'dirichlet')")


def partition_report(labels, parts, num_classes: int | None = None) -> dict:
    """Shard sizes + per-client class histogram + non-IID severity scalar."""
    labels = np.asarray(labels)
    num_classes = num_classes if num_classes is not None else int(labels.max()) + 1
    sizes = [int(len(p)) for p in parts]
    hist = np.zeros((len(parts), num_classes), dtype=int)
    for k, p in enumerate(parts):
        if len(p):
            classes, counts = np.unique(labels[p], return_counts=True)
            hist[k, classes.astype(int)] = counts
    fracs = hist / np.clip(hist.sum(axis=1, keepdims=True), 1, None)
    return {
        "num_clients": len(parts),
        "sizes": sizes,
        "min_size": int(min(sizes)),
        "max_size": int(max(sizes)),
        "total": int(sum(sizes)),
        "mean_top_class_fraction": float(np.mean(fracs.max(axis=1))),
        "class_hist": hist.tolist(),
    }


def partition_spatial(labels, cluster_assignment, alpha: float,
                      rng: np.random.Generator, min_size: int = 1):
    """Spatial non-IID: clients in the same viewpoint cluster share a label skew.

    Cluster-level Dirichlet partition, then an IID split within each cluster — so
    nearby clients (same sensing bearing) also share a data distribution, coupling
    the learning and sensing heterogeneity (the SCOUT/JEDI-relevant non-IID).
    """
    labels = np.asarray(labels)
    cluster_assignment = np.asarray(cluster_assignment)
    n_clusters = int(cluster_assignment.max()) + 1
    cluster_parts = partition_dirichlet(labels, n_clusters, alpha, rng, min_size=min_size)
    parts = [np.array([], dtype=int) for _ in range(len(cluster_assignment))]
    for c in range(n_clusters):
        members = np.where(cluster_assignment == c)[0]
        if len(members) == 0:
            continue
        shards = np.array_split(rng.permutation(cluster_parts[c]), len(members))
        for m, sh in zip(members, shards):
            parts[m] = np.sort(sh)
    return parts
