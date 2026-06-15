"""Run management: per-run output directory + CSV/JSON logging.

Each experiment run gets ``outputs/<experiment>/<run_id>/`` containing the
resolved config, the seed, per-round metrics, and any plots. This keeps every
experiment self-describing and reproducible.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


class RunLogger:
    """Creates a timestamped run directory and writes config/metrics/plots."""

    def __init__(self, base_dir: str | Path, experiment: str, seed: int,
                 config: Mapping[str, Any] | None = None) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment}_seed{seed}_{stamp}"
        self.dir = Path(base_dir) / experiment / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "plots").mkdir(exist_ok=True)
        self.seed = seed
        self.experiment = experiment
        if config is not None:
            self.save_json("config.json", config)
        self._rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ paths
    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    # ----------------------------------------------------------------- writers
    def save_json(self, name: str, obj: Any) -> Path:
        target = self.dir / name
        with target.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
        return target

    def log_row(self, row: Mapping[str, Any]) -> None:
        self._rows.append(dict(row))

    def save_csv(self, name: str = "metrics.csv",
                 rows: Iterable[Mapping[str, Any]] | None = None) -> Path | None:
        data = list(rows) if rows is not None else self._rows
        if not data:
            return None
        fieldnames = sorted({key for row in data for key in row})
        target = self.dir / name
        with target.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return target
