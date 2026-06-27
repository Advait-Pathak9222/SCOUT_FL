"""Download/verify campaign datasets before launching long HPC runs.

This is intentionally a standalone preflight so cluster job logs show dataset
progress before the expensive experiment loops begin.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from scout_fl.fl.datasets import load_fl_dataset
from scout_fl.utils.config import load_config

_DEFAULT_CONFIGS = (
    "scout_fl/configs/ablation.yaml",
    "scout_fl/configs/ablation_vismaya.yaml",
    "scout_fl/configs/campaign_main.yaml",
)
_CAMPAIGN_SWEEP_DATASETS = ("fashion_mnist", "cifar10", "cifar100", "emnist", "uci_har")
_SMOKE_DATASETS = ("mnist", "fashion_mnist", "cifar10")


def _collect_from_configs(configs: list[str], overrides: list[str] | None) -> tuple[str, list[str]]:
    root, datasets = "data", []
    for cfg_path in configs:
        cfg = load_config(cfg_path, overrides)
        root = str(cfg.fl.get("data_root", root))
        datasets.append(str(cfg.fl.dataset).lower())
    datasets.extend(_CAMPAIGN_SWEEP_DATASETS)
    return root, sorted(set(datasets))


def _size(path: Path) -> str:
    if not path.exists():
        return "missing"
    total = 0
    if path.is_file():
        total = path.stat().st_size
    else:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    units = ("B", "KB", "MB", "GB")
    val = float(total)
    for unit in units:
        if val < 1024.0 or unit == units[-1]:
            return f"{val:.1f}{unit}"
        val /= 1024.0
    return f"{total}B"


def prepare(datasets: list[str], root: str, download: bool = True) -> None:
    print(f"[data] root={root} download={download} datasets={datasets}", flush=True)
    Path(root).mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(datasets, 1):
        tic = time.perf_counter()
        print(f"[data] ({i}/{len(datasets)}) preparing {name} ...", flush=True)
        ds = load_fl_dataset(name, root=root, download=download)
        dt = time.perf_counter() - tic
        print(
            f"[data] ({i}/{len(datasets)}) ready {name}: "
            f"train={len(ds.y_train)} test={len(ds.y_test)} classes={ds.num_classes} "
            f"shape={ds.input_shape} root_size={_size(Path(root))} elapsed={dt:.1f}s",
            flush=True,
        )
    print("[data] dataset preflight complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/verify SCOUT-FL datasets")
    parser.add_argument("--config", nargs="*", default=list(_DEFAULT_CONFIGS),
                        help="configs whose fl.dataset/fl.data_root should be prepared")
    parser.add_argument("--override", nargs="*", default=None,
                        help="same key=value overrides used by the experiment scripts")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="explicit dataset list; overrides config-derived list")
    parser.add_argument("--root", default=None, help="data root; defaults to config fl.data_root")
    parser.add_argument("--quick", action="store_true",
                        help="prepare only the small smoke-run dataset set")
    parser.add_argument("--no-download", action="store_true",
                        help="verify local data only; do not download missing torchvision datasets")
    args = parser.parse_args()

    root, datasets = _collect_from_configs(args.config, args.override)
    if args.quick:
        datasets = list(_SMOKE_DATASETS)
    if args.datasets:
        datasets = sorted(set(d.lower() for d in args.datasets))
    if args.root:
        root = args.root
    prepare(datasets, root=root, download=not args.no_download)


if __name__ == "__main__":
    main()
