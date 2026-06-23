"""Statistical module (Test E): summary stats, paired tests, Friedman + Nemenyi.

Run:  pytest scout_fl/tests/test_stats.py -q
"""
from __future__ import annotations

import math

import numpy as np

from scout_fl.analysis.stats import (average_ranks, friedman, nemenyi,
                                      paired_ttest, statistical_report,
                                      summarize, wilcoxon_test)


def test_summarize_ci_contains_mean():
    s = summarize([0.80, 0.82, 0.78, 0.81, 0.79])
    assert s["ci_low"] < s["mean"] < s["ci_high"]
    assert s["n"] == 5 and s["std"] > 0


def test_paired_ttest_detects_clear_shift():
    a = [0.90, 0.91, 0.89, 0.92, 0.90]
    b = [0.80, 0.81, 0.79, 0.82, 0.80]               # uniformly ~0.10 lower
    assert paired_ttest(a, b)["p"] < 0.05
    assert math.isnan(paired_ttest(a, a)["p"])       # identical -> nan, not a crash


def test_wilcoxon_runs_and_flags_difference():
    a = [0.90, 0.91, 0.89, 0.92, 0.90, 0.93]
    b = [0.80, 0.81, 0.79, 0.82, 0.80, 0.78]
    assert wilcoxon_test(a, b)["p"] < 0.05


def test_friedman_and_nemenyi_rank_best_method_first():
    # three methods, 6 seeds; A clearly best, C clearly worst (higher better)
    mv = {"A": [0.90, 0.91, 0.92, 0.90, 0.93, 0.91],
          "B": [0.80, 0.81, 0.82, 0.80, 0.83, 0.81],
          "C": [0.70, 0.71, 0.72, 0.70, 0.73, 0.71]}
    assert friedman(mv)["p"] < 0.05
    ranks = average_ranks(mv, direction=1)
    assert ranks["A"] < ranks["B"] < ranks["C"]      # rank 1 = best
    nm = nemenyi(mv, direction=1)
    assert nm["critical_difference"] > 0
    assert nm["significant"]["A vs C"] is True


def test_full_report_structure():
    rng = np.random.default_rng(0)
    per_seed = {
        "jedi": [{"acc": 0.88 + 0.01 * rng.standard_normal(), "crb": 0.13 + 0.01 * rng.standard_normal()}
                   for _ in range(5)],
        "random": [{"acc": 0.80 + 0.01 * rng.standard_normal(), "crb": 0.30 + 0.01 * rng.standard_normal()}
                   for _ in range(5)],
        "scout_v2": [{"acc": 0.85 + 0.01 * rng.standard_normal(), "crb": 0.18 + 0.01 * rng.standard_normal()}
                     for _ in range(5)],
    }
    rep = statistical_report(per_seed, metrics=["acc", "crb"], reference="jedi")
    assert rep["reference"] == "jedi"
    assert "acc" in rep["metrics"] and "crb" in rep["metrics"]
    assert "random" in rep["metrics"]["acc"]["vs_reference"]
    assert "jedi" not in rep["metrics"]["acc"]["vs_reference"]    # reference excluded
    assert rep["metrics"]["crb"]["direction"] == -1
