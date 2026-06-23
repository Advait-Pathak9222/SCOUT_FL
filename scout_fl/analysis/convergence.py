"""P6 convergence validation — experimentally confirm the per-round descent bound

    E[F(w_t) - F(w_{t+1})]  ~  (eta/2)||grad F||^2  -  (L eta^2/2)( sigma^2/|S| + eps_agg ).

From the resumable run store we form, per round, the realized loss decrease
Delta_t = test_loss[t-1] - test_loss[t] and regress it on the two logged drivers:

    Delta_t = b0 + b1 * grad_sq_t + b2 * agg_mse_t + noise,

where grad_sq = ||aggregated update||^2 (descent driver, proxy for ||grad F||^2) and
agg_mse = AirComp aggregation MSE. The bound predicts **b1 > 0** (more gradient signal =>
more descent) and **b2 < 0** (more AirComp distortion => less descent) — the headline
wireless-learning claim. We report coefficients, 95% CIs, p-values, R^2, and a verdict.

CLI:  python -m scout_fl.analysis.convergence [runs_root] [--tag campaign_main]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from scout_fl.analysis.collect import _iter_units


def _per_round_descent(runs_root: Path, tag):
    """Collect (delta_loss, grad_sq, agg_mse) triples across all rounds/units/seeds."""
    rows = []
    for _tag, _point, d in _iter_units(runs_root, tag):
        rnds = d.get("rounds", [])
        for a, b in zip(rnds[:-1], rnds[1:]):
            if all(k in b for k in ("test_loss", "grad_sq", "agg_mse")) and "test_loss" in a:
                rows.append((a["test_loss"] - b["test_loss"],     # Delta_t (descent)
                             b["grad_sq"], b["agg_mse"], b.get("method", "?")))
    return rows


def _ols(y, X):
    """OLS with t-stats: returns (beta, se, t, p, r2). X already includes intercept col."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - k, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tval = beta / np.where(se > 0, se, np.nan)
    p = 2 * stats.t.sf(np.abs(tval), dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, tval, p, r2


def validate(runs_root="runs", tag=None) -> dict:
    rows = _per_round_descent(Path(runs_root), tag)
    if len(rows) < 8:
        return {"ok": False, "note": f"only {len(rows)} round-transitions; run more rounds/seeds"}
    delta = np.array([r[0] for r in rows])
    gsq = np.array([r[1] for r in rows])
    mse = np.array([r[2] for r in rows])
    # standardize predictors so coefficients are comparable; intercept + grad_sq + agg_mse
    def z(a): s = a.std(); return (a - a.mean()) / s if s > 1e-12 else a * 0.0
    X = np.column_stack([np.ones_like(delta), z(gsq), z(mse)])
    beta, se, tval, p, r2 = _ols(delta, X)
    res = {
        "ok": True, "n_transitions": len(rows),
        "beta_grad_sq": float(beta[1]), "p_grad_sq": float(p[1]),
        "beta_agg_mse": float(beta[2]), "p_agg_mse": float(p[2]),
        "r2": float(r2),
        "grad_sq_sign_ok": bool(beta[1] > 0),                 # bound predicts > 0
        "agg_mse_sign_ok": bool(beta[2] < 0),                 # bound predicts < 0 (headline)
    }
    res["bound_supported"] = bool(res["grad_sq_sign_ok"] and res["agg_mse_sign_ok"])
    return res


def _format(res: dict) -> str:
    if not res.get("ok"):
        return f"[P6] {res.get('note')}"
    out = ["=== P6 convergence-bound validation (per-round descent regression) ===",
           f"  transitions: {res['n_transitions']}   R^2 = {res['r2']:.3f}",
           f"  beta(grad_sq) = {res['beta_grad_sq']:+.4e}  p={res['p_grad_sq']:.3g}  "
           f"(predicted > 0: {'YES' if res['grad_sq_sign_ok'] else 'NO'})",
           f"  beta(agg_mse) = {res['beta_agg_mse']:+.4e}  p={res['p_agg_mse']:.3g}  "
           f"(predicted < 0: {'YES' if res['agg_mse_sign_ok'] else 'NO'})",
           f"  => descent bound experimentally supported: {'YES' if res['bound_supported'] else 'NO'}",
           "  (more gradient signal -> more descent; more AirComp MSE -> less descent.)"]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="P6 convergence-bound validation from runs/")
    p.add_argument("runs_root", nargs="?", default="runs")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    res = validate(args.runs_root, args.tag)
    print(_format(res))


if __name__ == "__main__":
    main()
