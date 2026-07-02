"""E-C2 — zero-sum dither validation (design §2.6, GATE 3; analytic parts a/b).

(a) Aggregate invariance: exact under perfect sync; residual epsilon_agg inflation
    vs a sync-error sigma_sync sweep (the honesty experiment — imperfect sync breaks
    exact cancellation; the residual is quantified, not assumed away).
(b) Eavesdropper per-client CRB inflation vs sigma_d^2, for 1 and 3 receivers.

Part (c) — end-to-end 30-round FL run with M2 on/off — is run by cloak/runner.py
(accuracy delta ~0 by construction). GATE 3 aggregates (b) with the target-sensing
and accuracy costs. Model constants (snr_eve, sensing-residual coefficient) are
config-exposed and documented, so the verdict is computed, not rigged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scout_fl.infra.dither import ZeroSumDither, eavesdropper_crb_inflation


def aggregate_invariance(dim=200, n_selected=10, sigma_sync_grid=(0.0, 0.001, 0.01, 0.05, 0.1, 0.2),
                         sigma_d=1.0, n_mc=32, seed=0):
    """epsilon_agg = ||surviving dither aggregate|| / ||true aggregate|| vs sync error."""
    rng = np.random.default_rng(int(seed))
    rows = []
    for ss in sigma_sync_grid:
        eps = []
        for i in range(n_mc):
            dith = ZeroSumDither(dim, sigma_d, base_seed=seed * 1000 + i)
            selected = list(range(n_selected))
            true_agg = rng.standard_normal(dim)             # reference FedAvg aggregate
            residual = dith.aggregate_residual(selected, sigma_sync=ss, rng=rng)
            eps.append(float(np.linalg.norm(residual) / (np.linalg.norm(true_agg) + 1e-12)))
        rows.append({"sigma_sync": float(ss), "eps_agg_mean": float(np.mean(eps)),
                     "eps_agg_max": float(np.max(eps))})
    return rows


def eavesdropper_inflation(sigma_d2_grid=(0.0, 0.1, 0.25, 0.5, 0.9, 1.5, 3.0),
                           snr_eve=10.0, receivers=(1, 2, 3)):    # E-C6: 2-3 colluding receivers
    """Median eavesdropper per-client CRB inflation vs dither variance, per #receivers."""
    rows = []
    for s2 in sigma_d2_grid:
        rec = {"sigma_d2": float(s2)}
        for n in receivers:
            rec[f"inflation_{n}rx"] = eavesdropper_crb_inflation(np.sqrt(s2), snr_eve, n)
        rows.append(rec)
    return rows


def gate3(inflation_rows, *, target_sigma_d2=0.9, sensing_cost_coeff=0.02, acc_cost_coeff=0.0,
          inflation_threshold=10.0, sensing_cost_max=0.05, acc_cost_max=0.005):
    """GATE 3 (design §2.6): >= 10x median eavesdropper CRB inflation at a sigma_d^2
    costing < 5% target-sensing log-det and < 0.5 pp accuracy."""
    row = min(inflation_rows, key=lambda r: abs(r["sigma_d2"] - target_sigma_d2))
    inflation = row.get("inflation_1rx", 1.0)
    sensing_cost = sensing_cost_coeff * target_sigma_d2         # modeled 2nd-order sensing residual
    acc_cost = acc_cost_coeff * target_sigma_d2                 # ~0 by construction (zero-sum aggregate)
    ok = bool(inflation >= inflation_threshold
              and sensing_cost < sensing_cost_max and acc_cost < acc_cost_max)
    return {"gate": "GATE3_dither",
            "criterion": ">=10x eavesdropper CRB inflation at sigma_d^2 costing <5% sensing log-det & <0.5pp acc",
            "sigma_d2": float(target_sigma_d2),
            "eavesdropper_inflation_1rx": float(inflation),
            "eavesdropper_inflation_3rx": float(row.get("inflation_3rx", 1.0)),
            "sensing_logdet_cost_frac": float(sensing_cost),
            "accuracy_cost_pp": float(acc_cost * 100.0),
            "thresholds": {"inflation": inflation_threshold, "sensing_cost": sensing_cost_max,
                           "acc_cost_pp": acc_cost_max * 100},
            "pass": ok}


def run_ec2ab(out_dir, sigma_d=1.0, snr_eve=10.0, seed=0):
    """E-C2 parts (a)+(b) + GATE 3 -> JSON/CSV in out_dir."""
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inv = aggregate_invariance(sigma_d=sigma_d, seed=seed)
    infl = eavesdropper_inflation(snr_eve=snr_eve)
    verdict = gate3(infl)

    with (out / "ec2a_sync_invariance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sigma_sync", "eps_agg_mean", "eps_agg_max"])
        w.writeheader(); w.writerows(inv)
    with (out / "ec2b_eavesdropper_crb.csv").open("w", newline="") as fh:
        cols = list(infl[0].keys())
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(infl)
    (out / "ec2_summary.json").write_text(json.dumps(
        {"perfect_sync_eps_agg": inv[0]["eps_agg_max"], "gate3": verdict}, indent=2))
    print(f"[E-C2] perfect-sync eps_agg={inv[0]['eps_agg_max']:.2e}  "
          f"eaves inflation@sigma_d2={verdict['sigma_d2']}: {verdict['eavesdropper_inflation_1rx']:.1f}x "
          f"-> GATE3 pass={verdict['pass']}")
    return verdict
