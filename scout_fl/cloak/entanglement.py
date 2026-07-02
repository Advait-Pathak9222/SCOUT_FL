"""E-C1 — entanglement kill test (design §2.6, GATE 2; analytic + small numeric).

Core question (theory T-C1): target and client information flow through the same
bistatic geometry. Can target sensing utility be extracted under a hard cap on
client-position leakage? We model each client's campaign participation as a
continuous usage a_k in [0, 1] (fraction of rounds / power). Then

  target FIM  J_t(a)     = J0_t + sum_k a_k J_target_k
  leakage FIM J_leak_k(a)= J0_c + a_k J_leak_k          (per client)
  target utility U(a)    = logdet(J_t(a)) - logdet(J0_t)
  client CRB floor r_k   = sqrt(tr(J_leak_k(a)^-1))     [meters]

For each geometry: uncapped uses a_k = 1 (sub-meter r_k); the capped policy raises
each a_k as high as possible while keeping r_k >= r_floor (default 10 m). Retained
fraction = U(capped) / U(uncapped). GATE 2 asks whether generic geometry retains
>= 50% of the unconstrained target log-det at a >= 10 m floor.
"""
from __future__ import annotations

import numpy as np

from scout_fl.sim.crb import logdet_spd
from scout_fl.sim.geometry import pairwise_geometry
from scout_fl.sim.fim import per_client_target_fim
from scout_fl.infra.leakage import client_leak_fim


def _target_util(J0_t, Jt_clients, a):
    acc = J0_t + (a[:, None, None] * Jt_clients).sum(axis=0)
    return float(logdet_spd(acc) - logdet_spd(J0_t))


def _client_crb(J0_c, Jleak_k, a_k):
    acc = J0_c + a_k * Jleak_k
    return float(np.sqrt(np.trace(np.linalg.inv(acc))))


def _max_usage_under_floor(J0_c, Jleak_k, r_floor, hi=1.0, iters=40):
    """Largest a_k in [0, hi] with client CRB floor >= r_floor (monotone: r decreases in a)."""
    if _client_crb(J0_c, Jleak_k, hi) >= r_floor:
        return hi
    if _client_crb(J0_c, Jleak_k, 0.0) < r_floor:
        return 0.0                                          # even the prior over-informs (won't happen at 100 m)
    lo, h = 0.0, hi
    for _ in range(iters):
        mid = 0.5 * (lo + h)
        if _client_crb(J0_c, Jleak_k, mid) >= r_floor:
            lo = mid
        else:
            h = mid
    return lo


def geometry_case(client_xy, target_xy, bs_xy, *, k_range=1.0, k_angle=0.05,
                  snr_target=375.0, snr_up=3750.0, prior_target=1e-3, prior_client_std=100.0,
                  r_floor=10.0):
    # snr_target / snr_up default to CAMPAIGN-CUMULATIVE effective SNR (~150 rounds at the
    # tx_power=-15 dBm operating point) so the uncapped client floor is ~sub-meter (design
    # §2.6) and the 10 m cap genuinely binds. Config-overridable via cloak.entanglement.*.
    """One (2-client, 1-target) case -> dict with uncapped/capped utility and retained frac."""
    clients = np.asarray(client_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float).reshape(1, 2)
    geom = pairwise_geometry(clients, target)
    Jt = per_client_target_fim(geom, np.full((clients.shape[0], 1), snr_target),
                               k_range, k_angle)[:, 0]       # (K, 2, 2) target FIM per client
    Jleak = snr_up * client_leak_fim(clients, np.asarray(bs_xy, float), k_range, k_angle)  # (K,2,2)
    J0_t = prior_target * np.eye(2)
    J0_c = np.eye(2) / prior_client_std ** 2

    K = clients.shape[0]
    a_full = np.ones(K)
    u_full = _target_util(J0_t, Jt, a_full)
    r_full = np.array([_client_crb(J0_c, Jleak[k], 1.0) for k in range(K)])

    a_cap = np.array([_max_usage_under_floor(J0_c, Jleak[k], r_floor) for k in range(K)])
    u_cap = _target_util(J0_t, Jt, a_cap)
    r_cap = np.array([_client_crb(J0_c, Jleak[k], a_cap[k]) for k in range(K)])

    retained = u_cap / u_full if u_full > 1e-12 else 0.0
    return {"u_full": u_full, "u_cap": u_cap, "retained": float(retained),
            "r_full_min": float(r_full.min()), "r_cap_min": float(r_cap.min()),
            "a_cap": a_cap.tolist()}


def bearing_sweep(n_bearings=37, target=(50.0, 50.0), bs=(50.0, 50.0), radius=20.0,
                  range_ratio=1.0, **kw):
    """Sweep the relative bearing between two clients (as seen from the target)."""
    tgt = np.asarray(target, float)
    out = []
    for deg in np.linspace(0.0, 180.0, n_bearings):
        th = np.deg2rad(deg)
        c1 = tgt + radius * np.array([1.0, 0.0])
        c2 = tgt + radius * range_ratio * np.array([np.cos(th), np.sin(th)])
        r = geometry_case([c1, c2], tgt, bs, **kw)
        r["bearing_deg"] = float(deg)
        out.append(r)
    return out


def monte_carlo(n=100, arena=100.0, seed=0, **kw):
    """N random 2-client/1-target layouts at arena scale -> per-case + summary."""
    rng = np.random.default_rng(int(seed))
    bs = np.array([arena / 2, arena / 2])
    cases = []
    for _ in range(n):
        tgt = rng.uniform(0.2 * arena, 0.8 * arena, size=2)
        c = rng.uniform(0.0, arena, size=(2, 2))
        cases.append(geometry_case(c, tgt, bs, **kw))
    retained = np.array([c["retained"] for c in cases])
    return {"cases": cases,
            "retained_median": float(np.median(retained)),
            "retained_mean": float(retained.mean()),
            "retained_q25": float(np.quantile(retained, 0.25)),
            "retained_q75": float(np.quantile(retained, 0.75)),
            "n": int(n)}


def gate2(mc_summary, threshold=0.50):
    """GATE 2 verdict (design §2.6): generic geometry retains >= 50% target log-det."""
    val = mc_summary["retained_median"]
    return {"gate": "GATE2_entanglement",
            "criterion": "median retained target log-det >= 0.50 at >=10 m client CRB floor",
            "measured": float(val), "threshold": float(threshold),
            "pass": bool(val >= threshold),
            "framing": "constructive" if val >= threshold else "impossibility (T-C1 fallback)"}


def run_ec1(out_dir, r_floor=10.0, n_mc=100, seed=0):
    """Full E-C1: bearing sweep + Monte-Carlo + GATE 2 -> JSON/CSV in out_dir."""
    import csv
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sweep = bearing_sweep(r_floor=r_floor)
    mc = monte_carlo(n=n_mc, seed=seed, r_floor=r_floor)
    verdict = gate2(mc)

    with (out / "ec1_bearing_sweep.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bearing_deg", "u_full", "u_cap", "retained", "r_full_min", "r_cap_min"])
        for r in sweep:
            w.writerow([r["bearing_deg"], r["u_full"], r["u_cap"], r["retained"],
                        r["r_full_min"], r["r_cap_min"]])
    (out / "ec1_summary.json").write_text(json.dumps(
        {"r_floor_m": r_floor, "monte_carlo": {k: v for k, v in mc.items() if k != "cases"},
         "gate2": verdict}, indent=2))
    print(f"[E-C1] retained median={mc['retained_median']:.3f} -> GATE2 pass={verdict['pass']} "
          f"({verdict['framing']})")
    return verdict
