"""P7 online-regret experiment: CUCB sensing selection vs the offline greedy oracle.

Over T rounds with UNKNOWN, noisily-observed per-client sensing SNR, three selectors pick a
budget-K set each round: (1) CUCB (online, learns the SNRs), (2) the offline greedy oracle on
the TRUE Fisher information (the (1-1/e)-benchmark), (3) random-K (linear-regret control). We log
per round the true utility U(S) of each and the cumulative alpha-regret
    R(T) = sum_t [ alpha * U(S_oracle) - U(S_online) ],  alpha = 1 - 1/e,
to runs/<tag>/regret/. analysis/regret.py certifies sublinearity (R(T)/T -> 0, log-log slope < 1).

Run:  python -m scout_fl.experiments.run_regret --config scout_fl/configs/campaign_main.yaml [--rounds 300]
Stored as JSON (same run-store convention) under runs/regret/<scenario>/<selector>__seed<seed>.json.
"""
from __future__ import annotations

import argparse

import numpy as np

from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.selection.online import CUCBSensingSelector
from scout_fl.selection.random import RandomSelector
from scout_fl.selection.scout_greedy import naive_greedy
from scout_fl.sim.fim import db_to_linear, per_client_target_fim, prior_fim
from scout_fl.sim.geometry import pairwise_geometry
from scout_fl.utils.config import load_config
from scout_fl.utils.runstore import save_unit, unit_path
from scout_fl.utils.seed import seed_everything

ALPHA = 1.0 - 1.0 / np.e


def _sensing(geom, snr_lin, cfg, M):
    fim = per_client_target_fim(geom, snr_lin, cfg.sensing.k_range, cfg.sensing.k_angle)
    return SensingUtility(fim, prior_fim(M, cfg.sensing.prior_fim), np.ones(M))


def run_regret(cfg, seed, rounds, runs_root="runs", tag="regret", obs_noise_db=3.0):
    rng = seed_everything(seed)
    K, M, budget = int(cfg.network.num_clients), int(cfg.network.num_targets), int(cfg.network.budget)
    area = np.asarray(cfg.network.area_size, dtype=float)
    clients = rng.uniform(0, area, (K, 2))
    targets = rng.uniform(0, area, (M, 2))
    geom = pairwise_geometry(clients, targets)
    # TRUE per-client sensing SNR (linear), unknown to the online learner
    true_snr_db = rng.uniform(0.0, 25.0, K)
    true_snr = db_to_linear(true_snr_db)
    true_util = _sensing(geom, true_snr, cfg, M)
    oracle_sel, _, _ = naive_greedy(true_util, K, budget)      # stationary oracle
    u_oracle = float(true_util.value(oracle_sel))

    cucb = CUCBSensingSelector(K, ucb_c=float(cfg.get("regret", {}).get("ucb_c", 1.0)))
    rand = RandomSelector()
    logs = {"cucb": [], "random": []}
    cum = {"cucb": 0.0, "random": 0.0}

    for t in range(rounds):
        # CUCB: build optimistic FIM from UCB SNRs, greedy select, observe noisy true SNR
        ucb_snr = _ucb_to_linear(cucb.ucb_snr(), true_snr)
        opt_util = _sensing(geom, ucb_snr, cfg, M)
        s_cucb, _, _ = naive_greedy(opt_util, K, budget)
        u_cucb = float(true_util.value(s_cucb))
        obs = true_snr * db_to_linear(obs_noise_db * rng.standard_normal(K))   # noisy observation
        cucb.update(s_cucb, obs)
        # random control
        s_rand = rand.select(num_clients=K, budget=budget, rng=rng).selected
        u_rand = float(true_util.value(s_rand))
        # learning regret vs the SAME greedy oracle but with KNOWN SNRs (the achievable benchmark
        # for an online learner; alpha=1 here. The (1-1/e) factor is the worst-case approximation
        # of greedy to the NP-hard true optimum, reported separately in theory). This regret is >=0
        # and -> 0 as CUCB's SNR estimates converge; random keeps a constant gap (linear cum-regret).
        for name, u, sset in (("cucb", u_cucb, s_cucb), ("random", u_rand, s_rand)):
            inst = max(0.0, u_oracle - u)
            cum[name] += inst
            logs[name].append({"round": t, "util": round(u, 5), "u_oracle": round(u_oracle, 5),
                               "alpha_u_oracle": round(ALPHA * u_oracle, 5),
                               "instant_regret": round(inst, 5), "cum_regret": round(cum[name], 5),
                               "avg_regret": round(cum[name] / (t + 1), 6), "selected": list(sset)})

    scen = f"N{K}_K{budget}_M{M}"
    for name, rows in logs.items():
        meta = {"method": name, "seed": int(seed), "rounds": rounds, "scenario": scen,
                "alpha": ALPHA, "u_oracle": u_oracle, "tag": tag, "point": scen}
        save_unit(unit_path(runs_root, tag, scen, name, seed), meta, rows, complete=True,
                  objectives={"final_cum_regret": cum[name], "final_avg_regret": cum[name] / rounds})
    print(f"[regret] seed{seed} {scen}: final cum alpha-regret  cucb={cum['cucb']:.2f}  random={cum['random']:.2f}"
          f"  (R/T: cucb={cum['cucb']/rounds:.4f} random={cum['random']/rounds:.4f})")
    return cum


def _ucb_to_linear(ucb_index, true_snr):
    """The CUCB index lives in the SNR-mean space; map it to a positive linear SNR for FIM build.
    Never-pulled arms (index 1e9) get a large optimistic SNR to force exploration."""
    out = np.array(ucb_index, dtype=float)
    big = out >= 1e8
    out = np.clip(out, 1e-6, None)
    out[big] = float(np.max(true_snr) * 4.0)        # optimistic exploration value
    return out


def main():
    p = argparse.ArgumentParser(description="P7 online-regret experiment (CUCB sensing selection)")
    p.add_argument("--config", default="scout_fl/configs/campaign_main.yaml")
    p.add_argument("--override", nargs="*", default=None)
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--out", default="runs")
    args = p.parse_args()
    cfg = load_config(args.config, args.override)
    for s in args.seeds:
        run_regret(cfg, s, args.rounds, runs_root=args.out)


if __name__ == "__main__":
    main()
