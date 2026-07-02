"""RECA-FL adapter reuse and cross-regime generalization experiment.

This is a lightweight mechanism-proof simulator: it does not train a neural FL
model, but it exercises the RECA world model, appraisal, adapter bank, matcher,
and selector with a controlled A -> normal -> A' regime schedule.
"""
from __future__ import annotations

import argparse

import numpy as np

from scout_fl.fl.adapters import ContextAdapterBank, MatchResult, RegimeSignature
from scout_fl.objectives.reca_appraisal import RECAAppraisal
from scout_fl.objectives.world_model import WorldModel
from scout_fl.selection.random import RandomSelector
from scout_fl.selection.reca_selector import RECASelector
from scout_fl.utils.config import load_config, to_plain
from scout_fl.utils.logging_utils import RunLogger
from scout_fl.utils.seed import seed_everything


_METHODS = ["reca", "reca_no_memory", "reca_score_only", "wrong_reuse", "random_reuse", "oracle_reuse"]
_EXTERNAL_BASELINES = {
    "random", "fedavg", "fedprox", "fedcs", "oort", "fedcor", "loss", "resource_aware",
    "ota_fedavg", "aircomp_mse_min", "comm_only", "ota_fl_iscc", "fed_iscc",
    "iscc_air_feel", "asaad", "sensing_native", "collabsensefed",
}


def _quick(cfg) -> None:
    cfg.rounds = 40
    cfg.first_shift_round = 10
    cfg.first_shift_end = 18
    cfg.second_shift_round = 28
    cfg.network.num_clients = 30
    cfg.network.budget = 5


def _regime(t: int, cfg) -> str:
    if int(cfg.first_shift_round) <= t < int(cfg.first_shift_end):
        return "A"
    if int(cfg.second_shift_round) <= t:
        return "Aprime"
    return "normal"


def _signature_vector(regime: str, dim: int) -> np.ndarray:
    v = np.zeros(dim)
    if regime == "A":
        v[: dim // 2] = 1.0
        v[dim // 2:] = 0.35
    elif regime == "Aprime":
        v[: dim // 2] = 0.9
        v[dim // 2:] = 0.4
    elif regime == "B":
        v[: dim // 2] = -0.8
        v[dim // 2:] = 1.1
    return v


def _make_signature(residuals, regime: str) -> RegimeSignature:
    r = np.atleast_2d(np.asarray(residuals, dtype=float))
    mean = r.mean(axis=0)
    var = r.var(axis=0) + 1e-6
    emb = _signature_vector(regime, mean.size) + 0.1 * mean
    half = max(1, mean.size // 2)
    return RegimeSignature(embedding=emb, grad_residual=mean[:half], sensing_residual=mean[half:],
                           residual_mean=mean, residual_var=var, n=int(r.shape[0]))


def _method_run(method: str, cfg, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    K = int(cfg.network.num_clients)
    budget = int(cfg.network.budget)
    rounds = int(cfg.rounds)
    feat_dim = int(cfg.reca.feature_dim)
    out_dim = int(cfg.reca.output_dim)
    tau_reuse = float(cfg.reca.tau_reuse)
    trigger_tau = float(cfg.reca.trigger_tau)

    base_features = rng.normal(size=(K, feat_dim))
    client_quality = rng.uniform(0.5, 1.5, size=K)
    world = WorldModel(feature_dim=feat_dim, output_dim=out_dim, l2=1.0)
    bank = ContextAdapterBank(tau_reuse=tau_reuse, consolidate_after=2)
    selector = RECASelector(RECAAppraisal(tau_trigger=trigger_tau))
    random = RandomSelector()
    rows, performance = [], 0.55
    recovered_round = None

    for t in range(rounds):
        regime = _regime(t, cfg)
        sig = _signature_vector(regime, out_dim)
        affected = np.arange(K) < max(budget * 2, K // 3)
        features = base_features + 0.05 * rng.normal(size=(K, feat_dim))
        targets = np.zeros((K, out_dim))
        targets += client_quality[:, None] * 0.1
        targets[affected] += sig
        targets += 0.02 * rng.normal(size=(K, out_dim))
        pred = world.predict(features)
        residual = targets - pred

        mismatch = np.linalg.norm(residual, axis=1)
        risk = np.maximum(0.0, mismatch - 0.15) + 0.05 * rng.random(K)
        progress = client_quality + 0.4 * mismatch - 0.2 * risk
        risk_sel, mismatch_sel, progress_sel = risk.copy(), mismatch.copy(), progress.copy()
        if method == "reca_no_risk":
            risk_sel = np.zeros_like(risk_sel)
        elif method == "reca_no_mismatch":
            mismatch_sel = np.zeros_like(mismatch_sel)
        elif method == "reca_no_progress":
            progress_sel = np.zeros_like(progress_sel)
        elif method == "reca_mean_risk":
            risk_sel = np.full_like(risk_sel, float(np.mean(risk_sel)))

        if method.startswith("reca") or method in ("wrong_reuse", "oracle_reuse"):
            res = selector.select(risk=risk_sel, mismatch=mismatch_sel, progress=progress_sel, budget=budget)
            selected = res.selected
            diag = dict(res.info)
        elif method == "random_reuse":
            res = random.select(num_clients=K, budget=budget, rng=rng)
            selected = res.selected
            diag = {}
        elif method in _EXTERNAL_BASELINES:
            selected = _external_select(method, risk, mismatch, progress, client_quality, budget, rng)
            diag = {"external_baseline": method}
        else:
            raise ValueError(method)

        current_sig = _make_signature(residual[selected], regime)
        adapter_id, adapter_state = None, "none"
        match = bank.match(current_sig)
        reused = False
        false_reuse = False
        wrong_reuse = False
        adapter_allowed = (
            method.startswith("reca")
            and method not in ("reca_score_only", "reca_no_adapter")
        ) or method in ("wrong_reuse", "random_reuse", "oracle_reuse")
        if regime != "normal" and adapter_allowed:
            if method == "reca_no_memory":
                adapter = bank.spawn(current_sig)
                match = MatchResult(None, 0.0, float("inf"), 0.0, float("inf"))
            elif method == "reca_random_trigger" and rng.random() < 0.5:
                adapter, match, reused = bank.route_or_spawn(current_sig)
            elif method == "reca_periodic_trigger" and t % max(2, int(cfg.get("trigger_period", 5))) == 0:
                adapter, match, reused = bank.route_or_spawn(current_sig)
            elif method == "reca_oracle_trigger":
                adapter, match, reused = bank.route_or_spawn(current_sig)
            elif method in ("reca_random_trigger", "reca_periodic_trigger"):
                adapter = None
            elif method == "oracle_reuse" and regime == "Aprime":
                if bank.adapters:
                    adapter = bank.adapters[0]
                    reused = True
                else:
                    adapter = bank.spawn(current_sig)
            elif method == "wrong_reuse" and regime == "Aprime":
                wrong_sig = _make_signature(np.tile(_signature_vector("B", out_dim), (budget, 1)), "B")
                if not bank.adapters:
                    bank.spawn(wrong_sig).state = "consolidated"
                adapter = bank.adapters[0]
                reused = True
                wrong_reuse = True
            elif method == "random_reuse":
                if bank.adapters:
                    adapter = bank.adapters[int(rng.integers(0, len(bank.adapters)))]
                    reused = True
                else:
                    adapter = bank.spawn(current_sig)
            else:
                adapter, match, reused = bank.route_or_spawn(current_sig)
            if adapter is not None:
                adapter_id, adapter_state = adapter.adapter_id, adapter.state
        if reused and regime == "normal":
            false_reuse = True

        # Simulated outcome: matching reuse accelerates recovery; wrong reuse hurts.
        base_gain = 0.005 + 0.002 * float(np.mean(progress[selected]))
        if regime != "normal" and reused and not wrong_reuse:
            base_gain += 0.04
        if wrong_reuse:
            base_gain -= 0.04
        if regime != "normal" and not reused:
            base_gain -= 0.015
        if regime != "normal" and adapter_id is not None and not reused and not wrong_reuse:
            # A newly spawned adapter should earn consolidation evidence when
            # the regime is useful and bounded-risk, even before memory reuse.
            base_gain += 0.018
        performance = float(np.clip(performance + base_gain, 0.0, 0.95))
        wrong_penalty = 0.08 if wrong_reuse else 0.0
        crb = float(max(0.02, 0.45 - performance + 0.08 * (regime != "normal")
                        - 0.05 * (reused and not wrong_reuse) + wrong_penalty))
        world.update(features[selected], targets[selected])
        if adapter_id is not None:
            bank.update_evidence(adapter_id, base_gain)
            adapter_state = bank.get(adapter_id).state
            if method == "reca_no_memory":
                bank.adapters.clear()
        cal = world.calibration_report(window=20)
        if regime == "Aprime" and recovered_round is None and performance >= 0.62:
            recovered_round = t

        row = {
            "round": t,
            "method": method,
            "regime": regime,
            "selected": list(selected),
            "test_acc_proxy": round(performance, 5),
            "crb_proxy": round(crb, 5),
            "world_rmse": round(cal.rmse, 6),
            "world_ece": round(cal.ece, 6),
            "adapter_id": adapter_id,
            "adapter_state": adapter_state,
            "adapter_reused": bool(reused),
            "adapter_match_confidence": round(float(match.confidence), 6),
            "adapter_embedding_distance": round(float(match.embedding_distance), 6) if np.isfinite(match.embedding_distance) else None,
            "adapter_residual_cosine": round(float(match.residual_cosine), 6),
            "adapter_residual_kl": round(float(match.residual_kl), 6) if np.isfinite(match.residual_kl) else None,
            "false_reuse": bool(false_reuse),
            "wrong_reuse": bool(wrong_reuse),
            "wrong_reuse_penalty_proxy": round(float(wrong_penalty), 6),
            "second_shift_recovered_round": recovered_round,
        }
        row.update({k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v for k, v in diag.items()})
        rows.append(row)
    return rows


def main(default_config: str = "scout_fl/configs/reca_twc_reuse.yaml") -> None:
    parser = argparse.ArgumentParser(description="RECA adapter reuse mechanism experiment")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--override", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    if args.quick:
        _quick(cfg)
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    experiment = str(cfg.get("experiment", "reca_reuse"))
    methods = list(cfg.selection.get("methods", _METHODS))
    logger = RunLogger(cfg.get("output_dir", "outputs"), experiment, seed, to_plain(cfg))
    all_rows = []
    for i, method in enumerate(methods):
        print(f"[{experiment}] method={method}", flush=True)
        all_rows.extend(_method_run(method, cfg, seed + i))
    logger.save_csv("metrics.csv", all_rows)
    summary = _summary(all_rows, cfg)
    logger.save_csv("summary.csv", summary)
    logger.save_json("summary.json", summary)
    print(f"[{experiment}] wrote {len(all_rows)} rows -> {logger.dir}")


def _external_select(method: str, risk, mismatch, progress, quality, budget: int, rng) -> list[int]:
    K = len(quality)
    if method in ("random", "fedavg", "fedprox", "ota_fedavg"):
        return sorted(int(k) for k in rng.choice(K, size=min(budget, K), replace=False))
    if method in ("loss", "oort", "fedcor"):
        score = progress + 0.2 * mismatch
    elif method in ("fedcs", "resource_aware", "comm_only", "aircomp_mse_min"):
        score = quality - 0.2 * risk
    elif method in ("ota_fl_iscc", "fed_iscc", "iscc_air_feel", "asaad", "sensing_native", "collabsensefed"):
        score = quality + 0.3 * progress - 0.3 * risk
    else:
        score = progress
    return sorted(int(k) for k in np.argsort(-np.asarray(score, dtype=float))[:budget])


def _summary(rows: list[dict], cfg) -> list[dict]:
    out = []
    second = int(cfg.second_shift_round)
    by_method = sorted({r["method"] for r in rows})
    for m in by_method:
        rs = [r for r in rows if r["method"] == m]
        second_rows = [r for r in rs if r["round"] >= second]
        reuse = [r for r in second_rows if r["adapter_reused"]]
        wrong = [r for r in second_rows if r["wrong_reuse"]]
        rec = [r["second_shift_recovered_round"] for r in rs if r["second_shift_recovered_round"] is not None]
        out.append({
            "method": m,
            "second_shift_recovery_time": (min(rec) - second) if rec else None,
            "adapter_reuse_rate": round(len(reuse) / max(len(second_rows), 1), 5),
            "adapter_match_confidence": round(float(np.mean([r["adapter_match_confidence"] for r in second_rows])), 5),
            "false_reuse_rate": round(float(np.mean([r["false_reuse"] for r in rs])), 5),
            "wrong_reuse_penalty": round(float(np.mean([r["wrong_reuse_penalty_proxy"] for r in second_rows])), 5),
            "final_acc_proxy": rs[-1]["test_acc_proxy"],
            "final_crb_proxy": rs[-1]["crb_proxy"],
        })
    return out


if __name__ == "__main__":
    main()
