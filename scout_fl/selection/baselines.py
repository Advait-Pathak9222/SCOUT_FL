"""Baseline client-selection methods for the bake-off (named per their source).

Each is implemented on the shared SCOUT-FL simulator, faithful to its paper's
selection principle, and exposed via ``BASELINE_REGISTRY``. They consume a
uniform keyword context (the per-round signals the runner provides) and ignore
the kwargs they don't need (``**_``):

  K, budget, rng, sensing, learning, g (comm gains), snr_scores (per-client
  sensing SNR), losses (per-client local loss), latency (per-client),
  P, sigma2, mse_eps.

References: FedCS (Nishio & Yonetani 2019), Oort (Lai et al. OSDI'21), FedGCS
(IJCAI'24), CollabSenseFed (multi-objective OTA-FEEL collaborative ISAC), OTA-FL
Client Selection in ISCC (Zheng et al. ICC-W'24), Sensing-Native OTA-FL,
AirComp-MSE-minimization, communication-only and sensing-only selection,
fixed-weighted-sum ISAC selection.
"""
from __future__ import annotations

import numpy as np

from scout_fl.selection.base import Selector, SelectionResult
from scout_fl.selection.scout_greedy import naive_greedy
from scout_fl.sim.aircomp import aggregation_mse, min_gain_for_mse


class CommOnlySelector(Selector):
    """Communication-only: pick the strongest comm channels (best AirComp links)."""
    name = "comm_only"

    def select(self, g, budget, **_) -> SelectionResult:
        order = np.argsort(-np.asarray(g, dtype=float))[:budget]
        return SelectionResult(selected=sorted(int(k) for k in order))


class AirCompMSEMinSelector(Selector):
    """Greedily select the set that minimizes the AirComp aggregation MSE.

    NOTE: under the channel-inversion AirComp model MSE = sigma^2/(|S|^2 P min_k g_k),
    for a FIXED cardinality this provably reduces to picking the largest-gain clients,
    i.e. it is mathematically IDENTICAL to ``comm_only``. Kept as a separately-named
    literature baseline; expect identical numbers to comm_only in the bake-off (they
    would only diverge under joint selection + power allocation / variable |S|)."""
    name = "aircomp_mse_min"

    def select(self, g, budget, P, sigma2, **_) -> SelectionResult:
        g = np.asarray(g, dtype=float)
        chosen, remaining = [], set(range(len(g)))
        while remaining and len(chosen) < budget:
            k = min(remaining, key=lambda c: aggregation_mse(g, chosen + [c], power=P, sigma2=sigma2))
            chosen.append(k); remaining.discard(k)
        return SelectionResult(selected=sorted(chosen))


class SensingOnlySelector(Selector):
    """Sensing-only (D-optimal): greedily maximize the log-det Fisher information."""
    name = "sensing_only"

    def select(self, sensing, K, budget, **_) -> SelectionResult:
        sel, _, _ = naive_greedy(sensing, K, budget)
        return SelectionResult(selected=sorted(int(k) for k in sel))


class CRBOnlySelector(Selector):
    """CRB-only (A-optimal): greedily add the client that most reduces aggregate CRB =
    trace(inv FIM). Distinct from sensing_only (D-optimal log-det): A-optimality targets
    estimation variance directly and is only weakly submodular. A geometry-aware sensing
    baseline that contrasts the two classical optimal-design criteria."""
    name = "crb_only"

    def select(self, sensing, K, budget, **_) -> SelectionResult:
        chosen, rem = [], set(range(K))
        def agg_crb(subset):
            return float((sensing.w * sensing.crb(subset)).sum())
        while rem and len(chosen) < budget:
            k = min(rem, key=lambda c: agg_crb(chosen + [c]))
            chosen.append(k); rem.discard(k)
        return SelectionResult(selected=sorted(chosen))


class FedISSelector(Selector):
    """FedIS — gradient-norm importance sampling (Rizk/Chen et al.; the canonical IS
    baseline DELTA compares against). p_i proportional to ||grad_i||, sampled without
    replacement. Distinct from DELTA (gradient DIVERSITY ||grad_i - mean grad||) and
    PO-FL (channel x gradient-importance)."""
    name = "fedis"

    def select(self, grad_norm, K, budget, rng, **_) -> SelectionResult:
        p = np.asarray(grad_norm, dtype=float) + 1e-12
        p = p / p.sum()
        sel = rng.choice(K, size=min(budget, K), replace=False, p=p)
        return SelectionResult(selected=sorted(int(k) for k in sel))


class FedCSSelector(Selector):
    """FedCS (Nishio & Yonetani 2019): resource-aware — the fastest feasible clients."""
    name = "fedcs"

    def select(self, latency, budget, **_) -> SelectionResult:
        order = np.argsort(np.asarray(latency, dtype=float))[:budget]
        return SelectionResult(selected=sorted(int(k) for k in order))


class OortSelector(Selector):
    """Oort (Lai et al. OSDI'21): statistical utility (loss) x system utility (speed),
    with exploration-exploitation."""
    name = "oort"

    def __init__(self, eps: float = 0.1, alpha: float = 0.5) -> None:
        self.eps, self.alpha = eps, alpha

    def select(self, losses, latency, budget, rng, **_) -> SelectionResult:
        losses = np.asarray(losses, dtype=float)
        lat = np.asarray(latency, dtype=float)
        thr = float(np.median(lat))
        sys = np.where(lat <= thr, 1.0, (thr / np.clip(lat, 1e-9, None)) ** self.alpha)
        util = losses * sys
        n_exploit = max(1, int(round(budget * (1.0 - self.eps))))
        exploit = list(np.argsort(-util)[:n_exploit])
        rest = [k for k in range(len(losses)) if k not in exploit]
        n_explore = min(budget - len(exploit), len(rest))
        explore = list(rng.choice(rest, size=n_explore, replace=False)) if n_explore > 0 else []
        return SelectionResult(selected=sorted(int(k) for k in exploit + explore))


class FedGCSSelector(Selector):
    """FedGCS (IJCAI'24): efficient + diverse selection — gradient-representation
    diversity (facility location) traded against execution cost (latency)."""
    name = "fedgcs"

    def select(self, learning, latency, K, budget, **_) -> SelectionResult:
        lat = np.asarray(latency, dtype=float)
        latn = lat / max(float(lat.max()), 1e-9)
        state = learning.init_state()
        chosen, remaining = [], set(range(K))
        while remaining and len(chosen) < budget:
            k = max(remaining, key=lambda c: learning.marginal_gain(state, c) - 0.5 * latn[c])
            chosen.append(k); state = learning.add(state, k); remaining.discard(k)
        return SelectionResult(selected=sorted(chosen))


class FixedWeightedSumSelector(Selector):
    """Fixed weighted-sum ISAC selection: a*learn + b*sense - c*MSE - d*latency
    with FIXED (hand-set) weights — the 'is it just a weighted score?' baseline."""
    name = "fixed_weighted"
    A, B, C, D = 1.0, 1.0, 1.0, 0.5

    def select(self, sensing, learning, g, latency, K, budget, P, sigma2, **_) -> SelectionResult:
        lat = np.asarray(latency, dtype=float)
        latn = lat / max(float(lat.max()), 1e-9)
        ss, ls = sensing.init_state(), learning.init_state()
        chosen, remaining = [], set(range(K))
        while remaining and len(chosen) < budget:
            def score(c):
                mse = aggregation_mse(g, chosen + [c], power=P, sigma2=sigma2)
                return (self.A * learning.marginal_gain(ls, c) + self.B * sensing.marginal_gain(ss, c)
                        - self.C * mse - self.D * latn[c])
            k = max(remaining, key=score)
            chosen.append(k); ss = sensing.add(ss, k); ls = learning.add(ls, k); remaining.discard(k)
        return SelectionResult(selected=sorted(chosen))


class CollabSenseFedSelector(Selector):
    """CollabSenseFed: multi-objective learning + sensing (CRB) with fixed (equal) weights."""
    name = "collabsensefed"

    def select(self, sensing, learning, K, budget, **_) -> SelectionResult:
        ss, ls = sensing.init_state(), learning.init_state()
        chosen, remaining = [], set(range(K))
        while remaining and len(chosen) < budget:
            k = max(remaining, key=lambda c: 0.5 * sensing.marginal_gain(ss, c)
                    + 0.5 * learning.marginal_gain(ls, c))
            chosen.append(k); ss = sensing.add(ss, k); ls = learning.add(ls, k); remaining.discard(k)
        return SelectionResult(selected=sorted(chosen))


class OTAFLClientSelISCCSelector(Selector):
    """OTA-FL Client Selection in ISCC (Zheng et al. ICC-W'24): AirComp-aware learning
    selection under an aggregation-MSE feasibility gate (detection/sensing requirement)."""
    name = "ota_fl_iscc"

    def select(self, learning, g, K, budget, P, sigma2, mse_eps, **_) -> SelectionResult:
        g = np.asarray(g, dtype=float)
        g_min = min_gain_for_mse(mse_eps if mse_eps else 0.05, budget, P, sigma2)
        feasible = [k for k in range(K) if g[k] >= g_min] or list(range(K))
        state = learning.init_state()
        chosen, rem = [], set(feasible)
        while rem and len(chosen) < budget:
            k = max(rem, key=lambda c: learning.marginal_gain(state, c))
            chosen.append(k); state = learning.add(state, k); rem.discard(k)
        if len(chosen) < budget:                              # relax if too few feasible
            for k in np.argsort(-g):
                if int(k) not in chosen:
                    chosen.append(int(k))
                if len(chosen) >= budget:
                    break
        return SelectionResult(selected=sorted(chosen))


class SensingNativeOTAFLSelector(Selector):
    """Sensing-Native OTA-FL: gradient signals reused for sensing (zero sensing overhead)
    -> jointly maximize learning + sensing information (no coverage/fairness/MSE coupling)."""
    name = "sensing_native"

    def select(self, sensing, learning, K, budget, **_) -> SelectionResult:
        full = list(range(K))
        rho = float(sensing.value(full)) / max(float(learning.value(full)), 1e-9)
        ss, ls = sensing.init_state(), learning.init_state()
        chosen, remaining = [], set(range(K))
        while remaining and len(chosen) < budget:
            k = max(remaining, key=lambda c: sensing.marginal_gain(ss, c)
                    + rho * learning.marginal_gain(ls, c))
            chosen.append(k); ss = sensing.add(ss, k); ls = learning.add(ls, k); remaining.discard(k)
        return SelectionResult(selected=sorted(chosen))


class OTAFedAvgSelector(Selector):
    """OTA-FedAvg (Random-K AirComp): random selection, AirComp aggregation (isolates
    the over-the-air distortion effect). Aggregation handled in the runner."""
    name = "ota_fedavg"

    def select(self, K, budget, rng, **_) -> SelectionResult:
        sel = rng.choice(K, size=min(budget, K), replace=False)
        return SelectionResult(selected=sorted(int(k) for k in sel))


class ISCCResourceSelector(Selector):
    """ISCC system selection (FedAVG-ISCC / FedSGD-ISCC): pick the clients with the best
    communication-computation throughput (channel gain / per-client latency). The two
    methods share this selection and differ only in the local-update rule (handled in
    the runner: FedAvg = E local epochs; FedSGD = a single SGD step)."""
    name = "fedavg_iscc"

    def select(self, g, latency, budget, **_) -> SelectionResult:
        score = np.asarray(g, dtype=float) / np.clip(np.asarray(latency, dtype=float), 1e-9, None)
        order = np.argsort(-score)[:budget]
        return SelectionResult(selected=sorted(int(k) for k in order))


class FedISCCSelector(Selector):
    """Fed-ISCC (Du et al. IoT-J'24): the BS jointly senses and runs OTA-FL over shared
    resources — select AirComp-MSE-feasible clients ranked by sensing SNR."""
    name = "fed_iscc"

    def select(self, g, snr_scores, K, budget, P, sigma2, mse_eps, **_) -> SelectionResult:
        g = np.asarray(g, dtype=float)
        snr = np.asarray(snr_scores, dtype=float)
        g_min = min_gain_for_mse(mse_eps if mse_eps else 0.05, budget, P, sigma2)
        feasible = [k for k in range(K) if g[k] >= g_min] or list(range(K))
        order = sorted(feasible, key=lambda k: -snr[k])[:budget]
        if len(order) < budget:                               # relax if too few feasible
            for k in np.argsort(-g):
                if int(k) not in order:
                    order.append(int(k))
                if len(order) >= budget:
                    break
        return SelectionResult(selected=sorted(int(k) for k in order))


class AsaadSelector(Selector):
    """Asaad-Wang-Tabassum (IEEE TWC 2025, arXiv 2501.06334): sensing-aware OTA-FEEL
    device scheduling by **step-wise dropping** of the least-effective device under a
    combined aggregation-MSE + target-CRB metric. Starting from all candidates, each
    iteration removes the device whose removal keeps the combined (normalized
    MSE + CRB) objective lowest, until the budget is met. The PRIMARY ISAC competitor."""
    name = "asaad"

    def select(self, sensing, g, K, budget, P, sigma2, **_) -> SelectionResult:
        S = list(range(K))
        mse_ref = max(aggregation_mse(g, S, power=P, sigma2=sigma2), 1e-12)
        crb_ref = max(float(np.sum(sensing.crb(S))), 1e-12)

        def obj(subset):
            if not subset:
                return float("inf")
            return (aggregation_mse(g, subset, power=P, sigma2=sigma2) / mse_ref
                    + float(np.sum(sensing.crb(subset))) / crb_ref)

        while len(S) > budget:                                # drop least-effective device
            drop = min(S, key=lambda k: obj([j for j in S if j != k]))
            S.remove(drop)
        return SelectionResult(selected=sorted(S))


class DivFLSelector(Selector):
    """DivFL (Balakrishnan et al., ICLR 2022): diverse client selection via submodular
    facility-location maximization over the gradient space (diversity only, no sensing/
    channel). Greedy on the learning utility = exactly DivFL's objective."""
    name = "divfl"

    def select(self, learning, K, budget, **_) -> SelectionResult:
        state = learning.init_state()
        chosen, rem = [], set(range(K))
        while rem and len(chosen) < budget:
            k = max(rem, key=lambda c: learning.marginal_gain(state, c))
            chosen.append(k); state = learning.add(state, k); rem.discard(k)
        return SelectionResult(selected=sorted(chosen))


class DELTASelector(Selector):
    """DELTA (Wang et al., NeurIPS 2023 / arXiv 2205.13925): unbiased diverse client
    IMPORTANCE SAMPLING. Per-client score = gradient DIVERSITY = ||grad_i - mean_grad||
    (local-vs-global gradient deviation, NOT facility location / NOT gradient norm);
    sample p_i proportional to sqrt(score) (with the paper's local-variance term dropped,
    as a single probe gives no per-batch variance). Sampling-without-replacement, which is
    paired with inverse-probability aggregation reweighting in the full method."""
    name = "delta"

    def select(self, embeddings, K, budget, rng, **_) -> SelectionResult:
        E = np.asarray(embeddings, dtype=float)
        div = np.linalg.norm(E - E.mean(axis=0), axis=1)       # ||grad_i - global grad||
        p = np.sqrt(div ** 2 + 1e-12)
        p = p / p.sum()
        sel = rng.choice(K, size=min(budget, K), replace=False, p=p)  # importance sampling (WOR)
        return SelectionResult(selected=sorted(int(k) for k in sel))


class POFLSelector(Selector):
    """PO-FL (Sun et al., arXiv 2305.16854): channel- and gradient-importance-aware
    PROBABILISTIC device scheduling for over-the-air FL. Per-device score (Eq. 35)
    Q_i = sqrt( (1+a)*sigma^2*Vg / (P*|h_i|^2) + (1+1/a)*||g_i||^2 ), where the channel
    term varies as 1/|h_i|^2 (WEAK channels score HIGHER, to cut their AirComp aggregation
    weight) and Vg ~ mean gradient energy. Sample without replacement with p_i ∝ Q_i (NOT
    deterministic top-K; the stochasticity underpins the unbiased estimator)."""
    name = "po_fl"

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)

    def select(self, grad_norm, g, K, budget, rng, P, sigma2, **_) -> SelectionResult:
        gn = np.asarray(grad_norm, dtype=float)
        h2 = np.clip(np.asarray(g, dtype=float), 1e-12, None)
        vg = float(np.mean(gn ** 2)) + 1e-12
        a = (1.0 + self.alpha) * float(sigma2) * vg / max(float(P), 1e-12)
        b = 1.0 + 1.0 / self.alpha
        Q = np.sqrt(a / h2 + b * gn ** 2)
        p = Q / Q.sum()
        sel = rng.choice(K, size=min(budget, K), replace=False, p=p)  # WOR sampling by p_i ∝ Q_i
        return SelectionResult(selected=sorted(int(k) for k in sel))


class FairEquityFLSelector(Selector):
    """FairEquityFL (Islam et al., arXiv 2509.20193, 2025): the 'sampling equalizer' is a
    rule-based gate, NOT a lowest-count argmin. Per round: (1) force-include clients whose
    skip-gap >= gap_max and lifetime count < nc_max, ordered by largest gap; (2) keep only
    clients past the gap_min cooldown as eligible; (3) fill remaining slots UNIFORMLY AT
    RANDOM from the eligible set. A misbehaving client (probe-loss outlier here, a
    simplification of the paper's accuracy/loss-degradation trend test) is suspended."""
    name = "fair_equity"

    def __init__(self, gap_min: int = 1, gap_max: int | None = None, nc_max: int | None = None) -> None:
        self.gap_min, self.gap_max, self.nc_max = gap_min, gap_max, nc_max

    def select(self, age, participation, losses, K, budget, rng, **_) -> SelectionResult:
        age = np.asarray(age, dtype=float)
        cnt = np.asarray(participation, dtype=float)
        loss = np.asarray(losses, dtype=float)
        gap_max = self.gap_max if self.gap_max is not None else max(2, int(round(K / max(budget, 1))))
        nc_max = self.nc_max if self.nc_max is not None else float("inf")
        suspended = {k for k in range(K) if loss[k] > loss.mean() + 2.0 * (loss.std() + 1e-9)}

        chosen = []
        for k in sorted((k for k in range(K) if age[k] >= gap_max and cnt[k] < nc_max
                         and k not in suspended), key=lambda k: -age[k]):   # forced inclusion
            if len(chosen) < budget:
                chosen.append(k)
        eligible = [k for k in range(K) if age[k] >= self.gap_min
                    and k not in chosen and k not in suspended]             # gap_min cooldown
        nr = budget - len(chosen)
        if nr > 0 and eligible:
            chosen += [int(k) for k in rng.choice(eligible, size=min(nr, len(eligible)), replace=False)]
        if len(chosen) < budget:                                            # relax if still short
            for k in np.argsort(-age):
                if int(k) not in chosen:
                    chosen.append(int(k))
                if len(chosen) >= budget:
                    break
        return SelectionResult(selected=sorted(chosen[:budget]))


class ISCCAirFEELSelector(Selector):
    """ISCC-Air-FEEL (Wen et al., arXiv 2508.15185, 2025) — selection RESTRICTION of its
    loss-degradation objective. NOTE: the paper itself optimizes resource allocation
    (batch size / sensing power / compute / AirComp power) over a FIXED device set and
    explicitly defers device scheduling to future work; this selector implements that
    deferred extension: pick the AirComp-MSE-feasible devices that minimize the
    convergence-bound penalty, i.e. highest sensing data quality (sensing SNR, = low
    inverse-sensing-noise) and channel gain (affordable AirComp / larger batch)."""
    name = "iscc_air_feel"

    def select(self, snr_scores, g, K, budget, P, sigma2, mse_eps, **_) -> SelectionResult:
        snr = np.asarray(snr_scores, dtype=float)
        h2 = np.asarray(g, dtype=float)
        g_min = min_gain_for_mse(mse_eps if mse_eps else 0.05, budget, P, sigma2)
        feasible = [k for k in range(K) if h2[k] >= g_min] or list(range(K))
        # merit = sensing data quality x affordable AirComp (channel); minimize loss-degradation penalty
        order = sorted(feasible, key=lambda k: -(snr[k] * np.sqrt(max(h2[k], 0.0))))[:budget]
        if len(order) < budget:
            for k in np.argsort(-h2):
                if int(k) not in order:
                    order.append(int(k))
                if len(order) >= budget:
                    break
        return SelectionResult(selected=sorted(int(k) for k in order))


_ISCC_RESOURCE = ISCCResourceSelector()                       # shared by FedAVG-/FedSGD-ISCC
BASELINE_REGISTRY = {
    s.name: s for s in (
        CommOnlySelector(), AirCompMSEMinSelector(), SensingOnlySelector(),
        FedCSSelector(), OortSelector(), FedGCSSelector(), FixedWeightedSumSelector(),
        CollabSenseFedSelector(), OTAFLClientSelISCCSelector(), SensingNativeOTAFLSelector(),
        OTAFedAvgSelector(), FedISCCSelector(), AsaadSelector(),
        DivFLSelector(), DELTASelector(), POFLSelector(), FairEquityFLSelector(),
        ISCCAirFEELSelector(), CRBOnlySelector(), FedISSelector(),
    )
}
BASELINE_REGISTRY["fedavg_iscc"] = _ISCC_RESOURCE
BASELINE_REGISTRY["fedsgd_iscc"] = _ISCC_RESOURCE             # same selection; FedSGD update rule in runner
