"""Closed-loop TEMPO controllers (design §1.3): Threshold, DPP, MPC.

Each consumes a per-round ``ControlContext`` (round index, horizon, current tracker
position-covariance traces tr(P_{t,m}), smoothed learning energy L_t, constraint
level P_max) and returns a ``Decision``: a learning weight and a per-target sensing
weight vector fed to the inner selector's mixed utility. Stationary policies are the
degenerate special case (constant weights).

Scale convention: the runner builds the mixed utility from PER-TERM-NORMALIZED
learning/sensing values (each divided by its full-set value), so a scalar lambda_s
in [0,1] is meaningful and the DPP queue weights Z_m/V are dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ControlContext:
    t: int
    T: int
    trP: np.ndarray            # (M,) current tr(P_{t,m}) from the tracker (predict step)
    L_t: float                 # smoothed ||grad||^2 learning energy (EMA)
    p_max: float               # per-target constraint level tr(P) <= p_max
    M: int
    inj_hat: float = 1.0       # estimated per-round tr(P) reduction from a sensing-weighted round
    q_growth: float = 0.0      # per-round tr(P) growth from mobility (~2*sigma_p^2 * dt^2)


@dataclass
class Decision:
    w_learn: float
    w_sense: np.ndarray        # (M,)
    lam: float                 # scalar sensing fraction (diagnostic; NaN for pure-DPP)
    diag: dict = field(default_factory=dict)


class Controller:
    name = "controller"

    def decide(self, ctx: ControlContext) -> Decision:  # pragma: no cover - interface
        raise NotImplementedError

    def observe(self, ctx: ControlContext) -> None:
        """Optional post-round state update (e.g. DPP virtual queues)."""


# --------------------------------------------------------------------------- #
class ThresholdController(Controller):
    """TEMPO-Threshold (§1.3.1). Pure learning until a switch time, then sensing.

    Two modes:
      * fixed ``tau``  -> lambda_s = 0 for t < tau, else lam_high (this also serves
        the E-T1 oracle 'learn_then_sense' schedules and T1's closed-form tau*);
      * adaptive       -> online threshold: keep learning while the sensing rounds
        still needed to meet the terminal constraint fit in the rounds remaining,
        i.e. switch when  rounds_needed(P) >= T - t. rounds_needed is estimated from
        the current worst-target trace and the injection estimate inj_hat. This is
        the online realization of the exchange-argument threshold structure.
    """
    name = "tempo_threshold"

    def __init__(self, tau: int | None = None, lam_high: float = 1.0,
                 adaptive: bool = False, margin: int = 2) -> None:
        self.tau = tau
        self.lam_high = float(lam_high)
        self.adaptive = bool(adaptive)
        self.margin = int(margin)
        self._switched_at = None

    def decide(self, ctx: ControlContext) -> Decision:
        if self.adaptive and self.tau is None:
            worst = float(np.max(ctx.trP)) if ctx.M else 0.0
            inj = max(ctx.inj_hat, 1e-9)
            # rounds of sensing needed to pull worst-target trace under p_max, net of growth
            deficit = max(0.0, worst - ctx.p_max)
            net = max(inj - ctx.q_growth, 1e-9)
            need = int(np.ceil(deficit / net)) if deficit > 0 else 0
            remaining = ctx.T - ctx.t
            sense_now = (need + self.margin) >= remaining
            if sense_now and self._switched_at is None:
                self._switched_at = ctx.t
            lam = self.lam_high if sense_now else 0.0
            tau_eff = self._switched_at if self._switched_at is not None else -1
        else:
            tau = int(self.tau if self.tau is not None else ctx.T // 2)
            lam = 0.0 if ctx.t < tau else self.lam_high
            tau_eff = tau
        w_sense = np.full(ctx.M, lam / max(ctx.M, 1))
        return Decision(1.0 - lam, w_sense, lam,
                        {"tempo_tau_eff": tau_eff, "tempo_mode": "adaptive" if self.adaptive else "fixed"})


# --------------------------------------------------------------------------- #
class DPPController(Controller):
    """TEMPO-DPP (§1.3.2). Drift-plus-penalty with per-target virtual queues.

    Z_{t+1,m} = max(0, Z_{t,m} + tr(P_{t,m}) - P_max).  Per-round objective
    f_learn(S) + (1/V) sum_m Z_{t,m} dI_m(S)  ->  w_learn = 1, w_sense_m = Z_m/V.
    Larger V -> more learning weight (O(V) learning loss, O(1/V) constraint slack).
    """
    name = "tempo_dpp"

    def __init__(self, V: float = 1.0, p_max: float = 1.0, M: int | None = None,
                 z_init: float = 0.0) -> None:
        self.V = float(V)
        self.p_max = float(p_max)
        self.Z = None if M is None else np.full(int(M), float(z_init))
        self._z_init = float(z_init)

    def _ensure(self, M):
        if self.Z is None or self.Z.shape[0] != M:
            self.Z = np.full(M, self._z_init)

    def decide(self, ctx: ControlContext) -> Decision:
        self._ensure(ctx.M)
        w_sense = self.Z / max(self.V, 1e-9)
        lam = float(w_sense.sum() / (1.0 + w_sense.sum()))     # diagnostic fraction in [0,1)
        return Decision(1.0, w_sense.copy(), lam,
                        {"tempo_Zmean": float(self.Z.mean()), "tempo_Zmax": float(self.Z.max()),
                         "tempo_V": self.V})

    def observe(self, ctx: ControlContext) -> None:
        self._ensure(ctx.M)
        self.Z = np.maximum(0.0, self.Z + (ctx.trP - self.p_max))


# --------------------------------------------------------------------------- #
class MPCController(Controller):
    """TEMPO-MPC (§1.3.3). Receding-horizon planning over a surrogate of the
    coupled dynamics (learning-descent decay + scalar Riccati tracking).

    Each round it searches a small family of candidate schedules over the horizon
    H (constant lambda in a grid, plus threshold-at-h switches), rolls the surrogate
    forward, scores the mission objective (terminal or sustained), and executes the
    first action; replans next round. The realized lambda_s(t) is therefore
    time-varying and adapts to the measured state — the upper-performance reference.
    """
    name = "tempo_mpc"

    def __init__(self, horizon: int = 10, p_max: float = 1.0, mission: str = "sustained",
                 lam_grid=(0.0, 0.25, 0.5, 0.75, 1.0), penalty: float = 50.0,
                 rho_L: float = 0.05) -> None:
        self.H = int(horizon)
        self.p_max = float(p_max)
        self.mission = str(mission)
        self.lam_grid = tuple(lam_grid)
        self.penalty = float(penalty)
        self.rho_L = float(rho_L)

    def _rollout(self, lam_traj, L0, trP0, inj, q):
        """Surrogate forward model; returns (accuracy_proxy, violation_proxy)."""
        L = float(L0)
        trP = float(trP0)
        acc, viol = 0.0, 0.0
        traces = []
        for lam in lam_traj:
            acc += (1.0 - lam) * L                      # descent ~ learning-effort * grad energy
            L *= np.exp(-self.rho_L * (1.0 - lam))      # gradient energy decays as we learn
            trP = max(0.0, trP + q - inj * lam)         # scalar Riccati: growth q, injection inj*lam
            traces.append(trP)
        if self.mission == "terminal":
            viol = max(0.0, traces[-1] - self.p_max)
        else:                                            # sustained: average-trace constraint
            viol = max(0.0, float(np.mean(traces)) - self.p_max)
        return acc, viol

    def _candidates(self, h):
        cands = [("const", lam, [lam] * h) for lam in self.lam_grid]
        for sw in range(1, h):                           # threshold-at-sw: learn then full sense
            cands.append(("switch", sw, [0.0] * sw + [1.0] * (h - sw)))
        return cands

    def decide(self, ctx: ControlContext) -> Decision:
        h = int(min(self.H, max(1, ctx.T - ctx.t)))
        worst0 = float(np.max(ctx.trP)) if ctx.M else 0.0
        best = None
        for kind, key, traj in self._candidates(h):
            acc, viol = self._rollout(traj, ctx.L_t, worst0, ctx.inj_hat, ctx.q_growth)
            score = acc - self.penalty * viol
            if best is None or score > best[0]:
                best = (score, traj[0], kind, key)
        lam = float(best[1])
        w_sense = np.full(ctx.M, lam / max(ctx.M, 1))
        return Decision(1.0 - lam, w_sense, lam,
                        {"tempo_mpc_kind": best[2], "tempo_mpc_h": h, "tempo_mpc_score": float(best[0])})


def build_controller(spec: dict, T: int, M: int, p_max: float) -> Controller:
    """Instantiate a controller from a config dict {kind, ...}."""
    kind = spec["kind"]
    if kind == "threshold":
        return ThresholdController(tau=spec.get("tau"), lam_high=spec.get("lam_high", 1.0),
                                   adaptive=spec.get("adaptive", spec.get("tau") is None),
                                   margin=spec.get("margin", 2))
    if kind == "dpp":
        return DPPController(V=spec.get("V", 1.0), p_max=p_max, M=M, z_init=spec.get("z_init", 0.0))
    if kind == "mpc":
        return MPCController(horizon=spec.get("horizon", 10), p_max=p_max,
                             mission=spec.get("mission", "sustained"),
                             lam_grid=tuple(spec.get("lam_grid", (0.0, 0.25, 0.5, 0.75, 1.0))),
                             penalty=spec.get("penalty", 50.0), rho_L=spec.get("rho_L", 0.05))
    raise ValueError(f"unknown controller kind {kind!r}")
