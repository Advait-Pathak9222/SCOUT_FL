"""TEMPO schedules / controllers / mixed-utility sanity (fast; run in preflight)."""
import numpy as np

from scout_fl.objectives.learning_utility import LearningUtility
from scout_fl.tempo import schedules as S
from scout_fl.tempo.controllers import (ControlContext, DPPController, MPCController,
                                        ThresholdController, build_controller)
from scout_fl.tempo.mixed_utility import MixedUtility
from scout_fl.sim.fim import per_client_target_fim, prior_fim
from scout_fl.sim.geometry import pairwise_geometry


def _toy_scenario(K=12, M=2, seed=0):
    rng = np.random.default_rng(seed)
    clients = rng.uniform(0, 100, size=(K, 2))
    targets = rng.uniform(30, 70, size=(M, 2))
    geom = pairwise_geometry(clients, targets)
    snr = np.full((K, M), 5.0)
    fim = per_client_target_fim(geom, snr, 1.0, 0.05)
    j0 = prior_fim(M, 1e-3)
    embs = rng.standard_normal((K, 8))
    return LearningUtility(embeddings=embs), fim, j0, K, M


def test_schedules_bounded_and_correct():
    lts = S.learn_then_sense(tau=50)
    assert lts(10, 100) == 0.0 and lts(60, 100) == 1.0        # learn early, sense late
    burst = S.bursting(burst_len=3, period=15)
    assert burst(0, 100) == 1.0 and burst(5, 100) == 0.0      # sensing burst then off
    rr = S.roundrobin()
    assert rr(0, 100) == 0.0 and rr(1, 100) == 1.0
    lin = S.linear_anneal()
    assert lin(0, 100) == 0.0 and abs(lin(99, 100) - 1.0) < 1e-9
    for sch in [lts, burst, rr, lin, S.two_phase(), S.static(0.7), S.random_schedule(0)]:
        vals = [sch(t, 50) for t in range(50)]
        assert all(0.0 <= v <= 1.0 for v in vals)


def test_mixed_utility_endpoints():
    learning, fim, j0, K, M = _toy_scenario()
    full = list(range(K))
    # lambda=0 -> pure learning: value equals normalized f_learn
    u0 = MixedUtility(learning, fim, j0, 1.0, np.zeros(M), K)
    assert abs(u0.value(full) - 1.0) < 1e-6                   # normalized full-set learning ~1
    # lambda=1 (uniform) -> sensing dominates; monotone submodular still holds
    u1 = MixedUtility(learning, fim, j0, 0.0, np.ones(M), K)
    assert u1.value([0, 1, 2]) <= u1.value([0, 1, 2, 3]) + 1e-9


def test_mixed_utility_incremental_matches_value():
    learning, fim, j0, K, M = _toy_scenario(seed=3)
    u = MixedUtility(learning, fim, j0, 0.6, np.array([0.3, 0.7]), K)
    subset = [1, 4, 7]
    st = u.init_state()
    for k in subset:
        st = u.add(st, k)
    # incremental sum of marginal gains == set value (submodular consistency)
    st2, acc = u.init_state(), 0.0
    for k in subset:
        acc += u.marginal_gain(st2, k)
        st2 = u.add(st2, k)
    assert abs(acc - u.value(subset)) < 1e-6


def test_threshold_controller_fixed_and_adaptive():
    fixed = ThresholdController(tau=40)
    ctx = ControlContext(t=10, T=100, trP=np.array([5.0, 5.0]), L_t=1.0, p_max=10.0, M=2)
    assert fixed.decide(ctx).lam == 0.0
    ctx2 = ControlContext(t=50, T=100, trP=np.array([5.0]), L_t=1.0, p_max=10.0, M=1)
    assert fixed.decide(ctx2).lam == 1.0
    adap = ThresholdController(adaptive=True)
    # huge deficit near the end -> should switch on sensing
    late = ControlContext(t=95, T=100, trP=np.array([100.0]), L_t=1.0, p_max=10.0, M=1,
                          inj_hat=5.0, q_growth=0.0)
    assert adap.decide(late).lam > 0.0


def test_dpp_queue_drives_sensing():
    dpp = DPPController(V=1.0, p_max=5.0, M=2)
    ctx = ControlContext(t=0, T=100, trP=np.array([20.0, 20.0]), L_t=1.0, p_max=5.0, M=2)
    d0 = dpp.decide(ctx)
    assert np.allclose(d0.w_sense, 0.0)                      # queues start empty
    for _ in range(5):                                       # violation accumulates the queue
        dpp.observe(ctx)
    d1 = dpp.decide(ctx)
    assert d1.w_sense.sum() > d0.w_sense.sum()               # sensing weight grows with violation


def test_mpc_prefers_learning_when_slack_and_sensing_when_tight():
    mpc = MPCController(horizon=10, p_max=10.0, mission="sustained")
    slack = ControlContext(t=0, T=100, trP=np.array([1.0]), L_t=5.0, p_max=10.0, M=1,
                           inj_hat=2.0, q_growth=0.1)
    tight = ControlContext(t=0, T=100, trP=np.array([50.0]), L_t=5.0, p_max=10.0, M=1,
                           inj_hat=2.0, q_growth=0.1)
    assert mpc.decide(slack).lam <= mpc.decide(tight).lam    # tighter constraint -> more sensing


def test_build_controller_registry():
    for spec in [{"kind": "threshold", "tau": 30}, {"kind": "dpp", "V": 2.0},
                 {"kind": "mpc", "horizon": 5}]:
        c = build_controller(spec, T=100, M=2, p_max=10.0)
        assert c.decide(ControlContext(0, 100, np.array([1.0, 1.0]), 1.0, 10.0, 2)) is not None
