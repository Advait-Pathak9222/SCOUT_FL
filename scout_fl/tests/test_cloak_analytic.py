"""E-C1 / E-C2 analytic-study sanity + gate plumbing (fast; run in preflight)."""
import numpy as np

from scout_fl.cloak import entanglement as E
from scout_fl.cloak import dither_study as D
from scout_fl.cloak.mechanisms import all_modes, mode_params
from scout_fl.cloak.selection import leakage_capped_greedy
from scout_fl.infra.leakage import LeakageAccountant


def test_ec1_capped_never_exceeds_uncapped():
    mc = E.monte_carlo(n=40, seed=1, r_floor=10.0)
    for c in mc["cases"]:
        assert c["u_cap"] <= c["u_full"] + 1e-9              # cap only removes utility
        assert 0.0 <= c["retained"] <= 1.0 + 1e-9


def test_ec1_tighter_floor_retains_less():
    loose = E.monte_carlo(n=40, seed=2, r_floor=5.0)["retained_median"]
    tight = E.monte_carlo(n=40, seed=2, r_floor=50.0)["retained_median"]
    assert tight <= loose + 1e-9                             # stricter privacy -> less target utility


def test_ec1_gate2_verdict_shape():
    v = E.gate2(E.monte_carlo(n=30, seed=0))
    assert set(v) >= {"gate", "criterion", "measured", "threshold", "pass", "framing"}


def test_ec2_perfect_sync_exact():
    inv = D.aggregate_invariance(sigma_d=1.0, seed=0)
    assert inv[0]["sigma_sync"] == 0.0 and inv[0]["eps_agg_max"] < 1e-9


def test_ec2_eaves_inflation_and_gate3():
    infl = D.eavesdropper_inflation(snr_eve=10.0)
    # inflation is monotone increasing in dither variance
    vals = [r["inflation_1rx"] for r in infl]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    v = D.gate3(infl)
    assert v["pass"] in (True, False) and v["eavesdropper_inflation_1rx"] >= 1.0


def test_leakage_capped_greedy_respects_cap():
    # a trivial monotone utility: prefer clients in index order
    class U:
        def init_state(self): return set()
        def add(self, s, k): return s | {k}
        def marginal_gain(self, s, k): return 100.0 - k       # prefers low indices
    clients = np.random.default_rng(0).uniform(0, 100, size=(20, 2))
    acct = LeakageAccountant(clients, np.array([50.0, 50.0]), k_range=1.0, k_angle=0.05)
    snr = np.full(20, 25.0)
    r_floor = 5.0
    # pre-load several clients near their cap so the gate must skip some
    for _ in range(30):
        acct.observe([0, 1, 2], snr, atten=1.0)
    sel, relaxed = leakage_capped_greedy(U(), 20, 5, accountant=acct, snr_up=snr, r_floor=r_floor)
    assert len(sel) == 5
    for k in sel:
        # each selected client stays at/above the CRB floor after this round (unless relaxed)
        assert acct.projected_crb_floor(k, snr[k]) >= r_floor - 1e-9 or relaxed > 0


def test_leakage_cap_forces_rotation():
    """A tight floor must push participation off the few over-exposed clients."""
    clients = np.random.default_rng(1).uniform(0, 100, size=(15, 2))
    acct = LeakageAccountant(clients, np.array([50.0, 50.0]), k_range=1.0, k_angle=0.05)
    snr = np.full(15, 25.0)

    class U:                                                   # everyone prefers client 0
        def init_state(self): return set()
        def add(self, s, k): return s | {k}
        def marginal_gain(self, s, k): return 1.0 if k else 100.0
    picked0 = 0
    for _ in range(60):
        sel, _ = leakage_capped_greedy(U(), 15, 3, accountant=acct, snr_up=snr, r_floor=8.0)
        acct.observe(sel, snr, atten=1.0)
        picked0 += (0 in sel)
    assert picked0 < 60                                        # client 0 eventually capped out


def test_all_modes_parse():
    for m in all_modes():
        p = mode_params(m)
        assert p["selector"] in ("m1", "random")
        assert p["mse_infl"] >= 1.0 and 0.0 <= p["leak_atten"] <= 1.0
