"""Preflight gate (design §3.1): the leakage accountant must reproduce a
hand-computable 2-client case to numerical tolerance.

Hand case (BS at origin):
  client 1 @ (10,0): range 10, u=(1,0), v=(0,1)
      unit FIM = k_range*uuᵀ + (k_angle/100)*vvᵀ = [[1,0],[0,0.01]]   (k_range=1, k_angle=1)
  client 2 @ (0,20): range 20, u=(0,1), v=(-1,0)
      unit FIM = [[0.0025,0],[0,1]]
Select client 1 for 3 rounds at snr_up=2 -> cumulative J1 = 6*unit1 = [[6,0],[0,0.06]].
Prior J0 = I/100^2. r1 = sqrt(tr((J0+J1)^-1)) ~ 4.0994 m.
"""
import numpy as np

from scout_fl.infra.leakage import LeakageAccountant, cap_from_crb_floor, client_leak_fim


def test_unit_fim_matches_hand():
    clients = np.array([[10.0, 0.0], [0.0, 20.0]])
    J = client_leak_fim(clients, np.zeros(2), k_range=1.0, k_angle=1.0)
    assert np.allclose(J[0], [[1.0, 0.0], [0.0, 0.01]], atol=1e-12)
    assert np.allclose(J[1], [[0.0025, 0.0], [0.0, 1.0]], atol=1e-12)


def test_accountant_two_client_hand_case():
    clients = np.array([[10.0, 0.0], [0.0, 20.0]])
    acct = LeakageAccountant(clients, np.zeros(2), k_range=1.0, k_angle=1.0, prior_std_m=100.0)
    for _ in range(3):
        acct.observe([0], snr_up=np.array([2.0, 2.0]), atten=1.0)

    # independent hand computation
    unit1 = np.array([[1.0, 0.0], [0.0, 0.01]])
    J0 = np.eye(2) / 100.0 ** 2
    acc1 = J0 + 3 * 2.0 * unit1
    r1_hand = float(np.sqrt(np.trace(np.linalg.inv(acc1))))

    r = acct.crb_floor()
    assert abs(r[0] - r1_hand) < 1e-9
    assert abs(r1_hand - 4.09942) < 1e-3                       # closed-form scalar
    # client 2 never selected -> stays at the prior floor sqrt(2)*100
    assert abs(r[1] - np.sqrt(2) * 100.0) < 1e-6
    assert abs(acct.trace_leak()[0] - (6.0 + 0.06)) < 1e-9     # tr(6*unit1)


def test_cap_guarantees_floor():
    """Enforcing tr(J) <= 4/r^2 must guarantee the CRB floor >= r (design leakage.py)."""
    clients = np.array([[15.0, 5.0], [3.0, 40.0]])
    acct = LeakageAccountant(clients, np.zeros(2), k_range=1.0, k_angle=0.05, prior_std_m=100.0)
    r_floor = 10.0
    cap = cap_from_crb_floor(r_floor)
    # accumulate right up to the cap for client 0
    snr = np.array([1.0, 1.0])
    while acct.projected_trace(0, snr[0]) <= cap:
        acct.observe([0], snr, atten=1.0)
    assert acct.crb_floor()[0] >= r_floor - 1e-6              # floor honored
