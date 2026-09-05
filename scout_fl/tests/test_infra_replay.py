"""Preflight gate: deterministic replay must reconstruct the campaign geometry
faithfully (else E-C4 leakage re-scoring and E-T2 tracker re-scoring are invalid).

Guards the RNG draw-order contract in infra/replay.reconstruct against regressions.
Skips gracefully if the campaign artifacts are absent (fresh checkout).
"""
import glob

import pytest

from scout_fl.infra import replay

_CFG = "scout_fl/configs/campaign_main.yaml"
_POINT = "A_datasets=cifar10"
_FILES = sorted(glob.glob(f"runs/campaign/{_POINT}/*__seed0.json"))


@pytest.mark.skipif(not _FILES, reason="no campaign artifacts present")
def test_replay_reconstructs_round0_logdet():
    # The skipif above covers pytest. This project's own convention is to import and call
    # the test functions directly, which does not see markers, and a fresh clone has no
    # runs/ because it is generated. Guard here so both runners agree.
    if not _FILES:
        return
    cfg = replay.config_for_point(_POINT, _CFG)
    checked = 0
    for f in _FILES[:6]:
        art = replay.load_artifact(f)
        if not art or not art.get("complete"):
            continue
        ok, rp, lg, diff = replay.verify_against_artifact(cfg, art, tol=1e-3)
        assert ok, f"{art['meta']['method']}: replay logdet {rp:.4f} != logged {lg:.4f} (|diff|={diff:.2e})"
        checked += 1
    assert checked >= 1, "no complete units verified"


def test_reconstruct_is_deterministic():
    cfg = replay.config_for_point(_POINT, _CFG)
    a = replay.reconstruct(cfg, 0)
    b = replay.reconstruct(cfg, 0)
    import numpy as np
    assert np.allclose(a["clients"], b["clients"])
    assert np.allclose(a["snr_up"], b["snr_up"])              # same seed -> identical channel
