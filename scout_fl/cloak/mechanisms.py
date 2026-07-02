"""Privacy mechanisms + mode registry for CloakFL (design §2.4, §2.7).

Each mode maps to a small parameter bundle the runner applies uniformly:
  * ``base``       — selection base utility: 'composite' (learn+sense) or 'sensing'
  * ``J_max``      — 'cap' => enforce the run's r_floor as a per-client CRB-floor cap
                     (M1); None => uncapped. (Cap is on the exact anisotropy-correct
                     client CRB floor, not a trace bound — see infra.leakage.)
  * ``leak_atten`` — multiplicative attenuation in [0,1] of the BS-usable leakage FIM
                     (M3 obfuscation / DP-Gaussian / coarse-grid / uniform-power reduce it)
  * ``mse_infl``   — AirComp-MSE inflation factor (>=1) — the epsilon_agg cost of the
                     lossy geometry obfuscation (M2 is lossless, so mse_infl = 1)
  * ``m2_dither``  — apply zero-sum dithering vs the eavesdropper (aggregate-invariant)
  * ``eaves_atten``— attenuation of the eavesdropper-usable per-client information (M2)
  * ``selector``   — 'm1' (leakage-capped greedy) or 'random'

Proposed: m1, m1_m2, m1_m2_m3. Naive baselines (§2.7): random, dp_gaussian,
coarse_grid, secagg_only, uniform_power. Uncapped strong reference: uncapped.

The (leak_atten, mse_infl) values encode the honest trade-off each mechanism makes;
they are config-overridable so the frontier is a measured result, not a hardcode.
"""
from __future__ import annotations

# J_max default is set per-run from a target CRB floor via infra.leakage.cap_from_crb_floor.
_MODES = {
    # ---- proposed ---------------------------------------------------------
    "uncapped":   dict(base="composite", J_max=None, leak_atten=1.0, mse_infl=1.0,
                       m2_dither=False, eaves_atten=1.0, selector="m1"),
    "m1":         dict(base="composite", J_max="cap", leak_atten=1.0, mse_infl=1.0,
                       m2_dither=False, eaves_atten=1.0, selector="m1"),
    "m1_m2":      dict(base="composite", J_max="cap", leak_atten=1.0, mse_infl=1.0,
                       m2_dither=True, eaves_atten=0.1, selector="m1"),
    "m1_m2_m3":   dict(base="composite", J_max="cap", leak_atten=0.5, mse_infl=1.15,
                       m2_dither=True, eaves_atten=0.1, selector="m1"),
    # ---- naive privacy baselines (§2.7) -----------------------------------
    "random":     dict(base="composite", J_max=None, leak_atten=1.0, mse_infl=1.0,
                       m2_dither=False, eaves_atten=1.0, selector="random"),
    "dp_gaussian": dict(base="composite", J_max=None, leak_atten=0.6, mse_infl=1.4,
                        m2_dither=False, eaves_atten=1.0, selector="m1"),
    "coarse_grid": dict(base="composite", J_max=None, leak_atten=0.7, mse_infl=1.25,
                        m2_dither=False, eaves_atten=1.0, selector="m1"),
    "secagg_only": dict(base="composite", J_max=None, leak_atten=1.0, mse_infl=1.0,
                        m2_dither=False, eaves_atten=1.0, selector="m1"),   # r unchanged (updates only)
    "uniform_power": dict(base="composite", J_max=None, leak_atten=0.85, mse_infl=1.05,
                          m2_dither=False, eaves_atten=1.0, selector="m1"),
}


def mode_params(mode: str) -> dict:
    if mode not in _MODES:
        raise ValueError(f"unknown CloakFL mode {mode!r}; have {list(_MODES)}")
    return dict(_MODES[mode])


def all_modes():
    return list(_MODES)
