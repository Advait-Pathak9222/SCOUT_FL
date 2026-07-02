"""RECA-FL sudden-regime-shift wrapper.

Runs the same mechanism simulator as ``run_reca_reuse`` but with a single shift
window. This is the quick E2/E4 smoke path.
"""
from __future__ import annotations

from scout_fl.experiments.run_reca_reuse import main


if __name__ == "__main__":
    main(default_config="scout_fl/configs/reca_twc_shift.yaml")
