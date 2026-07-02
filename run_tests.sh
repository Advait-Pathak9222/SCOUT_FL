#!/usr/bin/env bash
# =====================================================================
# One-shot test entrypoint for ALL THREE proposed methods:
#   RECA-FL   — risk-bounded epistemic context accommodation (appraisal/adapters/selector)
#   TEMPO-FL  — temporal mission planning (schedules, Threshold/DPP/MPC, tracker)
#   CloakFL   — location privacy (leakage accountant, M1 caps, M2 dither, E-C1/E-C2 gates)
#
#   bash run_tests.sh                # unit tests + schema/replay verification (fast, ~15 s)
#   bash run_tests.sh --with-smoke   # + tiny functional end-to-end runs of each method
#   bash run_tests.sh --full         # + the ENTIRE repo test suite (SCOUT/JEDI/VISMAYA too)
#
# Exit code is non-zero on any failure — safe to use as a CI / pre-launch gate.
# The TEMPO/CloakFL campaign launcher (run_all.sh) runs the same preflight subset
# before dispatching GPU work.
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
WITH_SMOKE=0
FULL=0
for a in "$@"; do
  case "$a" in
    --with-smoke) WITH_SMOKE=1 ;;
    --full) FULL=1 ;;
    *) echo "unknown arg $a (use --with-smoke / --full)"; exit 2 ;;
  esac
done

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/run_tests_${TS}.log"
FAILED=0

banner() { echo | tee -a "$LOG"; echo "===== $* =====" | tee -a "$LOG"; }
run()    { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; local rc="${PIPESTATUS[0]}";
           [[ "$rc" -ne 0 ]] && { echo "FAILED: $*" | tee -a "$LOG"; FAILED=1; }; return 0; }

# ------------------------------------------------------------------ RECA-FL
banner "RECA-FL: appraisal / adapter-bank / world-model / registry selector"
run "$PY" -m pytest scout_fl/tests/test_reca.py scout_fl/tests/test_baselines.py -q
run "$PY" - <<'EOF'
from scout_fl.selection.baselines import BASELINE_REGISTRY
assert "reca" in BASELINE_REGISTRY, "RECA not registered in the FL bake-off"
print("[reca] registered as:", type(BASELINE_REGISTRY["reca"]).__name__)
EOF

# ------------------------------------------------------------------ TEMPO-FL
banner "TEMPO-FL: schedules / controllers / mixed utility / tracker (NEES gate, design R4)"
run "$PY" -m pytest scout_fl/tests/test_tempo.py scout_fl/tests/test_infra_tracker.py -q

# ------------------------------------------------------------------ CloakFL
banner "CloakFL: leakage accountant (hand-case) / M1 caps / M2 exact invariance / E-C1-E-C2"
run "$PY" -m pytest scout_fl/tests/test_infra_leakage.py scout_fl/tests/test_infra_dither.py \
    scout_fl/tests/test_cloak_analytic.py -q

# ------------------------------------------------------------------ shared campaign infra
banner "Campaign infra: replay faithfulness / unit-grid integrity / schema report"
run "$PY" -m pytest scout_fl/tests/test_infra_replay.py scout_fl/tests/test_units_grid.py -q
run "$PY" -m scout_fl.analysis.schema_report --strict

# ------------------------------------------------------------------ optional functional smokes
if [[ "$WITH_SMOKE" -eq 1 ]]; then
  banner "SMOKE: RECA mechanism simulator (quick, no GPU)"
  run "$PY" -m scout_fl.experiments.run_reca_reuse \
      --config scout_fl/configs/reca_twc_shift.yaml --quick \
      --override "selection.methods=[reca,reca_no_memory,reca_score_only]"

  banner "SMOKE: RECA inside the real FL trainer (registry path, tiny)"
  run "$PY" -m scout_fl.experiments.run_fl_synthetic \
      --config scout_fl/configs/reca_twc_nonstationary_fl.yaml --quick \
      --override "experiment=reca_smoke_fl" "fl.device=auto" \
                 "selection.methods=[reca]" "seeds=[0]" "fl.download=false"

  banner "SMOKE: TEMPO + CloakFL (one tiny unit per experiment type, isolated roots)"
  SMOKE_UIDS="logs/uids_tests_smoke_${TS}.txt"
  run "$PY" -m scout_fl.experiments.plan --smoke --list --incomplete-only
  "$PY" -m scout_fl.experiments.plan --smoke --list --incomplete-only > "$SMOKE_UIDS" || true
  if [[ -s "$SMOKE_UIDS" ]]; then
    run "$PY" -m scout_fl.experiments.dispatch --uids-file "$SMOKE_UIDS" \
        --num-gpus "${NUM_GPUS:-0}" --smoke --log-dir "logs/units_tests_${TS}"
  else
    echo "[smoke] all smoke units already complete (runs_smoke/)" | tee -a "$LOG"
  fi
fi

# ------------------------------------------------------------------ optional full suite
if [[ "$FULL" -eq 1 ]]; then
  banner "FULL repo test suite (SCOUT/JEDI/VISMAYA + RECA + TEMPO + CloakFL)"
  run "$PY" -m pytest scout_fl/tests -q
fi

banner "SUMMARY"
if [[ "$FAILED" -ne 0 ]]; then
  echo "RESULT: FAILED — see $LOG" | tee -a "$LOG"
  exit 1
fi
echo "RESULT: ALL TESTS PASSED (log: $LOG)" | tee -a "$LOG"
