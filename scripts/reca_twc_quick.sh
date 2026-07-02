#!/usr/bin/env bash
# =====================================================================
# Quick RECA-FL TWC smoke test.
#
#   bash scripts/reca_twc_quick.sh [DEVICE] [--skip-fl] [--skip-data] [--no-download] [key=value ...]
#
# Examples:
#   bash scripts/reca_twc_quick.sh
#   bash scripts/reca_twc_quick.sh cuda
#   bash scripts/reca_twc_quick.sh cuda --no-download
#
# This does NOT rerun SCOUT/JEDI/baseline campaigns. It checks that RECA is
# registered as a proposed method, runs a tiny RECA-only FL pass, and then runs
# quick versions of the TWC RECA mechanism experiments.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
FL_CONFIG="${RECA_FL_CONFIG:-scout_fl/configs/reca_twc_nonstationary_fl.yaml}"
DEVICE="auto"
RUN_FL=1
PREPARE_DATA=1
DOWNLOAD_FLAG=""
EXTRA_OVERRIDES=()

while [ $# -gt 0 ]; do
    case "$1" in
        auto|cuda|mps|cpu) DEVICE="$1";;
        --skip-fl) RUN_FL=0;;
        --skip-data|--skip-datasets|--skip-data-prepare) PREPARE_DATA=0;;
        --no-download) DOWNLOAD_FLAG="--no-download"; EXTRA_OVERRIDES+=("fl.download=false");;
        *=*) EXTRA_OVERRIDES+=("$1");;
        *) echo "unknown arg: $1"; exit 2;;
    esac
    shift
done

if [ "$DEVICE" = "auto" ]; then
    DEVICE=$("$PYTHON" -c "from scout_fl.utils.device import resolve_device; print(resolve_device('auto'))")
fi
[ "$DEVICE" = "mps" ] && export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

banner() {
    echo
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

echo "[reca_quick] device=$DEVICE run_fl=$RUN_FL prepare_data=$PREPARE_DATA overrides=${EXTRA_OVERRIDES[*]:-none}"

if [ "$DEVICE" = "cuda" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[reca_quick] GPU(s):"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
    fi
    "$PYTHON" - <<'PY'
import torch
print(f"[reca_quick] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
PY
fi

banner "Compile and RECA registry check"
"$PYTHON" -m compileall -q \
    scout_fl/selection/baselines.py \
    scout_fl/experiments/run_fl_synthetic.py \
    scout_fl/experiments/run_reca_reuse.py \
    scout_fl/experiments/run_reca_shift.py \
    scout_fl/selection/reca_selector.py \
    scout_fl/objectives/reca_appraisal.py \
    scout_fl/objectives/world_model.py \
    scout_fl/fl/adapters.py \
    scout_fl/tests/test_reca.py
"$PYTHON" - <<'PY'
from scout_fl.selection.baselines import BASELINE_REGISTRY
assert "reca" in BASELINE_REGISTRY
print("[reca_quick] RECA registered:", type(BASELINE_REGISTRY["reca"]).__name__)
PY

if [ "$RUN_FL" -eq 1 ]; then
    if [ "$PREPARE_DATA" -eq 1 ]; then
        banner "Dataset preflight for quick RECA-only FL"
        "$PYTHON" -m scout_fl.experiments.prepare_datasets \
            --quick $DOWNLOAD_FLAG \
            --config "$FL_CONFIG" \
            --override "fl.device=$DEVICE" "${EXTRA_OVERRIDES[@]}"
    fi

    banner "Quick RECA-only FL training pass"
    "$PYTHON" -m scout_fl.experiments.run_fl_synthetic \
        --config "$FL_CONFIG" \
        --quick \
        --override \
        "experiment=reca_twc_quick_fl" \
        "fl.device=$DEVICE" \
        "selection.methods=[reca]" \
        "seeds=[0]" \
        "${EXTRA_OVERRIDES[@]}"
else
    echo "[reca_quick] RECA-only FL pass skipped"
fi

run_mechanism_quick() {
    local config="$1"
    local methods="$2"
    banner "Quick mechanism test: $(basename "$config") methods=$methods"
    "$PYTHON" -m scout_fl.experiments.run_reca_reuse \
        --config "$config" \
        --quick \
        --override "selection.methods=$methods"
}

run_mechanism_quick scout_fl/configs/reca_twc_stationary.yaml "[reca]"
run_mechanism_quick scout_fl/configs/reca_twc_shift.yaml "[reca,reca_no_memory,reca_score_only]"
run_mechanism_quick scout_fl/configs/reca_twc_ablation.yaml "[reca,reca_score_only,reca_no_adapter,reca_no_memory,reca_random_trigger,reca_periodic_trigger,reca_oracle_trigger,reca_no_quarantine,reca_frozen_adapter,wrong_reuse,oracle_reuse]"
run_mechanism_quick scout_fl/configs/reca_twc_tailrisk.yaml "[reca,reca_mean_risk,reca_no_risk,reca_no_overwhelm]"
run_mechanism_quick scout_fl/configs/reca_twc_resource_tradeoff.yaml "[reca]"
run_mechanism_quick scout_fl/configs/reca_twc_scalability.yaml "[reca,reca_score_only,reca_no_memory]"
run_mechanism_quick scout_fl/configs/reca_twc_reuse.yaml "[reca,reca_no_memory,reca_score_only,wrong_reuse,random_reuse,oracle_reuse]"
run_mechanism_quick scout_fl/configs/reca_twc_snr_mobility.yaml "[reca]"
run_mechanism_quick scout_fl/configs/reca_twc_powercontrol.yaml "[reca]"

banner "RECA TWC QUICK TEST PASSED"
