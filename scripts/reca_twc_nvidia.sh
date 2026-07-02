#!/usr/bin/env bash
# =====================================================================
# Full RECA-FL TWC campaign on NVIDIA CUDA.
#
#   bash scripts/reca_twc_nvidia.sh [--no-quick] [--skip-data] [--no-download] [--seeds 0,1,2,3,4] [key=value ...]
#
# Design:
#   * Runs ONLY RECA as the proposed method in the real FL trainer.
#   * Does NOT rerun SCOUT/JEDI or external baseline campaigns.
#   * Runs the extra RECA-specific TWC mechanism experiments from the plan.
#   * Uses CUDA and prints progress for every experiment, method, and seed.
#
# Examples:
#   bash scripts/reca_twc_nvidia.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/reca_twc_nvidia.sh --seeds 0,1,2,3,4
#   bash scripts/reca_twc_nvidia.sh --no-download fl.dataset=cifar10
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
FL_CONFIG="${RECA_FL_CONFIG:-scout_fl/configs/reca_twc_nonstationary_fl.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

RUN_QUICK=1
PREPARE_DATA=1
DOWNLOAD_FLAG=""
SEED_LIST=(0 1 2 3 4)
FL_EXTRA_OVERRIDES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --no-quick) RUN_QUICK=0;;
        --skip-data|--skip-datasets|--skip-data-prepare) PREPARE_DATA=0;;
        --no-download) DOWNLOAD_FLAG="--no-download"; FL_EXTRA_OVERRIDES+=("fl.download=false");;
        --seeds)
            shift
            IFS=',' read -r -a SEED_LIST <<< "${1:-0,1,2,3,4}"
            ;;
        seeds=*)
            raw="${1#seeds=}"
            raw="${raw#[}"
            raw="${raw%]}"
            IFS=',' read -r -a SEED_LIST <<< "$raw"
            ;;
        *=*) FL_EXTRA_OVERRIDES+=("$1");;
        *) echo "unknown arg: $1"; exit 2;;
    esac
    shift
done

SEEDS_YAML="[$(IFS=,; echo "${SEED_LIST[*]}")]"

banner() {
    echo
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

echo "[reca_nvidia] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES seeds=$SEEDS_YAML quick_gate=$RUN_QUICK prepare_data=$PREPARE_DATA"
echo "[reca_nvidia] FL extra overrides=${FL_EXTRA_OVERRIDES[*]:-none}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[reca_nvidia] GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi
"$PYTHON" - <<'PY'
import torch
print(f"[reca_nvidia] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
print("[reca_nvidia] gpu=", torch.cuda.get_device_name(0))
PY

if [ "$RUN_QUICK" -eq 1 ]; then
    banner "Quick gate before full RECA campaign"
    bash scripts/reca_twc_quick.sh cuda --skip-data --skip-fl
fi

if [ "$PREPARE_DATA" -eq 1 ]; then
    banner "Dataset preflight for RECA-only CUDA FL"
    "$PYTHON" -m scout_fl.experiments.prepare_datasets \
        $DOWNLOAD_FLAG \
        --config "$FL_CONFIG" \
        --override "fl.device=cuda" "${FL_EXTRA_OVERRIDES[@]}"
else
    echo "[reca_nvidia] dataset preflight skipped"
fi

banner "E1/E2 wireless FL backbone: RECA-only training on CUDA"
"$PYTHON" -m scout_fl.experiments.run_fl_synthetic \
    --config "$FL_CONFIG" \
    --override \
    "experiment=reca_twc_fl_reca_only" \
    "fl.device=cuda" \
    "selection.methods=[reca]" \
    "seeds=$SEEDS_YAML" \
    "${FL_EXTRA_OVERRIDES[@]}"

run_reca_config_all_seeds() {
    local label="$1"
    local config="$2"
    local methods="$3"
    shift 3
    local extra=("$@")

    banner "$label  config=$(basename "$config") methods=$methods"
    for seed in "${SEED_LIST[@]}"; do
        echo "[reca_nvidia] $label seed=$seed methods=$methods"
        "$PYTHON" -m scout_fl.experiments.run_reca_reuse \
            --config "$config" \
            --override \
            "seed=$seed" \
            "selection.methods=$methods" \
            "${extra[@]}"
    done
}

run_reca_config_all_seeds "E1 stationary wireless ISAC-FEEL mechanism" \
    scout_fl/configs/reca_twc_stationary.yaml "[reca]"

run_reca_config_all_seeds "E2 non-stationary wireless ISAC shift" \
    scout_fl/configs/reca_twc_shift.yaml "[reca,reca_no_memory,reca_score_only]"

run_reca_config_all_seeds "E3 adapter mechanism proof" \
    scout_fl/configs/reca_twc_ablation.yaml "[reca,reca_score_only,reca_no_adapter,reca_no_memory,reca_random_trigger,reca_periodic_trigger,reca_oracle_trigger,reca_no_quarantine,reca_frozen_adapter,wrong_reuse,oracle_reuse]"

run_reca_config_all_seeds "E5 tail-risk constrained wireless reliability" \
    scout_fl/configs/reca_twc_tailrisk.yaml "[reca,reca_mean_risk,reca_no_risk,reca_no_overwhelm]"

run_reca_config_all_seeds "E6 wireless resource trade-off curves" \
    scout_fl/configs/reca_twc_resource_tradeoff.yaml "[reca]"

run_reca_config_all_seeds "E7 overhead and scalability" \
    scout_fl/configs/reca_twc_scalability.yaml "[reca,reca_score_only,reca_no_memory]"

run_reca_config_all_seeds "E8 adapter reuse and cross-regime generalization" \
    scout_fl/configs/reca_twc_reuse.yaml "[reca,reca_no_memory,reca_score_only,wrong_reuse,random_reuse,oracle_reuse]"

run_reca_config_all_seeds "E9 SNR, mobility, and channel robustness" \
    scout_fl/configs/reca_twc_snr_mobility.yaml "[reca]"

run_reca_config_all_seeds "E10 beamforming / power-control compatibility" \
    scout_fl/configs/reca_twc_powercontrol.yaml "[reca]"

banner "DONE: RECA TWC NVIDIA campaign"
echo "[reca_nvidia] RECA-only FL runs: runs/reca_twc_fl_reca_only/base/"
echo "[reca_nvidia] RECA TWC mechanism outputs: outputs/reca_twc_*/"
