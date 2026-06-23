#!/usr/bin/env bash
# =====================================================================
# Run the SCOUT-FL / ADVAYA-FL campaign on an NVIDIA GPU server (CUDA).
#
#   bash scripts/run_nvidia.sh [CONFIG] [EXTRA OVERRIDES...]
#
# Examples:
#   bash scripts/run_nvidia.sh                                  # main campaign, CIFAR-10, CUDA:0
#   bash scripts/run_nvidia.sh scout_fl/configs/campaign_main.yaml fl.dataset=cifar100
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_nvidia.sh           # pin to GPU 1
#   bash scripts/run_nvidia.sh scout_fl/configs/campaign_main.yaml fl.dirichlet_alpha=0.5
#
# Tips:
#   * Select a specific GPU:  CUDA_VISIBLE_DEVICES=2 bash scripts/run_nvidia.sh ...
#   * Force CPU:              ... fl.device=cpu
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-scout_fl/configs/campaign_main.yaml}"
shift || true

# Default to the first visible GPU unless the caller already pinned one.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Split args into flags (--quick, ...) and key=value config overrides.
FLAGS=(); OVERRIDES=(fl.device=cuda)
for a in "$@"; do
    case "$a" in --*) FLAGS+=("$a");; *) OVERRIDES+=("$a");; esac
done

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[run_nvidia] GPU(s):"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi
echo "[run_nvidia] config=$CONFIG  device=cuda  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  flags: ${FLAGS[*]:-none}  overrides: ${OVERRIDES[*]}"
python -m scout_fl.experiments.run_fl_synthetic \
    --config "$CONFIG" \
    ${FLAGS[@]+"${FLAGS[@]}"} \
    --override "${OVERRIDES[@]}"
