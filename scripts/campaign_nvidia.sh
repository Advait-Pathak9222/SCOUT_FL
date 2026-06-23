#!/usr/bin/env bash
# =====================================================================
# Run the FULL OFAT campaign (Tests A-E) on an NVIDIA GPU server (CUDA).
#
#   bash scripts/campaign_nvidia.sh [EXTRA ARGS/OVERRIDES...]
#
# Examples:
#   bash scripts/campaign_nvidia.sh --dry-run                      # print the matrix, no compute
#   bash scripts/campaign_nvidia.sh --quick                        # tiny end-to-end smoke
#   CUDA_VISIBLE_DEVICES=1 bash scripts/campaign_nvidia.sh         # full campaign, pinned GPU 1
#   bash scripts/campaign_nvidia.sh --sweeps A_datasets B_wireless_snr
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

FLAGS=(); OVERRIDES=(fl.device=cuda)
for a in "$@"; do
    case "$a" in --*) FLAGS+=("$a");; *) OVERRIDES+=("$a");; esac
done

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[campaign_nvidia] GPU(s):"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi
echo "[campaign_nvidia] device=cuda  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  flags: ${FLAGS[*]:-none}  overrides: ${OVERRIDES[*]}"
python -m scout_fl.experiments.run_campaign \
    ${FLAGS[@]+"${FLAGS[@]}"} \
    --override "${OVERRIDES[@]}"
