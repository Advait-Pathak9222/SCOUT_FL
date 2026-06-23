#!/usr/bin/env bash
# =====================================================================
# Run the FULL OFAT campaign (Tests A-E) on Apple-silicon macOS (MPS GPU).
#
#   bash scripts/campaign_macos.sh [EXTRA ARGS/OVERRIDES...]
#
# Examples:
#   bash scripts/campaign_macos.sh --dry-run                       # print the matrix, no compute
#   bash scripts/campaign_macos.sh --quick                         # tiny end-to-end smoke
#   bash scripts/campaign_macos.sh --sweeps B_wireless_snr         # one sweep only
#   bash scripts/campaign_macos.sh                                 # the whole campaign on MPS
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

FLAGS=(); OVERRIDES=(fl.device=mps)
for a in "$@"; do
    case "$a" in --*) FLAGS+=("$a");; *) OVERRIDES+=("$a");; esac
done

echo "[campaign_macos] device=mps  flags: ${FLAGS[*]:-none}  overrides: ${OVERRIDES[*]}"
python -m scout_fl.experiments.run_campaign \
    ${FLAGS[@]+"${FLAGS[@]}"} \
    --override "${OVERRIDES[@]}"
