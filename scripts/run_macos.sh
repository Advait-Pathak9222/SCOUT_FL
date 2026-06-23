#!/usr/bin/env bash
# =====================================================================
# Run the SCOUT-FL / ADVAYA-FL campaign on Apple-silicon macOS (MPS GPU).
#
#   bash scripts/run_macos.sh [CONFIG] [EXTRA OVERRIDES...]
#
# Examples:
#   bash scripts/run_macos.sh                                   # main campaign, CIFAR-10, MPS
#   bash scripts/run_macos.sh scout_fl/configs/campaign_main.yaml fl.dataset=cifar100
#   bash scripts/run_macos.sh scout_fl/configs/campaign_main.yaml fl.dataset=fashion_mnist  # debug layer
#   bash scripts/run_macos.sh scout_fl/configs/campaign_main.yaml fl.dirichlet_alpha=0.1     # non-IID sweep point
#
# Tips:
#   * Force CPU instead of MPS:   ... fl.device=cpu
#   * A quick end-to-end smoke:   bash scripts/run_macos.sh scout_fl/configs/campaign_main.yaml fl.dataset=mnist fl.model=mlp  (add --quick manually below)
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-scout_fl/configs/campaign_main.yaml}"
shift || true

# Let unsupported MPS ops silently fall back to CPU instead of crashing.
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Reproducible BLAS threading on macOS.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Split args into flags (--quick, ...) and key=value config overrides.
FLAGS=(); OVERRIDES=(fl.device=mps)
for a in "$@"; do
    case "$a" in --*) FLAGS+=("$a");; *) OVERRIDES+=("$a");; esac
done

echo "[run_macos] config=$CONFIG  device=mps (Apple MPS)  flags: ${FLAGS[*]:-none}  overrides: ${OVERRIDES[*]}"
python -m scout_fl.experiments.run_fl_synthetic \
    --config "$CONFIG" \
    ${FLAGS[@]+"${FLAGS[@]}"} \
    --override "${OVERRIDES[@]}"
