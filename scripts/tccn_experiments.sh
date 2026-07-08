#!/usr/bin/env bash
# =====================================================================
# TCCN revision-proofing experiments for the SCOUT-FL paper.
#
#   bash scripts/tccn_experiments.sh              # full run (GPU)
#   DEVICE=cpu bash scripts/tccn_experiments.sh   # CPU fallback
#   bash scripts/tccn_experiments.sh --quick      # tiny smoke of every stage
#
# Three experiments, then the paper figures:
#
#   E-R1  Cognitive adaptation (paper Fig. 12). Main operating point, but the
#         aggregation-MSE budget is CUT at round 75 from 1e-3 to 1.5e-4, below
#         the realised floor of ~2.0e-4, so the constraint binds mid-run. The
#         dual price mu must rise, reprice weak-channel clients, and pull the
#         realised MSE back under the new budget. SCOUT-FL and the hard-gate
#         ablation run under the identical disturbance.
#         Cost: 2 methods x 5 seeds x 150 rounds.
#
#   E-R2  Budget sweep, soft vs hard gate (paper Fig. 13). Sweeps the static
#         budget eps from deep in the binding regime (5e-5) to the slack
#         operating point (1e-3). Backs the paper's claim that the soft/hard
#         accuracy gap widens as the budget binds.
#         Cost: 5 budgets x 2 methods x 5 seeds x 150 rounds.
#
#   E-R3  Empirical curvature of the sensing utility (paper Remark 3).
#         Exact, offline, seconds on CPU. Already reflected in the paper text.
#
# All runs are resumable (re-run to resume; delete runs_tccn/ to recompute).
# Figures land in research/paper/figures/ with stats JSON next to them.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

QUICK=(); RUNS_DIR="runs_tccn"
for a in "$@"; do
    case "$a" in --quick) QUICK=(--quick); RUNS_DIR="runs_tccn_smoke";; esac
done
# --quick writes to runs_tccn_smoke so 3-round smoke artifacts can never be
# mistaken for completed full runs by the resume logic.

CFG=scout_fl/configs/campaign_main.yaml
SEEDS='[0,1,2,3,4]'
METHODS='[scout_v2,scout_greedy]'

echo "=============================================================="
echo "[tccn] device=$DEVICE  quick=${QUICK[*]:-no}  runs -> $RUNS_DIR/"
echo "=============================================================="

# ---------------------------------------------------------- E-R1 adaptation
echo "[tccn] E-R1 cognitive adaptation: budget cut 1e-3 -> 1.5e-4 at round 75"
python -m scout_fl.experiments.run_fl_synthetic --config "$CFG" ${QUICK[@]+"${QUICK[@]}"} \
    --override "fl.device=$DEVICE" "runs_dir=$RUNS_DIR" "experiment=adapt_eps" \
    "seeds=$SEEDS" "selection.methods=$METHODS" \
    "constraints.dual_normalized=true" "constraints.mse_eps_schedule=75:1.5e-4"

# ---------------------------------------------------------- E-R2 budget sweep
for EPS in 5e-5 1e-4 1.5e-4 2e-4 1e-3; do
    echo "[tccn] E-R2 budget sweep: eps=$EPS"
    python -m scout_fl.experiments.run_fl_synthetic --config "$CFG" ${QUICK[@]+"${QUICK[@]}"} \
        --override "fl.device=$DEVICE" "runs_dir=$RUNS_DIR" "experiment=eps_$EPS" \
        "seeds=$SEEDS" "selection.methods=$METHODS" \
        "constraints.dual_normalized=true" "constraints.mse_agg_max=$EPS"
done

# ---------------------------------------------------------- E-R3 curvature
echo "[tccn] E-R3 empirical curvature of the sensing utility"
python -m scout_fl.analysis.curvature --config "$CFG" --seeds 0 1 2 3 4

# ---------------------------------------------------------- paper figures
echo "[tccn] rendering paper figures 12 and 13"
( cd research/paper && python -c "import paperfigs as pf; pf.fig_adaptation(); pf.fig_eps_sweep()" )

echo "=============================================================="
echo "[tccn] done. Check:"
echo "  research/paper/figures/fig12_adaptation.{pdf,png}"
echo "  research/paper/figures/fig13_epsgate.{pdf,png}"
echo "  research/paper/figures/stats/{adaptation,curvature}.json"
echo "=============================================================="
