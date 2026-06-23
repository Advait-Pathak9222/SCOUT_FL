#!/usr/bin/env bash
# =====================================================================
# Run the ENTIRE SCOUT-FL / JEDI-FL experiment program, step by step.
#
#   bash scripts/run_all_experiments.sh [DEVICE] [--quick] [--from STEP]
#
#   DEVICE   auto (default) | mps | cuda | cpu
#   --quick  tiny smoke of every step (proves the pipeline end-to-end fast)
#   --from N start at step N (1..4); earlier completed work is skipped anyway
#
# Examples:
#   bash scripts/run_all_experiments.sh                 # full program, auto device
#   bash scripts/run_all_experiments.sh cuda            # full program on NVIDIA
#   bash scripts/run_all_experiments.sh mps --quick     # fast end-to-end smoke on Apple GPU
#   bash scripts/run_all_experiments.sh cuda --from 3   # resume at the OFAT campaign
#
# RESUMABLE: every (point, method, seed) federated training is checkpointed per-round
# to runs/<tag>/<point>/<method>__seed<seed>.json. Kill the process any time and just
# re-run this script — completed units are loaded from disk and skipped. Delete a
# unit's JSON (or its folder) to force recomputation.
#
# Per-round results for analysis live under runs/ ; collected tables under runs/<tag>/.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="auto"; QUICK=""; FROM=1
while [ $# -gt 0 ]; do
    case "$1" in
        --quick) QUICK="--quick";;
        --from) shift; FROM="${1:-1}";;
        auto|mps|cuda|cpu) DEVICE="$1";;
        *) echo "unknown arg: $1"; exit 2;;
    esac
    shift
done

# Resolve 'auto' for the banner + MPS fallback env.
if [ "$DEVICE" = "auto" ]; then
    DEVICE=$(python -c "from scout_fl.utils.device import resolve_device; print(resolve_device('auto'))")
fi
[ "$DEVICE" = "mps" ] && export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OVR="fl.device=$DEVICE"

banner() { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }
echo "[run_all] device=$DEVICE  quick=${QUICK:-no}  from step $FROM  (resumable; re-run to continue)"

# --- Step 0: sanity (fast, always) ---------------------------------------
if [ "$FROM" -le 0 ] || [ "$FROM" -le 1 ]; then
    banner "Step 1/4  JEDI-FL ablation study (small rounds)"
    python -m scout_fl.experiments.run_fl_synthetic \
        --config scout_fl/configs/ablation.yaml $QUICK --override $OVR
    python -m scout_fl.analysis.collect --tag ablation
fi

# --- Step 2: main multi-seed bake-off (Tests A-E at the nominal operating point) ---
if [ "$FROM" -le 2 ]; then
    banner "Step 2/4  Main bake-off: 20 methods x seeds x 150 rounds (Tests A-E core)"
    python -m scout_fl.experiments.run_fl_synthetic \
        --config scout_fl/configs/campaign_main.yaml $QUICK --override $OVR
    python -m scout_fl.analysis.collect --tag campaign_main
fi

# --- Step 3: full OFAT campaign (Tests A-C sweeps) ----------------------------
if [ "$FROM" -le 3 ]; then
    banner "Step 3/4  OFAT campaign: learning / wireless / sensing sweeps (24 points)"
    python -m scout_fl.experiments.run_campaign $QUICK --override $OVR
    python -m scout_fl.analysis.collect --tag campaign
fi

# --- Step 4: P7 online-regret experiment (standalone; CUCB vs offline oracle) ----
if [ "$FROM" -le 4 ]; then
    banner "Step 4/5  P7 online-regret experiment (CUCB sensing selection)"
    python -m scout_fl.experiments.run_regret --config scout_fl/configs/campaign_main.yaml --rounds 300
fi

# --- Step 5: collect + plot + validate theory (P6 / P3-dual / P7) -----------------
if [ "$FROM" -le 5 ]; then
    banner "Step 5/5  Collect tables + plots + theory validation (P6, feasibility, P7)"
    python -m scout_fl.analysis.collect          # all tags -> runs/_all/{all_rounds,summary}.csv
    python -m scout_fl.analysis.plots            # convergence + energy-per-accuracy figures
    echo; python -m scout_fl.analysis.convergence --tag campaign_main   # P6 descent-bound regression
    echo; python -m scout_fl.analysis.feasibility --tag campaign_main   # P3-dual bounded violation
    echo; python -m scout_fl.analysis.regret                            # P7 sublinear regret
fi

banner "DONE. Per-round JSON: runs/<tag>/<point>/  |  tables: runs/_all/  |  plots+stats: outputs/"
