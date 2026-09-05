#!/usr/bin/env bash
# =====================================================================
# SUPERSEDED by scripts/schedule_experiments.sh, which covers everything this
# script does and adds the interference, bandwidth and coherence sweeps, the
# regret certificate and the theory validation. Use that one. This script is kept
# because it reproduces the narrower AirComp-only re-run against campaign_main.yaml.
#
# Re-run the SCOUT-FL paper campaign with the FIXED AirComp MSE model.
#
#   bash scripts/rerun_fixed_aircomp.sh                 # full re-run (auto device)
#   DEVICE=cuda bash scripts/rerun_fixed_aircomp.sh     # on the GPU box
#   SHARDS=8 bash scripts/rerun_fixed_aircomp.sh        # more parallelism
#   bash scripts/rerun_fixed_aircomp.sh --stage 1       # one stage only
#
# WHY: sim/aircomp.py floored eta = P * g_min at 1e-12 as a divide-by-zero guard.
# Under the physical link budget eta is 1e-14..1e-10 W, so the floor was always
# active and capped the aggregation MSE at sigma^2/(K^2 * 1e-12) = 2.0067e-4,
# making the AirComp error independent of transmit power. 75.4% of all stored
# campaign rounds sit exactly on that ceiling. Fixed 2026-09-05; every number in
# the paper must be regenerated.
#
# SCOPE (decided): the sensing-aware pool only -- 13 methods, which is exactly
# the set the paper reports. Nominal operating point stays at -15 dBm, where the
# aggregation-MSE constraint now genuinely binds.
#
# All stages are resumable: re-run to continue, delete the run dir to recompute.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-auto}"
SHARDS="${SHARDS:-4}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
CFG=scout_fl/configs/campaign_main.yaml
SEEDS='[0,1,2,3,4]'
# the sensing-aware pool: SCOUT-FL + its hard-gate ablation + the 11 baselines the paper reports
POOL='[scout_v2,scout_greedy,collabsensefed,sensing_native,asaad,fixed_weighted,fed_iscc,ota_fl_iscc,iscc_air_feel,fedavg_iscc,fedsgd_iscc,crb_only,sensing_only]'

STAGE="${STAGE:-all}"
while [ $# -gt 0 ]; do case "$1" in --stage) STAGE="$2"; shift 2;; *) shift;; esac; done

archive() {   # move a clamped store aside instead of deleting it
    local d="$1"
    if [ -d "$d" ] && [ ! -d "runs/_clamped_$STAMP/$(basename "$d")" ]; then
        mkdir -p "runs/_clamped_$STAMP"
        mv "$d" "runs/_clamped_$STAMP/$(basename "$d")"
        echo "[archive] $d -> runs/_clamped_$STAMP/$(basename "$d")"
    fi
}

run_stage() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

echo "=============================================================="
echo "[rerun] device=$DEVICE shards=$SHARDS stage=$STAGE"
echo "[rerun] pool = 13 sensing-aware methods x 5 seeds x 150 rounds"
echo "=============================================================="

# ---------------------------------------------------------------- stage 1
# Main operating point (Table II, Fig. 3 frontier, Fig. 7 convergence,
# Fig. 10 runtime, Table VI ablation).  65 trainings -- run this first and
# check the realised MSE before committing to the long tail.
if run_stage 1; then
    archive runs/campaign_main
    echo "[rerun] stage 1/4: main operating point (13 methods x 5 seeds)"
    python -m scout_fl.experiments.run_fl_synthetic --config "$CFG" \
        --override "fl.device=$DEVICE" "experiment=campaign_main" \
        "seeds=$SEEDS" "selection.methods=$POOL"
fi

# ---------------------------------------------------------------- stage 2
# 25-point OFAT campaign (Fig. gap_dist, Fig. 4 paired, Fig. 6 threshold,
# Table III head-to-head, Table V cross-dataset).  1,625 trainings, sharded
# by sweep so the shards never write the same point directory.
if run_stage 2; then
    archive runs/campaign
    echo "[rerun] stage 2/4: 25-point OFAT campaign, $SHARDS shards"
    SH1="B_wireless_snr"                                        # 7 points (the wireless axis)
    SH2="A_datasets"                                            # 5 points (cifar100/emnist are slow)
    SH3="A_learning_noniid A_learning_partition"                # 5 points
    SH4="C_sensing_targets C_sensing_kangle B_wireless_channel" # 8 points
    pids=()
    for sh in "$SH1" "$SH2" "$SH3" "$SH4"; do
        log="logs/rerun_$(echo "$sh" | tr ' ' '_').log"; mkdir -p logs
        echo "  [shard] $sh -> $log"
        python -m scout_fl.experiments.run_campaign --config "$CFG" --tag campaign \
            --sweeps $sh --override "fl.device=$DEVICE" "seeds=$SEEDS" \
            "selection.methods=$POOL" >"$log" 2>&1 &
        pids+=($!)
    done
    fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [ "$fail" -eq 0 ] || { echo "[rerun] a shard failed -- see logs/rerun_*.log"; exit 1; }
fi

# ---------------------------------------------------------------- stage 3
# Trade-off weight sweep (Fig. 9).  7 lambda values x 5 seeds = 35 trainings.
if run_stage 3; then
    [ -d runs_lambda ] && [ ! -d "runs/_clamped_$STAMP/runs_lambda" ] && {
        mkdir -p "runs/_clamped_$STAMP"; mv runs_lambda "runs/_clamped_$STAMP/runs_lambda"
        echo "[archive] runs_lambda -> runs/_clamped_$STAMP/runs_lambda"; }
    echo "[rerun] stage 3/4: lambda sweep"
    for LAM in 0.0 0.25 0.5 1.0 2.0 4.0 8.0; do
        TAG="lam_${LAM//./p}"
        python -m scout_fl.experiments.run_fl_synthetic --config "$CFG" \
            --override "fl.device=$DEVICE" "runs_dir=runs_lambda" "experiment=$TAG" \
            "seeds=$SEEDS" "selection.methods=[scout_v2]" "objectives.lambda_sense=$LAM"
    done
fi

# ---------------------------------------------------------------- stage 4
# Budget adaptation + budget sweep (Fig. 12).  DEFERRED ON PURPOSE: the old
# epsilon grid (5e-5 .. 1e-3) was chosen around the 2.0e-4 clamp ceiling. With
# the clamp gone the realised MSE at -15 dBm is ~1e-3 before the dual engages,
# so the grid has to be re-centred on the value stage 1 actually reports.
if run_stage 4; then
    echo "[rerun] stage 4/4: budget experiments"
    echo "  NOT launched automatically -- re-centre the epsilon grid first."
    echo "  Read the realised MSE from stage 1:"
    echo "    python -c \"import json,glob,numpy as np; print(np.mean([json.load(open(f))['objectives']['agg_mse'] for f in glob.glob('runs/campaign_main/base/scout_v2__seed*.json')]))\""
    echo "  then edit the EPS grid in scripts/tccn_experiments.sh and run it."
fi

echo "=============================================================="
echo "[rerun] done (stage=$STAGE). Next: python -m scout_fl.analysis.collect"
echo "        then: cd research/paper && python paperfigs.py"
echo "=============================================================="
