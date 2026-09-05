#!/usr/bin/env bash
# =====================================================================
# SCOUT-FL, full experiment schedule for the TCCN submission.
#
#   bash scripts/schedule_experiments.sh --dry-run       # print the plan and the cost
#   DEVICE=cuda SHARDS=8 bash scripts/schedule_experiments.sh
#   bash scripts/schedule_experiments.sh --stage 1       # one stage only
#   bash scripts/schedule_experiments.sh --from 4        # stage 4 onwards
#
# Everything is resumable. Re-run to continue, delete the run directory to force a
# recompute. Stages are ordered so that the cheap stages that gate the expensive ones
# come first, and stage 4 reads its grid from what stage 1 actually measured.
#
# WHY A FULL RE-RUN. Five things changed in the simulator, each of which moved the
# numbers, and all of them are in scout_fl/configs/campaign_tccn.yaml.
#   1. the AirComp error no longer saturates on a numerical floor
#   2. the small scale channel moves between rounds instead of being frozen
#   3. the sensing SNR tracks the transmit power, so a power sweep is an ISAC sweep
#   4. the distortion injected into training carries the measured update power
#   5. co-channel interference exists and enters through the noise floor
# The archived stores under runs/_clamped_* are the old numbers, kept for the record.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-auto}"
SHARDS="${SHARDS:-4}"
# Shards share one card, so each is capped at its slice. The cap bounds the allocator and
# also scales how many clients the probe differentiates at once, which is what would
# otherwise oversubscribe the device when every shard sizes itself as if it were alone.
SHARE=$(python -c "print(round(0.92/max($SHARDS,1), 4))")
CFG=scout_fl/configs/campaign_tccn.yaml
SEEDS='[0,1,2,3,4]'
POOL='[scout_v2,scout_greedy,collabsensefed,sensing_native,asaad,fixed_weighted,fed_iscc,ota_fl_iscc,iscc_air_feel,fedavg_iscc,fedsgd_iscc,crb_only,sensing_only,comm_only]'
# The physical-layer sweeps interrogate the scheduler's mechanism rather than run a
# thirteen way bake-off, so they use SCOUT-FL, its ablation, and the four strongest rivals.
CORE='[scout_v2,scout_greedy,collabsensefed,sensing_native,asaad,crb_only,comm_only]'

STAGE=""; FROM=1; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2;;
        --from) FROM="$2"; shift 2;;
        --dry-run) DRY=1; shift;;
        *) shift;;
    esac
done
run_stage() {
    [ "$DRY" -eq 0 ] || return 1
    if [ -n "$STAGE" ]; then [ "$STAGE" = "$1" ]; else [ "$1" -ge "$FROM" ]; fi
}
banner() { echo; echo "=============================================================="; echo "[stage $1] $2"; echo "=============================================================="; }
# Throughput. The probe is 69 percent of a round and was 100 separate forward and backward
# passes; it is now one vectorised call over device resident data. Measured 1.7x on the
# probe and 1.66x per round on Apple MPS, where the probe is now compute bound. On CUDA the
# saving is larger, because what the batching removes is launch overhead. Set
# fl.deterministic=false to add the cuDNN autotuner when a sweep does not need to be exactly
# reproducible.

if [ "$DRY" -eq 1 ]; then
cat <<'PLAN'
==============================================================
 SCOUT-FL experiment schedule
==============================================================
 stage  what                                    runs    what it buys
 -----  --------------------------------------  ------  --------------------------------
   1    main operating point, 14 methods          70    Table II, frontier, convergence,
                                                        overhead, constraint ablation
   2    OFAT campaign, 25 points, 14 methods    1750    Pareto share, paired deltas,
                                                        threshold sweep, cross dataset
   3    trade off weight lambda, 7 values          35    the lambda frontier figure
   4    aggregation budget, 5 static + 1 cut      60    adaptation and budget figures,
                                                        grid taken from stage 1
   5    interference floor, 6 levels              210    the claim that the dual price
                                                        absorbs interference unestimated
   6    bandwidth, 5 values                      175    the closed form optimal bandwidth
   7    channel coherence, 4 values              140    whether an adaptive price earns
                                                        its place on a moving channel
   8    online regret, CUCB against the oracle      5    the sublinear regret certificate
   9    theory validation, no new training          0    convergence and feasibility checks
  10    baseline collapse analysis, no training     0    which baselines reduce to channel
                                                        quality selection
 -----  --------------------------------------  ------
 total                                           2445
==============================================================
 cost   Measured on Apple MPS, 234 ms per round with the fast path against 389 ms
        without it, so about 35 s per 150 round run. That is roughly 24 hours on one
        worker and 3 hours at SHARDS=8. A CUDA card should beat both, since most of
        what the batching removes is launch overhead.
        Stage 2 is 72 percent of it. Stages 1, 3, 4, 8, 9, 10 together are around an
        hour and produce every headline number, so run those first if time is short.
==============================================================
 speed  fl.fast_path holds the training subsample on the device and differentiates
        every client's probe in one vectorised call. It is exact, and the tests in
        tests/test_fastpath.py check the gradients and the SGD updates against the
        reference path. Turn it off with fl.fast_path=false if you need the old path.
        fl.probe_chunk=auto sizes the probe from free memory and this process's share.
        fl.deterministic=false adds the cuDNN autotuner for sweeps that need not be
        exactly reproducible.
==============================================================
PLAN
exit 0
fi

# --------------------------------------------------------------- 1. main point
if run_stage 1; then
banner 1 "main operating point, 14 methods x 5 seeds"
python -m scout_fl.experiments.run_fl_synthetic --config "$CFG" \
    --override "fl.device=$DEVICE" "experiment=tccn_main" "seeds=$SEEDS" "selection.methods=$POOL"
fi

# --------------------------------------------------------------- 2. OFAT campaign
if run_stage 2; then
banner 2 "OFAT campaign, 25 points x 14 methods x 5 seeds, $SHARDS shards"
mkdir -p logs
pids=()
for sh in "A_datasets" "B_wireless_snr" "A_learning_noniid A_learning_partition" \
          "C_sensing_targets C_sensing_kangle B_wireless_channel"; do
    log="logs/tccn_$(echo "$sh" | tr ' ' '_').log"
    echo "  [shard] $sh -> $log"
    python -m scout_fl.experiments.run_campaign --config "$CFG" --tag tccn_campaign \
        --sweeps $sh --override "fl.device=$DEVICE" "seeds=$SEEDS" "selection.methods=$POOL" \
        "fl.cuda_memory_fraction=$SHARE" >"$log" 2>&1 &
    pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "[stage 2] a shard failed, see logs/tccn_*.log"; exit 1; }
fi

# --------------------------------------------------------------- 3. lambda sweep
if run_stage 3; then
banner 3 "trade off weight lambda, 7 values x 5 seeds"
for LAM in 0.0 0.25 0.5 1.0 2.0 4.0 8.0; do
    python -m scout_fl.experiments.run_fl_synthetic --config "$CFG" \
        --override "fl.device=$DEVICE" "runs_dir=runs_lambda_tccn" "experiment=lam_${LAM//./p}" \
        "seeds=$SEEDS" "selection.methods=[scout_v2]" "objectives.lambda_sense=$LAM"
done
fi

# --------------------------------------------------------------- 4. budget experiments
if run_stage 4; then
banner 4 "aggregation budget, grid centred on what stage 1 measured"
EPS_GRID=$(python - <<'PY'
import glob, json, numpy as np
fs = glob.glob("runs/tccn_main/base/scout_v2__seed*.json")
vals = []
for f in fs:
    d = json.load(open(f))
    if d.get("complete"):
        vals += [r["agg_mse"] for r in d["rounds"][-50:]]
if not vals:
    raise SystemExit("stage 1 has not produced runs/tccn_main yet; run stage 1 first")
m = float(np.median(vals))
# span the binding regime around the realised error, from a quarter of it to four times it
print(" ".join(f"{m*f:.3g}" for f in (0.25, 0.5, 1.0, 2.0, 4.0)))
PY
)
echo "  realised-error-centred grid: $EPS_GRID"
for EPS in $EPS_GRID; do
    python -m scout_fl.experiments.run_fl_synthetic --config "$CFG" \
        --override "fl.device=$DEVICE" "runs_dir=runs_eps_tccn" "experiment=eps_$EPS" \
        "seeds=$SEEDS" "selection.methods=[scout_v2,scout_greedy]" "constraints.mse_agg_max=$EPS"
done
CUT=$(echo "$EPS_GRID" | awk '{print $2}')      # mid-run cut to half the realised error
python -m scout_fl.experiments.run_fl_synthetic --config "$CFG" \
    --override "fl.device=$DEVICE" "runs_dir=runs_eps_tccn" "experiment=adapt_eps" \
    "seeds=$SEEDS" "selection.methods=[scout_v2,scout_greedy]" \
    "constraints.mse_eps_schedule=75:$CUT"
python -m scout_fl.analysis.curvature --config "$CFG" --seeds 0 1 2 3 4 || true
fi

# --------------------------------------------------------------- 5-7. physical layer sweeps
if run_stage 5; then
banner 5 "interference floor, 6 levels x 7 methods x 5 seeds"
python -m scout_fl.experiments.run_campaign --config "$CFG" --tag tccn_campaign \
    --sweeps B_wireless_interference \
    --override "fl.device=$DEVICE" "seeds=$SEEDS" "selection.methods=$CORE"
fi
if run_stage 6; then
banner 6 "bandwidth, 5 values x 7 methods x 5 seeds"
python -m scout_fl.experiments.run_campaign --config "$CFG" --tag tccn_campaign \
    --sweeps B_wireless_bandwidth \
    --override "fl.device=$DEVICE" "seeds=$SEEDS" "selection.methods=$CORE"
fi
if run_stage 7; then
banner 7 "channel coherence, 4 values x 7 methods x 5 seeds"
python -m scout_fl.experiments.run_campaign --config "$CFG" --tag tccn_campaign \
    --sweeps B_wireless_coherence \
    --override "fl.device=$DEVICE" "seeds=$SEEDS" "selection.methods=$CORE"
fi

# --------------------------------------------------------------- 8. online regret
if run_stage 8; then
banner 8 "online regret, CUCB against the offline oracle"
python -m scout_fl.experiments.run_regret --config "$CFG" --seeds 0 1 2 3 4 \
    --rounds 300 --out runs --override "fl.device=$DEVICE" || true
fi

# --------------------------------------------------------------- 9-10. analysis, no training
if run_stage 9; then
banner 9 "theory validation and collection"
python -m scout_fl.analysis.collect runs || true
python -m scout_fl.analysis.convergence runs --tag tccn_main || true
python -m scout_fl.analysis.feasibility runs --tag tccn_main --method scout_v2 || true
python -m scout_fl.analysis.regret runs --tag regret || true
fi
if run_stage 10; then
banner 10 "baseline collapse analysis"
python -m scout_fl.analysis.baseline_overlap --tag tccn_main
fi

echo
echo "=============================================================="
echo "[schedule] done. Render the paper figures with:"
echo "    cd research/paper && python paperfigs.py"
echo "=============================================================="
