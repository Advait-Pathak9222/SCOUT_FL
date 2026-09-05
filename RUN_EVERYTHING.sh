#!/usr/bin/env bash
# =====================================================================
#  SCOUT-FL, one command, no attention required.
#
#      bash RUN_EVERYTHING.sh                 # the whole thing
#      bash RUN_EVERYTHING.sh --smoke         # same pipeline, tiny, ~5 min
#      nohup bash RUN_EVERYTHING.sh &         # leave it and walk away
#
#  It picks its own device and shard count, downloads what it needs, runs every
#  stage in order, prints results after each one, and writes everything to
#  RESULTS/. No stage can stop the run. If a stage fails it is recorded and the
#  pipeline carries on, so you come back to results plus a list of what broke.
#
#  Kill it and run it again and it continues where it stopped. Every stage is
#  resumable at the level of a single training run.
#
#  Environment you may set, none of which is required:
#      DEVICE=cuda|mps|cpu    default auto
#      SHARDS=8               default from the core count, capped at 8
#      SEEDS='[0,1,2,3,4]'    default five seeds
#      SKIP=2,6               stage numbers to leave out
#      ONLY=1,3               stage numbers to run, everything else skipped
# =====================================================================

# Deliberately no `set -e`. An unattended run must survive a failing stage.
set -uo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------- options
SMOKE=0
for a in "$@"; do case "$a" in --smoke) SMOKE=1;; esac; done

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS="RESULTS"
LOGDIR="$RESULTS/logs"
STATUS="$RESULTS/STATUS.txt"
REPORT="$RESULTS/REPORT.md"
MAIN_LOG="$LOGDIR/run_${STAMP}.log"
MARKERS="$RESULTS/.done"
mkdir -p "$LOGDIR" "$MARKERS"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg                 # never try to open a window
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then echo "no python3 or python on PATH, stopping." >&2; exit 1; fi

# ---------------------------------------------------------------- device and shards
if [ -z "${DEVICE:-}" ]; then
    DEVICE=$("$PY" - <<'EOF' 2>/dev/null || echo cpu
from scout_fl.utils.device import resolve_device
print(resolve_device("auto"))
EOF
)
fi
if [ -z "${SHARDS:-}" ]; then
    CORES=$("$PY" -c "import os; print(os.cpu_count() or 4)" 2>/dev/null || echo 4)
    SHARDS=$(( CORES / 2 )); [ "$SHARDS" -lt 1 ] && SHARDS=1; [ "$SHARDS" -gt 8 ] && SHARDS=8
fi
SEEDS="${SEEDS:-[0,1,2,3,4]}"
SHARE=$("$PY" -c "print(round(0.92/max($SHARDS,1),4))" 2>/dev/null || echo 0.2)

CFG=scout_fl/configs/campaign_tccn.yaml
POOL='[scout_v2,scout_greedy,collabsensefed,sensing_native,asaad,fixed_weighted,fed_iscc,ota_fl_iscc,iscc_air_feel,fedavg_iscc,fedsgd_iscc,crb_only,sensing_only,comm_only]'
CORE='[scout_v2,scout_greedy,collabsensefed,sensing_native,asaad,crb_only,comm_only]'
COMMON=("fl.device=$DEVICE" "seeds=$SEEDS")

if [ "$SMOKE" -eq 1 ]; then
    SEEDS='[0,1]'
    COMMON=("fl.device=$DEVICE" "seeds=$SEEDS" "fl.rounds=3" "network.num_clients=30"
            "network.budget=5" "fl.subsample_train=3000" "fl.subsample_test=800"
            "fl.dataset=mnist" "fl.download=true")
    POOL='[scout_v2,scout_greedy,asaad,collabsensefed,comm_only]'
    CORE='[scout_v2,scout_greedy,asaad,comm_only]'
    RESULTS_NOTE="SMOKE RUN, tiny settings, results are not publishable"
else
    RESULTS_NOTE="full run"
fi

# ---------------------------------------------------------------- plumbing
say()  { printf '%s\n' "$*" | tee -a "$MAIN_LOG"; }
rule() { say "=================================================================="; }

record() {  # stage, status, seconds
    printf '%-4s %-46s %-9s %8s\n' "$1" "$2" "$3" "$4" >> "$STATUS"
}

wanted() {   # honour ONLY and SKIP
    local n="$1"
    if [ -n "${ONLY:-}" ]; then case ",${ONLY}," in *,"$n",*) ;; *) return 1;; esac; fi
    if [ -n "${SKIP:-}" ]; then case ",${SKIP}," in *,"$n",*) return 1;; esac; fi
    return 0
}

stage() {   # stage number, human name, then the command
    local n="$1" name="$2"; shift 2
    if ! wanted "$n"; then say ""; say "[stage $n] $name -- skipped by request"; record "$n" "$name" skipped "-"; return 0; fi
    if [ -f "$MARKERS/stage$n" ]; then say ""; say "[stage $n] $name -- already done, skipping"; record "$n" "$name" done-earlier "-"; return 0; fi
    say ""; rule; say "[stage $n] $name"; say "  started $(date '+%F %T')"; rule
    local log="$LOGDIR/stage${n}_${STAMP}.log" t0 t1 rc
    t0=$(date +%s)
    ( "$@" ) >>"$log" 2>&1
    rc=$?
    t1=$(date +%s)
    local secs=$((t1-t0))
    if [ "$rc" -eq 0 ]; then
        touch "$MARKERS/stage$n"
        say "  finished in $((secs/60)) min $((secs%60)) s"
        record "$n" "$name" ok "${secs}s"
    else
        say "  FAILED after $((secs/60)) min $((secs%60)) s, exit $rc. Tail of $log:"
        tail -n 12 "$log" | sed 's/^/    /' | tee -a "$MAIN_LOG"
        say "  carrying on to the next stage."
        record "$n" "$name" "FAILED($rc)" "${secs}s"
    fi
    return 0
}

digest() { say ""; say "  results so far"; "$PY" -m scout_fl.analysis.digest "$@" 2>&1 | tee -a "$MAIN_LOG"; }

finish() {
    rule; say "PIPELINE FINISHED $(date '+%F %T')"
    say ""; say "stage status"
    [ -f "$STATUS" ] && sed 's/^/  /' "$STATUS" | tee -a "$MAIN_LOG"
    # grep -c prints 0 and exits 1 when nothing matches, so a `|| echo 0` here would
    # produce two lines and break the integer test on a clean run.
    local failed
    failed=$(grep -c FAILED "$STATUS" 2>/dev/null); failed=${failed:-0}
    say ""
    if [ "$failed" -gt 0 ]; then
        say "$failed stage(s) failed. Their logs are in $LOGDIR. Re-running this script"
        say "retries only what did not finish, because completed stages are marked."
    else
        say "every stage completed."
    fi
    say ""; say "results in $RESULTS/"; say "  REPORT.md    the summary"
    say "  STATUS.txt   what ran and how long"
    say "  tables/      collected CSV"
    say "  figures/     rendered paper figures"
    say "  logs/        per stage output"
    rule
    write_report
    restore_figures
}
trap 'say ""; say "interrupted, writing the report for what finished"; finish; exit 130' INT TERM

write_report() {
    {
        echo "# SCOUT-FL results"
        echo
        echo "Generated $(date '+%F %T'). $RESULTS_NOTE."
        echo
        echo "Device \`$DEVICE\`, $SHARDS shards, seeds \`$SEEDS\`, config \`$CFG\`."
        echo
        echo "## Stages"
        echo
        echo '```'
        [ -f "$STATUS" ] && cat "$STATUS"
        echo '```'
        echo
        echo "## Main operating point"
        echo
        echo '```'
        "$PY" -m scout_fl.analysis.digest --tag tccn_main 2>&1
        echo '```'
        echo
        echo "## Campaign, best method per operating point"
        echo
        echo '```'
        "$PY" -m scout_fl.analysis.digest --tag tccn_campaign --by-point 2>&1
        echo '```'
        echo
        echo "## Which baselines reduce to channel quality selection"
        echo
        echo '```'
        "$PY" -m scout_fl.analysis.baseline_overlap --tag tccn_main \
              --out "$RESULTS/tables/baseline_overlap.json" 2>&1
        echo '```'
        echo
        echo "## Files"
        echo
        echo '```'
        ls -1 "$RESULTS/tables" 2>/dev/null | sed 's/^/tables\//'
        ls -1 "$RESULTS/figures" 2>/dev/null | sed 's/^/figures\//'
        echo '```'
    } > "$REPORT" 2>/dev/null
    say "report written to $REPORT"
}

# A smoke run writes three-round output into research/paper/figures, which holds the
# figures the paper uses. Stages 4 and 10 both write there, so the originals are put
# aside for the whole pipeline and restored at the end.
FIGKEEP="$RESULTS/.figures_backup"
if [ "$SMOKE" -eq 1 ] && [ -d research/paper/figures ]; then
    rm -rf "$FIGKEEP"; mkdir -p "$FIGKEEP"
    cp -a research/paper/figures/. "$FIGKEEP"/ 2>/dev/null
fi
restore_figures() {
    [ "$SMOKE" -eq 1 ] && [ -d "$FIGKEEP" ] || return 0
    rm -rf research/paper/figures; mkdir -p research/paper/figures
    cp -a "$FIGKEEP"/. research/paper/figures/ 2>/dev/null
    rm -rf "$FIGKEEP"
    say "  smoke mode, the paper figures were restored to what they were"
}

# ---------------------------------------------------------------- start
: > "$STATUS"
printf '%-4s %-46s %-9s %8s\n' "st" "stage" "status" "time" >> "$STATUS"
rule
say " SCOUT-FL, unattended run          $RESULTS_NOTE"
say " started   $(date '+%F %T')"
say " device    $DEVICE"
say " shards    $SHARDS   (device share $SHARE per shard)"
say " seeds     $SEEDS"
say " log       $MAIN_LOG"
rule

# ---------------------------------------------------------------- 0. preflight
say ""; say "[stage 0] preflight"
"$PY" - <<'EOF' 2>&1 | tee -a "$MAIN_LOG"
import shutil, sys
print(f"  python {sys.version.split()[0]}")
try:
    import torch, numpy, scipy
    print(f"  torch {torch.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}")
    from torch.func import vmap                     # the fast path needs this
    print("  torch.func available, the batched probe will be used")
except Exception as exc:
    print(f"  WARNING {exc}. The run will fall back to the slower path.")
try:
    import scout_fl, scout_fl.experiments.run_fl_synthetic, scout_fl.analysis.digest
    print("  scout_fl imports cleanly")
except Exception as exc:
    print(f"  FATAL cannot import scout_fl: {exc}"); raise SystemExit(1)
free = shutil.disk_usage(".").free / 1e9
print(f"  free disk {free:.1f} GB" + ("" if free > 20 else "   WARNING, under 20 GB"))
EOF
if [ "${PIPESTATUS[0]}" -ne 0 ]; then say "preflight failed, stopping."; exit 1; fi
record 0 "preflight" ok "-"

# ---------------------------------------------------------------- 0b. self test
stage 0b "self test, the whole suite" "$PY" - <<'EOF'
import importlib, inspect
mods = ["test_aircomp","test_aggregate","test_objectives","test_primal_dual","test_baselines",
        "test_theory","test_joint_information","test_sensing","test_fair_testbed","test_vismaya",
        "test_units_grid","test_stats","test_fl_pipeline","test_fastpath","test_infra_replay"]
p=f=0; fails=[]
for m in mods:
    try: mod = importlib.import_module(f"scout_fl.tests.{m}")
    except Exception as e: fails.append((m,"IMPORT",repr(e)[:160])); continue
    for n,fn in vars(mod).items():
        if n.startswith("test_") and callable(fn) and not inspect.signature(fn).parameters:
            try: fn(); p+=1
            except Exception as e: f+=1; fails.append((m,n,repr(e)[:200]))
print(f"passed={p} failed={f}")
for x in fails: print("  FAIL", x)
raise SystemExit(1 if f else 0)
EOF

# ---------------------------------------------------------------- 0c. datasets
if [ "$SMOKE" -eq 0 ]; then
stage 0c "download and verify the datasets" \
    "$PY" -m scout_fl.experiments.prepare_datasets --config "$CFG" \
        --datasets cifar10 cifar100 emnist fashion_mnist uci_har
fi

# ---------------------------------------------------------------- 1. main point
stage 1 "main operating point" \
    "$PY" -m scout_fl.experiments.run_fl_synthetic --config "$CFG" \
        --override "${COMMON[@]}" "experiment=tccn_main" "selection.methods=$POOL"
digest --tag tccn_main

# ---------------------------------------------------------------- 2. OFAT campaign
run_campaign_sharded() {
    local pids=() sh log rc=0
    for sh in "A_datasets" "B_wireless_snr" "A_learning_noniid A_learning_partition" \
              "C_sensing_targets C_sensing_kangle B_wireless_channel"; do
        log="$LOGDIR/campaign_$(echo "$sh" | tr ' ' '_')_${STAMP}.log"
        "$PY" -m scout_fl.experiments.run_campaign --config "$CFG" --tag tccn_campaign \
            --sweeps $sh --override "${COMMON[@]}" "selection.methods=$POOL" \
            "fl.cuda_memory_fraction=$SHARE" >"$log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    return $rc
}
if [ "$SMOKE" -eq 1 ]; then
    stage 2 "OFAT campaign, one sweep only in smoke mode" \
        "$PY" -m scout_fl.experiments.run_campaign --config "$CFG" --tag tccn_campaign \
            --sweeps B_wireless_channel --override "${COMMON[@]}" "selection.methods=$POOL"
else
    stage 2 "OFAT campaign, 25 points, $SHARDS shards" run_campaign_sharded
fi
digest --tag tccn_campaign --by-point

# ---------------------------------------------------------------- 3. lambda
run_lambda() {
    local rc=0 lam
    for lam in 0.0 0.25 0.5 1.0 2.0 4.0 8.0; do
        "$PY" -m scout_fl.experiments.run_fl_synthetic --config "$CFG" \
            --override "${COMMON[@]}" "runs_dir=runs_lambda_tccn" \
            "experiment=lam_${lam//./p}" "selection.methods=[scout_v2]" \
            "objectives.lambda_sense=$lam" || rc=1
    done
    return $rc
}
stage 3 "trade off weight lambda, seven values" run_lambda

# ---------------------------------------------------------------- 4. budget
run_budget() {
    local grid cut rc=0 eps
    grid=$("$PY" - <<'EOF'
import glob, json, numpy as np
vals = []
for f in glob.glob("runs/tccn_main/base/scout_v2__seed*.json"):
    d = json.load(open(f))
    if d.get("complete"):
        vals += [r["agg_mse"] for r in d["rounds"][-50:]]
m = float(np.median(vals)) if vals else 1e-3
print(" ".join(f"{m*f:.3g}" for f in (0.25, 0.5, 1.0, 2.0, 4.0)))
EOF
) || return 1
    echo "  budget grid centred on the realised error: $grid"
    for eps in $grid; do
        "$PY" -m scout_fl.experiments.run_fl_synthetic --config "$CFG" \
            --override "${COMMON[@]}" "runs_dir=runs_eps_tccn" "experiment=eps_$eps" \
            "selection.methods=[scout_v2,scout_greedy]" "constraints.mse_agg_max=$eps" || rc=1
    done
    cut=$(echo "$grid" | awk '{print $2}')
    # The schedule goes in as a list of pairs. Written as "75:$cut" it would be read as a
    # YAML 1.1 base sixty number and silently collapse to a single float.
    local at=$([ "$SMOKE" -eq 1 ] && echo 2 || echo 75)
    "$PY" -m scout_fl.experiments.run_fl_synthetic --config "$CFG" \
        --override "${COMMON[@]}" "runs_dir=runs_eps_tccn" "experiment=adapt_eps" \
        "selection.methods=[scout_v2,scout_greedy]" \
        "constraints.mse_eps_schedule=[[$at,$cut]]" || rc=1
    "$PY" -m scout_fl.analysis.curvature --config "$CFG" --seeds 0 1 2 3 4 || true
    return $rc
}
stage 4 "aggregation budget, static sweep plus a mid run cut" run_budget

# ---------------------------------------------------------------- 5-7. physical layer
stage 5 "interference floor" \
    "$PY" -m scout_fl.experiments.run_campaign --config "$CFG" --tag tccn_campaign \
        --sweeps B_wireless_interference --override "${COMMON[@]}" "selection.methods=$CORE"
stage 6 "bandwidth" \
    "$PY" -m scout_fl.experiments.run_campaign --config "$CFG" --tag tccn_campaign \
        --sweeps B_wireless_bandwidth --override "${COMMON[@]}" "selection.methods=$CORE"
stage 7 "channel coherence" \
    "$PY" -m scout_fl.experiments.run_campaign --config "$CFG" --tag tccn_campaign \
        --sweeps B_wireless_coherence --override "${COMMON[@]}" "selection.methods=$CORE"
digest --tag tccn_campaign --by-point

# ---------------------------------------------------------------- 8. regret
stage 8 "online regret against the offline oracle" \
    "$PY" -m scout_fl.experiments.run_regret --config "$CFG" --seeds 0 1 2 3 4 \
        --rounds $([ "$SMOKE" -eq 1 ] && echo 20 || echo 300) --out runs \
        --override "fl.device=$DEVICE"

# ---------------------------------------------------------------- 9. analysis
run_analysis() {
    mkdir -p "$RESULTS/tables"
    "$PY" -m scout_fl.analysis.collect runs --out "$RESULTS/tables" || true
    "$PY" -m scout_fl.analysis.convergence runs --tag tccn_main   || true
    "$PY" -m scout_fl.analysis.feasibility runs --tag tccn_main --method scout_v2 || true
    "$PY" -m scout_fl.analysis.regret runs --tag regret || true
    "$PY" -m scout_fl.analysis.baseline_overlap --tag tccn_main \
        --out "$RESULTS/tables/baseline_overlap.json" || true
    return 0
}
stage 9 "collect the tables and validate the theory" run_analysis

# ---------------------------------------------------------------- 10. figures
run_figures() {
    mkdir -p "$RESULTS/figures"
    # paperfigs writes into research/paper/figures, which holds the figures the paper
    # actually uses. A smoke run would overwrite them with three-round output, so the
    # originals are put aside first and restored afterwards.
    ( cd research/paper && "$PY" paperfigs.py ) || true
    cp -f research/paper/figures/*.pdf research/paper/figures/*.png "$RESULTS/figures/" 2>/dev/null
    cp -rf research/paper/figures/stats "$RESULTS/figures/" 2>/dev/null
    ls -1 "$RESULTS/figures" | head -30
    return 0
}
stage 10 "render the paper figures" run_figures

finish
