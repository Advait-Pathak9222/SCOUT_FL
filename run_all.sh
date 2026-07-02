#!/usr/bin/env bash
# =====================================================================
# TEMPO-FL / CloakFL campaign entrypoint (design research/RESEARCH_DESIGN_TEMPO_CLOAKFL.md).
#
#   bash run_all.sh                 # full campaign, strict gates (default)
#   bash run_all.sh --resume        # skip completed units, continue
#   bash run_all.sh --smoke         # tiny end-to-end (one unit per experiment type)
#   bash run_all.sh --preflight     # only env check + unit tests + schema report
#   bash run_all.sh --stage train   # run a single stage
#   bash run_all.sh --gates-strict  # (default) failed gate cancels its downstream units
#   bash run_all.sh --gates-soft    # log gate verdicts but run everything
#
# GPU parallelism: NUM_GPUS=<n> (env). n=0 -> CPU/MPS single/worker pool.
# Stages run in cost order: preflight -> smoke -> analytic -> train -> analyze.
# Every stage writes a timestamped log to logs/. Resumable: killed jobs continue.
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
NUM_GPUS="${NUM_GPUS:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
GATES_MODE="strict"
SMOKE=""
RESUME=""
ONLY_STAGE=""
DO_PREFLIGHT_ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE="--smoke"; shift ;;
    --resume) RESUME="1"; shift ;;
    --preflight) DO_PREFLIGHT_ONLY="1"; shift ;;
    --gates-strict) GATES_MODE="strict"; shift ;;
    --gates-soft) GATES_MODE="soft"; shift ;;
    --stage) ONLY_STAGE="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

mkdir -p logs outputs/tempo_cloak analysis
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="logs/run_all_${TS}.log"
ANALYTIC_DIR="outputs/tempo_cloak/analytic"
GATE_VERDICTS="analysis/gate_verdicts.json"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }
plan() { $PY -m scout_fl.experiments.plan $SMOKE "$@"; }

run_stage() {   # run_stage <stage> [exclude_csv] [include_csv]
  local stage="$1"; local exclude="${2:-}"; local include="${3:-}"
  local uids_file="logs/uids_${stage}_${TS}_$$.txt"
  local args=(--list --stage "$stage" --incomplete-only)
  [[ -n "$exclude" ]] && args+=(--exclude-experiments "$exclude")
  [[ -n "$include" ]] && args+=(--experiments "$include")
  plan "${args[@]}" > "$uids_file"
  local n; n=$(wc -l < "$uids_file" | tr -d ' ')
  log "STAGE $stage: $n incomplete unit(s)  (include='${include:-all}' exclude='${exclude:-none}')"
  if [[ "$n" -eq 0 ]]; then log "STAGE $stage: nothing to do"; return 0; fi
  NUM_GPUS="$NUM_GPUS" WORKERS_PER_GPU="$WORKERS_PER_GPU" \
    $PY -m scout_fl.experiments.dispatch --uids-file "$uids_file" \
      --num-gpus "$NUM_GPUS" --workers-per-gpu "$WORKERS_PER_GPU" $SMOKE \
      --log-dir "logs/units_${TS}" 2>&1 | tee -a "$MAIN_LOG"
  return "${PIPESTATUS[0]}"
}

# --------------------------------------------------------------- PREFLIGHT
preflight() {
  log "=== PREFLIGHT: env + unit tests + schema report ==="
  $PY - <<'EOF' 2>&1 | tee -a "$MAIN_LOG"
import importlib, sys
for m in ["numpy","scipy","torch","yaml","pandas","matplotlib"]:
    importlib.import_module(m)
import torch
print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"mps={getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()}")
EOF
  log "--- preflight unit tests (NEES R4, leakage hand-case, M2 invariance, replay) ---"
  $PY -m pytest scout_fl/tests/test_infra_tracker.py scout_fl/tests/test_infra_leakage.py \
     scout_fl/tests/test_infra_dither.py scout_fl/tests/test_infra_replay.py \
     scout_fl/tests/test_cloak_analytic.py scout_fl/tests/test_tempo.py -q 2>&1 | tee -a "$MAIN_LOG"
  local rc="${PIPESTATUS[0]}"
  if [[ "$rc" -ne 0 ]]; then
    log "PREFLIGHT FAILED: unit tests did not pass (design R4 — tracking/leakage invalid). ABORT."
    exit 1
  fi
  log "--- schema report (strict: replay faithfulness gates E-C4/E-T2) ---"
  $PY -m scout_fl.analysis.schema_report --strict 2>&1 | tee -a "$MAIN_LOG"
  if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
    log "PREFLIGHT FAILED: replay verification failed. ABORT."
    exit 1
  fi
  log "PREFLIGHT OK"
}

# --------------------------------------------------------------- BUDGET
budget() {
  log "=== BUDGET: extrapolate wall-clock from smoke ==="
  # measure one smoke train unit, scale to full rounds
  local smoke_uid; smoke_uid=$($PY -m scout_fl.experiments.plan --smoke --list --stage train | head -1)
  if [[ -z "$smoke_uid" ]]; then log "no train units to budget"; return 0; fi
  local t0 t1 secs
  t0=$(date +%s)
  $PY -m scout_fl.experiments.run_unit --uid "$smoke_uid" --smoke --force >/dev/null 2>&1 || true
  t1=$(date +%s); secs=$((t1 - t0))
  # smoke ~5 rounds; full rounds from config -> scale
  local full; full=$($PY -c "import scout_fl.experiments.units as U;c=U.load_campaign_config();print(c['fl']['rounds'])")
  local smoke_r; smoke_r=$($PY -c "import scout_fl.experiments.units as U;c=U.load_campaign_config();print(c['smoke']['rounds'])")
  local per_unit; per_unit=$($PY -c "print(max(1,$secs)*$full/max(1,$smoke_r))")
  log "smoke unit ($smoke_uid): ${secs}s for $smoke_r rounds -> ~${per_unit}s per full-rounds unit"
  plan --budget --per-unit-seconds "$per_unit" --num-gpus "$NUM_GPUS" \
       --workers-per-gpu "$WORKERS_PER_GPU" --stage train 2>&1 | tee -a "$MAIN_LOG"
}

# --------------------------------------------------------------- GATES
eval_gates() {
  log "=== GATES: evaluate 1-3 from analytic + Phase-1 artifacts ==="
  $PY -m scout_fl.analysis.gates --analytic-dir "$ANALYTIC_DIR" --out "$GATE_VERDICTS" \
     2>&1 | tee -a "$MAIN_LOG"
}

# returns a comma list of experiments to EXCLUDE from train (strict-mode cancellation)
gate_exclusions() {
  [[ "$GATES_MODE" != "strict" ]] && { echo ""; return; }
  [[ ! -f "$GATE_VERDICTS" ]] && { echo ""; return; }
  $PY - "$GATE_VERDICTS" <<'EOF'
import json,sys
v=json.load(open(sys.argv[1]))
excl=set()
DOWN={"GATE1":["E-T3","E-T4","E-T5","E-T6"],"GATE2":["E-C3","E-C5"],"GATE3":["E-C3","E-C5"]}
for g,exps in DOWN.items():
    if v.get(g,{}).get("pass") is False:
        excl.update(exps)
print(",".join(sorted(excl)))
EOF
}

# --------------------------------------------------------------- ANALYZE
analyze() {
  log "=== ANALYZE: collect + figures + stats + decision_summary ==="
  $PY -m scout_fl.analysis.collect --tag tempo  2>&1 | tee -a "$MAIN_LOG" || true
  $PY -m scout_fl.analysis.collect --tag cloak  2>&1 | tee -a "$MAIN_LOG" || true
  $PY -m scout_fl.analysis.figures_tc 2>&1 | tee -a "$MAIN_LOG" || true
  $PY -m scout_fl.analysis.decision $SMOKE 2>&1 | tee -a "$MAIN_LOG"
  log "ANALYZE done -> analysis/decision_summary.md"
}

# =====================================================================
log "run_all start (gates=$GATES_MODE, NUM_GPUS=$NUM_GPUS, smoke='${SMOKE:-no}', resume='${RESUME:-no}')"
preflight
[[ -n "$DO_PREFLIGHT_ONLY" ]] && { log "preflight-only: done"; exit 0; }

if [[ -z "$ONLY_STAGE" || "$ONLY_STAGE" == "smoke" ]]; then
  log "=== SMOKE: one tiny unit per experiment type ==="
  smoke_uids="logs/uids_smoke_${TS}.txt"
  $PY -m scout_fl.experiments.plan --smoke --list --incomplete-only > "$smoke_uids" || true
  NUM_GPUS="$NUM_GPUS" $PY -m scout_fl.experiments.dispatch --uids-file "$smoke_uids" \
     --num-gpus "$NUM_GPUS" --smoke --log-dir "logs/units_${TS}" 2>&1 | tee -a "$MAIN_LOG" || \
     { log "SMOKE FAILED — see logs/failed_units.txt"; [[ -z "$ONLY_STAGE" ]] && exit 1; }
  log "SMOKE OK"
fi

[[ -n "$SMOKE" && -z "$ONLY_STAGE" ]] && { analyze; log "smoke run complete"; exit 0; }

if [[ -z "$ONLY_STAGE" ]]; then
  budget
  run_stage analytic || log "WARN: some analytic units failed (continuing)"
  eval_gates                                  # GATE 2/3 from analytic; GATE 1 still pending
  # Phase 1: run the GATE-1 units (E-T1 oracle schedules) FIRST, then re-evaluate gates,
  # so strict-mode cancellation of the Phase-2/3 TEMPO units (design §3.2) is honored.
  run_stage train "" "E-T1" || log "WARN: some E-T1 units failed"
  eval_gates                                  # GATE 1 now resolves
  EXCL="$(gate_exclusions)"
  [[ -n "$EXCL" ]] && log "GATES strict: cancelling downstream experiments: $EXCL"
  # Phase 2/3: everything else (E-T1 already complete -> skipped by --incomplete-only)
  run_stage train "$EXCL" || log "WARN: some train units failed (see logs/failed_units.txt)"
  eval_gates
  analyze
else
  case "$ONLY_STAGE" in
    preflight) : ;;  # already ran
    analytic)  run_stage analytic; eval_gates ;;
    train)     eval_gates; run_stage train "$(gate_exclusions)" ;;
    analyze)   analyze ;;
    *) log "unknown stage $ONLY_STAGE"; exit 2 ;;
  esac
fi
log "run_all complete. See analysis/decision_summary.md"
