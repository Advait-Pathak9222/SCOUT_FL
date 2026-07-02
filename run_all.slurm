#!/usr/bin/env bash
#SBATCH --job-name=tempo_cloak
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.out
#SBATCH --array=0-63%16          # 64 shards, <=16 concurrent (tune to the cluster/queue)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
# =====================================================================
# SLURM array wrapper (OPTIONAL — the plain `bash run_all.sh` path works standalone).
# Each array task runs a 1/N shard of the incomplete TRAIN units on its own GPU;
# resumability makes shards independent and safe to re-queue. Analytic + preflight +
# analyze run once (on task 0) since they are cheap and shared.
#
#   sbatch run_all.slurm
#   # then, after the array finishes, gates+analysis run automatically on task 0's
#   # post-step, or run:  bash run_all.sh --stage analyze
# =====================================================================
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PY="${PYTHON:-python3}"
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-64}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
export NUM_GPUS=1                     # one GPU per array task
TS="$(date +%Y%m%d_%H%M%S)_shard${SHARD}"

# Task 0 runs preflight + analytic + gates once (cheap, shared).
if [[ "$SHARD" == "0" ]]; then
  bash run_all.sh --preflight
  $PY -m scout_fl.experiments.plan --list --stage analytic --incomplete-only > "logs/uids_analytic_${TS}.txt"
  $PY -m scout_fl.experiments.dispatch --uids-file "logs/uids_analytic_${TS}.txt" --num-gpus 1 \
     --log-dir "logs/units_${TS}"
  $PY -m scout_fl.analysis.gates --out analysis/gate_verdicts.json || true
fi

# All tasks: shard the incomplete train units deterministically and run this shard.
EXCL="$($PY - analysis/gate_verdicts.json <<'EOF'
import json,sys,os
p=sys.argv[1]
if not os.path.exists(p): print(""); sys.exit()
v=json.load(open(p)); excl=set()
DOWN={"GATE1":["E-T3","E-T4","E-T5","E-T6"],"GATE2":["E-C3","E-C5","E-C6"],"GATE3":["E-C3","E-C5","E-C6"]}
for g,e in DOWN.items():
    if v.get(g,{}).get("pass") is False: excl.update(e)
print(",".join(sorted(excl)))
EOF
)"
ARGS=(--list --stage train --incomplete-only)
[[ -n "$EXCL" ]] && ARGS+=(--exclude-experiments "$EXCL")
$PY -m scout_fl.experiments.plan "${ARGS[@]}" | awk "NR % $N_SHARDS == $SHARD" > "logs/uids_train_${TS}.txt"
$PY -m scout_fl.experiments.dispatch --uids-file "logs/uids_train_${TS}.txt" --num-gpus 1 \
   --log-dir "logs/units_${TS}"

# Task 0 also runs the final analysis once the array is (re-)run to completion.
if [[ "$SHARD" == "0" ]]; then
  bash run_all.sh --stage analyze
fi
