# TEMPO-FL / CloakFL — Experiment Campaign

Batch-executable implementation of the full research program in
[research/RESEARCH_DESIGN_TEMPO_CLOAKFL.md](research/RESEARCH_DESIGN_TEMPO_CLOAKFL.md).
Two new sensing-aware FL methods for ISAC networks:

- **TEMPO-FL** — a time-varying schedule λ_s(t) of the learning/sensing balance, chosen
  by a controller (Threshold / DPP / MPC) against explicit two-state dynamics (learning
  descent + Kalman/Riccati tracking). Stationary policies are the degenerate case.
- **CloakFL** — location-private client participation: leakage-capped selection (M1),
  aggregate-transparent zero-sum dithering vs an eavesdropper (M2), and calibrated
  geometry obfuscation vs the base station (M3).

## TL;DR — launch on the cluster

```bash
NUM_GPUS=4 bash run_all.sh              # full campaign, strict gates (default)
NUM_GPUS=4 bash run_all.sh --resume    # continue after a kill (every unit is checkpointed)
# come back to:  analysis/decision_summary.md
```

CPU/MPS or a single box: omit `NUM_GPUS` (defaults to 0 → worker pool, no GPU pinning).
SLURM cluster: `sbatch run_all.slurm` (array job; the plain bash path also works standalone).

## What runs, in cost order (stages)

`run_all.sh` runs these stages; each writes a timestamped log to `logs/`:

1. **preflight** — env check + self-validating unit tests + schema report. **Aborts the
   batch** if the tracker fails the NEES consistency test (design R4), the leakage
   accountant misses the hand-computed 2-client case, the M2 dither is not exactly
   aggregate-invariant, or replay can't reproduce the campaign geometry (E-C4/E-T2 depend
   on it). Run alone with `bash run_all.sh --preflight`.
2. **smoke** — one tiny end-to-end unit per experiment type (~5 rounds, 1 seed) into
   **isolated** roots (`runs_smoke/`, `outputs/tempo_cloak_smoke/`), asserting every
   artifact parses and metrics compute. Never collides with real units.
3. **analytic** (near-zero GPU) — E-C1 (entanglement, GATE 2), E-C2a/b (dither, GATE 3),
   E-C4 (leakage re-scoring of all 32 existing methods), E-T2 (gradient-decay premise),
   and the static-frontier tracker re-scoring for GATE 1. These re-score existing `runs/`
   artifacts and run analytic studies — no FL training.
4. **train** — all FL training units (TEMPO E-T1/E-T3/E-T4/E-T5/E-T6, CloakFL
   E-C2c/E-C3/E-C5), dispatched across `$NUM_GPUS`.
5. **analyze** — collect tables, regenerate every figure (with a JSON/CSV stat dump
   beside each), evaluate gates, and auto-write `analysis/decision_summary.md`.

### Budget printout
Before the train stage, `run_all.sh` times one smoke unit, extrapolates to full rounds,
and prints the train-unit count + estimated wall-clock (also written to the log).

## Gates (pre-registered, computed from artifacts)

| Gate | Experiment | Criterion (design) |
|---|---|---|
| GATE 1 | E-T1 | ≥1 oracle schedule beats the static frontier by ≥1.0 pp accuracy at matched-or-better time-averaged tracking, paired 95% CI excludes 0 |
| GATE 2 | E-C1 | generic geometry retains ≥50% target log-det at a ≥10 m client CRB floor |
| GATE 3 | E-C2 | ≥10× eavesdropper CRB inflation at a σ_d² costing <5% sensing log-det & <0.5 pp accuracy |

**`--gates-strict` (default):** a failed gate cancels that method's remaining Phase-2/3
units (§3.2 decision rules: GATE 1→cancel E-T3/4/5/6; GATE 2 or 3 fail→cancel E-C3/E-C5;
E-C4 always runs). **`--gates-soft`:** log the verdict and run everything.

## Where each design experiment's output lands

| Design | What | Location |
|---|---|---|
| Phase 0 | mobility, tracker (+NEES), trajectory logger, leakage accountant, replay, dither | `scout_fl/infra/`, tests in `scout_fl/tests/test_infra_*.py` |
| E-T1 | oracle-schedule kill test | `runs/tempo/ET1_*/` |
| E-T2 | gradient-decay premise + tracker sanity | `outputs/tempo_cloak/analytic/tempo/et2/` |
| E-T1-static | static frontier re-scored through the tracker (GATE 1 null) | `outputs/tempo_cloak/analytic/tempo/static_frontier/` |
| E-T3 | mobility regime sweep | `runs/tempo/ET3_*/` |
| E-T4 | main controller bake-off | `runs/tempo/ET4_*/` |
| E-T5 | ablations (MPC horizon, DPP V) | `runs/tempo/ET5_*/` |
| E-T6 | online-regret (300 rd) | `runs/tempo/ET6_*/` |
| E-C1 | entanglement kill test (GATE 2) | `outputs/tempo_cloak/analytic/cloak/ec1/` |
| E-C2a/b | dither validation (GATE 3) | `outputs/tempo_cloak/analytic/cloak/ec2/` |
| E-C2c | M2 on/off FL run (acc delta ≈ 0) | `runs/cloak/EC2c_*/` |
| E-C3 | privacy–utility frontier (headline) | `runs/cloak/EC3_*/` |
| E-C4 | "every method localizes its clients" | `outputs/tempo_cloak/analytic/cloak/ec4/` |
| E-C5 | composition & rotation (300 rd) | `runs/cloak/EC5_*/` |
| — | figures + stat dumps | `outputs/tempo_cloak/figures/*.{pdf,png,csv}` |
| — | **verdicts** | `analysis/decision_summary.md`, `analysis/gate_verdicts.json` |
| — | schema discovery | `analysis/schema_report.md` (+ `.json`) |

## Config & scale

[experiments/config.yaml](experiments/config.yaml) sets the scale (seed counts,
rounds, dataset subsets, which experiments run); `scout_fl/experiments/units.py` expands
it into the flat unit grid. Seed policy: **10 seeds** for headline comparisons
(E-T4, E-C3), **5** for sweeps/ablations. Full grid ≈ 2,830 FL trainings + 5 analytic
units; inspect it with:

```bash
python -m scout_fl.experiments.plan --count                       # counts by experiment
python -m scout_fl.experiments.plan --list --stage train | head   # unit ids
python -m scout_fl.experiments.plan --budget --per-unit-seconds 40 --num-gpus 4 --stage train
```

## Resumability

Every unit is one checkpointed JSON (`runs/<tag>/<point>/<method>__seed<seed>.json`,
rewritten each round). A completed unit is skipped; a killed job resumes with
`bash run_all.sh --resume`. Analytic units complete when their marker JSON exists.
Determinism: every seed is threaded through and logged in each artifact's `meta`.

## Run one unit / one stage

```bash
python -m scout_fl.experiments.run_unit --uid "tempo:ET1_cifar10_sp0:oracle_lts_tau50:s0"
python -m scout_fl.experiments.run_unit --uid "cloak:EC3_cifar10:m1__r1:s3"
python -m scout_fl.experiments.run_unit --uid "ec1:E-C1"        # analytic study
bash run_all.sh --stage analytic        # just the near-zero-GPU stage
bash run_all.sh --stage analyze         # regenerate figures + decision_summary
```

## Key results are computed, never hardcoded

`analysis/decision_summary.md` states, for each gate, the pre-registered criterion, the
measured value with CI, and PASS/FAIL — all from artifacts. Missing/failed units are
listed as missing, never imputed (design §0.3). The analytic gates (2, 3) and the E-C4
measurement run locally in seconds, so the summary is meaningful even before the cluster
train stage completes.
