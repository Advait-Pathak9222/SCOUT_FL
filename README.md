# SCOUT-FL

**Sensing-Coverage and Uncertainty-Aware Federated Client Selection for ISAC Networks**

Research codebase targeting *IEEE Transactions on Wireless Communications*. Full
research plan: [`~/.claude/plans/i-am-starting-a-joyful-ullman.md`](../.claude/plans/i-am-starting-a-joyful-ullman.md)
(also `research/SCOUT-FL_Research_Plan.docx`).

**Core idea.** Select FL clients by their *marginal contribution to a multi-view
sensing mission* — log-det Fisher-information gain, region freshness/coverage,
fairness, and OTA aggregation reliability — not by raw sensing SNR. Two
high-SNR clients viewing a target from the same angle give redundant
information; a medium-SNR client at a complementary angle reduces localization
uncertainty far more. SCOUT-FL formalizes this as a monotone-submodular
selection problem with a `(1 − 1/e)` greedy guarantee.

---

## Quickstart

```bash
pip install -r requirements.txt

# Milestone-1 gate — prove "high-SNR != high sensing value" (one command, no edits):
python -m scout_fl.experiments.run_microbenchmark --config scout_fl/configs/microbenchmark.yaml

# Override any field without touching files:
python -m scout_fl.experiments.run_microbenchmark --override selection.budget=2 seed=1

# Tests:
pytest scout_fl/tests -q
```

Every experiment is a single command: `python -m scout_fl.experiments.<name> --config <path> [--override k=v ...]`.
Each run writes a self-describing folder `outputs/<experiment>/<run_id>/`
(resolved config, metrics CSV/JSON, plots).

---

## Stages (build target)

- **A1-Full** — the main method, **FL included**: federated client selection +
  local training + FedAvg/OTA aggregation + learning/sensing/coverage/fairness
  utilities + CRB & AirComp-MSE & latency/energy constraints + baselines +
  ablations + convergence & sensing metrics. This is what we build and evaluate
  deeply. The microbenchmark is only proof-of-concept.
- **A2** — a *small* resource-allocation add-on **inside A1** (closed-form/convex
  power/equalizer/bandwidth + AirComp MSE feasibility). Supporting module, not a
  separate experiment family.
- **A3** — *optional*, small online/freshness extension **after A1 is strong**
  (dynamic region coverage, freshness/AoI, maybe a light online variant). Not a
  heavy DRL/bandit system.

## Implementation order (staged; do NOT skip ahead)

The #1 risk is **scope explosion**, not novelty. Keep v1 clean and TWC-style.
Steps 5–8 below are all part of building **A1-Full**; A2 folds into Steps 6–7;
A3 is Step 11.

1. **[done]** Repo skeleton, config system, logging, tests.
2. **[done]** Geometry, FIM, log-det, CRB, **microbenchmark**.
3. **[done]** SCOUT-FL greedy selection (sensing-only, no FL yet).
4. **[done]** Sensing-only baselines (random / SNR / CRB) + submodularity check.
5. **[done]** Coverage/freshness map + fairness utility + composite `TotalUtility`; multi-round loop (`run_synthetic`, no FL yet).
6. **[done]** AirComp aggregation MSE + comm channels + energy/latency + constraint-integrated greedy (feasibility gate + relax-and-log); **A2** demo (`run_aircomp`).
7. **[in progress]** FL pipeline: **datasets (MNIST/Fashion-MNIST) + IID/Dirichlet partitioning [done]**; models, client/server, FedAvg + OTA distortion, and the federated training loop [next].
8. **[next]** **Asaad-style MSE+CRB scheduling baseline** (non-negotiable).
9. Small synthetic experiments (`run_synthetic`, `run_baselines`, `run_ablations`).
10. Semi-real data: **DeepSense 6G** (primary), WiMANS (secondary). *Not* MNIST/CIFAR as the main result.
11. Only after A1 is strong: online A3 (bandit/Lyapunov) and the multi-cell extension.

**Milestone-1 definition of done:** microbenchmark passes the gate, greedy
selection implemented, `f_sense` submodularity numerically verified, sensing
baselines implemented, selection runtime logged, and complementary-angle
selection beats redundant high-SNR selection on CRB. Only then start FL.

---

## Repository map & module responsibilities

```
scout_fl/
  configs/            YAML configs (single source of truth)
    microbenchmark.yaml      1-target / 3-client gate
    synthetic_small.yaml     K=20 dev scenario (all later-step fields documented)
    synthetic_main.yaml      (Step 9) full-scale scenario
  sim/                Synthetic ISAC simulator
    geometry.py       pairwise_geometry(clients,targets) -> {range,u,v,bearing}; sample_positions()
    fim.py            per_client_target_fim(geom,snr,k_range,k_angle); prior_fim(); db_to_linear()
    crb.py            logdet_spd() [D-optimal]; crb_trace() [A-optimal]; accumulate()
    channel.py        (Step 6) Rayleigh/Rician + path loss -> h_k, sensing SNR
    aircomp.py        (Step 6) AirComp aggregation MSE, power/equalizer allocation
    energy_latency.py (Step 6) per-client sense/compute/tx energy + latency
    mobility.py       (Step 11) target/region dynamics for the tracking extension
  objectives/         Selection utility terms
    sensing_utility.py   SensingUtility: value()/crb()/rmse() + init_state()/add()/marginal_gain()
    coverage_utility.py  (Step 5) dynamic region freshness map U_r(t+1)=rho*U_r+xi-contrib
    fairness_utility.py  (Step 5) client/region age-since-selection (submodular saturating)
    learning_utility.py  (Step 7) DivFL facility-location over gradients/representations
    constraints.py       (Step 6) CRB/MSE/latency/energy feasibility + relax_and_log policy
    total_utility.py     (Step 5/6) composite monotone-submodular utility + weights
  selection/          Selection policies (shared Selector interface -> SelectionResult)
    base.py           Selector ABC; SelectionResult(selected, select_time, info)
    lazy_greedy.py    lazy_greedy() — CELF for monotone submodular maximization
    scout_greedy.py   ScoutGreedy (A1); naive_greedy() reference
    random.py snr_based.py crb_based.py            implemented baselines
    loss_based.py grad_norm.py channel_aware.py    (Step 7) FL baselines
    divfl_like.py oort_like.py                      (Step 7) FL baselines
    asaad_style.py    (Step 8) Asaad-style MSE+CRB step-wise device dropping
  fl/                 (Step 7) datasets, partitioning, models, client, server, aggregation, training
  analysis/
    verify_submodularity.py  verify_submodular(value_fn, ground_set, n_samples, rng)
    check_units.py    (Step 5) print ranges of every utility/constraint term
    plot_results.py statistical_tests.py  (Step 9)
  experiments/        one command each
    run_microbenchmark.py    build_utility(cfg) + main()
    run_synthetic.py run_baselines.py run_ablations.py run_sweeps.py   (Step 9)
  utils/
    config.py         load_config(path, overrides); Config (attr access); to_plain()
    seed.py           seed_everything(seed) -> np.random.Generator; make_rng()
    logging_utils.py  RunLogger(base_dir, experiment, seed, config): save_json/log_row/save_csv
  tests/              pytest (test_sensing.py: FIM PSD, logdet monotonicity, CRB, gate, submod, repro)
outputs/              per-run artifacts (gitignored in practice)
```

### Efficiency contract (designed in from the start)
- Per-client FIMs are **cached** as a `(K, M, d, d)` array; selection never rebuilds them.
- `SensingUtility` keeps an **incremental accumulated-FIM state**; `marginal_gain` is `O(M·d³)`.
- **Lazy-greedy (CELF)** avoids recomputing most marginals; use it for `K > ~50`.
- All sensing/selection math is **vectorized NumPy** (no Python loops over clients in the hot path).
- CRB (matrix inverse) is computed only for constraints/evaluation, not inside the log-det objective.
- Selection / sensing-eval / (later) training / aggregation **times are logged per round**.

---

## Config field reference

### `microbenchmark.yaml`
`experiment`, `seed`, `output_dir`; `geometry.{bs_position, targets[[x,y]], clients[[x,y]]}`;
`sensing.{snr_db[K], k_range, k_angle, prior_fim, target_weights[M]}`;
`selection.budget`; `verify.submodularity_samples`; `logging.save_plots`.

### `synthetic_small.yaml`
Adds `rounds`; `network.{num_clients, budget, num_targets, num_regions, area_size}`;
`geometry.{random_clients, random_targets}`; `channel.*` (Step 6);
`sensing.{tx_power_dbm, rcs_mean, rcs_std, ...}`; `coverage.*` (Step 5);
`fairness.*` (Step 5); `objectives.{alpha_learning, lambda_sense, lambda_coverage, lambda_fairness}`;
`constraints.{crb_max, mse_agg_max, latency_max, energy_max, power_budget, infeasible_policy}`;
`aircomp.*` (Step 6); `selection.{method, use_lazy_greedy}`; `fl.*` (Step 7);
`mobility.*` (Step 11); `verify.submodularity_samples`; `logging.*`.

**λ strategy (primary).** Sensing/latency/energy are **hard constraints**, not
weighted penalties (cleaner for TWC, avoids "hand-tuned weights"). Any surviving
soft weight is normalized and swept (Pareto knee as the default operating point).
`constraints.infeasible_policy = relax_and_log` — never silently drop a constraint.

---

## Sensing model (one paragraph)

Each client's per-target position FIM is `J = a_r·uuᵀ + a_a·vvᵀ` in the (x,y)
frame, where `u` is the radial (range) direction, `v` the tangential
(angle/cross-range) direction, `a_r = γ·k_range`, `a_a = γ·k_angle/range²`, and
`γ` is linear sensing SNR. Accumulated information is `J_m(S)=J₀+Σ_{k∈S}J_{k,m}`;
the selection objective is `f_sense(S)=Σ_m w_m[logdet J_m(S) − logdet J₀]`
(D-optimal, submodular), and CRB `=Σ_m w_m·tr(J_m(S)⁻¹)` (A-optimal) is the
constraint/evaluation metric. Anisotropy + geometry are what make angular
diversity beat raw SNR.
