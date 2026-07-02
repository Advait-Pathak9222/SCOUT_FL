# Research Design: TEMPO-FL and CloakFL
## Two new methods for sensing-aware federated learning in ISAC networks

**Status:** Design document v1.0 — pre-implementation
**Target venue:** IEEE Transactions on Wireless Communications (TWC); CloakFL secondary venue: IEEE TIFS
**Context:** Successor directions to the SCOUT-FL / JEDI project. The v2 audit established that
subset-selection methods are trapped on a crowded learning–sensing Pareto frontier
(SCOUT-FL v2 statistically tied with CollabSenseFed; margins ~2 pp). Both methods below
change the problem structure rather than the selection score.

---

# Part 0 — Shared context and infrastructure

## 0.1 Existing assets (reuse, do not rebuild)

| Asset | Location / convention | Reused by |
|---|---|---|
| FL training harness | N=100, K=10, 150 rounds, 5 datasets, small-CNN/MLP | Both |
| Channel model | Rician/Rayleigh, 3.5 GHz link budget, AirComp OTA-FedAvg, MSE budget 1e-3 | Both |
| Per-target FIM machinery | J_{k,m} = a_r uu^T + a_a vv^T; log-det utility; CRB (A-opt) | Both |
| 28 methods | 25 baselines + SCOUT-FL v1/v2 + JEDI (runs/ artifacts) | Both (as baselines) |
| Metric collection | analysis/collect.py; final-round mean over seeds | Both |
| Audit tooling | Paired-diff CIs, threshold sweep, dominance counts, corrected CD | Both |
| SCOUT-FL v2 greedy + primal–dual | Selection engine | TEMPO (inner loop), CloakFL (base selector) |

## 0.2 New shared infrastructure (build once, Phase 0)

1. **Target mobility model.** Constant-velocity (CV) Gaussian random-walk:
   state x_m = [pos, vel], process noise covariance Q(sigma_p). Sweepable
   sigma_p in {0, 0.01, 0.02, 0.05, 0.1, 0.2} (units consistent with the existing
   arena scale; sigma_p = 0.05 matches the single existing mobility point so old
   artifacts can be sanity-checked against new code).
2. **Bayesian tracker.** Per-target Kalman filter: predict with Q; update by adding the
   selected set's FIM increment J_m(S_t) as measurement information
   (information-filter form: P_t^-1 = (P_{t|t-1})^-1 + J_m(S_t)).
   Outputs per-round posterior covariance P_{t,m} and tracking metrics.
3. **Ground-truth trajectory logger** so tracking RMSE (not just CRB) is measurable.
4. **Client-position leakage accountant** (CloakFL, but instrumented globally):
   per-round client-position FIM J_k^leak accumulated per client per campaign
   (same rank-structured FIM code as targets, pointed at clients).
5. **Seed policy.** 10 seeds for headline comparisons; 5 seeds acceptable for sweeps
   and ablations. All paired tests on shared seeds.

## 0.3 Evaluation standards (lessons from the v2 audit — non-negotiable)

- **Final-round CRB is the primary sensing convention.** Round-mean reported as secondary.
  (JEDI's 91%→27% Pareto collapse under convention switch must never recur.)
- **Weight-free evidence first:** Pareto dominance counts, accuracy at matched sensing
  WITH threshold sweeps (never a single tau), paired per-point CIs vs the strongest
  baseline (CollabSenseFed, Sensing-Native OTA-FL — never Asaad-only).
- Any equal-weight joint score is illustrative only and labelled as such.
- CD diagrams carry the correlation caveat (shared seeds/system) and are illustrative.
- Missing/partial runs are reported as missing; never imputed or silently dropped.
- Every experiment has a pre-registered decision gate written BEFORE the run.

---

# Part 1 — TEMPO-FL: Temporal Mission Planning for ISAC-FL

## 1.1 Thesis

Every existing method (all 28) is a stationary, myopic per-round policy, so learning
and sensing fight for K slots every round. But the objectives have opposite temporal
structure: learning utility is front-loaded (per-round descent scales with ||grad F||^2,
which decays; critical early periods), while sensing utility under a tracking mission is
a maintenance process (posterior information persists, decaying at a rate set by target
process noise Q). Therefore a **time-varying schedule** of the learning/sensing balance
can strictly dominate the entire stationary Pareto frontier.

**Headline claim to test:** there exists a schedule lambda_s(t) whose (final accuracy,
time-averaged tracking error) point Pareto-dominates every stationary policy measured
in the existing campaign.

## 1.2 Problem formulation

- Learning state: L_t = smoothed estimate of ||grad F(w_t)||^2 (proxy: mean squared
  norm of received aggregated update; fallback: loss decrement EMA).
- Sensing state: per-target posterior covariance P_{t,m} from the Kalman tracker.
- Control: mixture weight lambda_s(t) in [0,1] fed to the inner selector; per-round
  utility U_t(S) = (1-lambda_s(t)) f_learn(S) + lambda_s(t) f_sense(S), subject to
  |S| <= K and AirComp-MSE budget (primal–dual, inherited from SCOUT-FL v2).
- Mission objective (two canonical forms, both evaluated):
  (a) Terminal: max accuracy(T) s.t. tr(P_{T,m}) <= P_max for all m.
  (b) Sustained: max accuracy(T) s.t. (1/T) * sum_t tr(P_{t,m}) <= P_max
      (track maintained throughout — the interesting case under mobility).

## 1.3 Method variants (all three implemented; they are also each other's ablations)

1. **TEMPO-Threshold.** Closed-form switch time tau*: pure learning (lambda_s=0) for
   t < tau*, sensing-weighted after. tau* derived from the descent-bound constants,
   estimated gradient-decay rate, and per-round FIM increments (stationary-target case).
   Simple, provable, the "teachable" variant.
2. **TEMPO-DPP (drift-plus-penalty).** Virtual queue Z_{t,m} tracking violation of
   tr(P_{t,m}) <= P_max; per-round selection maximizes
   f_learn(S) + (1/V) * sum_m Z_{t,m} * dI_m(S) where dI_m(S) is the FIM increment.
   No horizon knowledge needed; Lyapunov-style guarantee target (constraint satisfied
   within O(1/V), learning loss within O(V)). Natural sibling of SCOUT-FL v2's
   primal–dual machinery.
3. **TEMPO-MPC.** Receding horizon H in {5, 10, 20}: plan lambda_s(t..t+H) against the
   coupled dynamics (descent inequality forward model + Riccati recursion), execute the
   first step, replan. Upper-performance reference among the variants.

**Not a linear combination because:** a constant lambda_s recovers exactly the existing
composite-utility methods; TEMPO's contribution is the *trajectory* lambda_s(t) chosen
by a controller against explicit two-state dynamics. Stationary policies are the
degenerate special case, and the dominance experiments treat them as the null.

## 1.4 Theory targets (paper Sections IV–V)

- **T1 (Threshold structure).** Stationary targets + terminal constraint: the optimal
  schedule is threshold-type; closed-form tau*(L-smoothness, gradient-decay rate,
  FIM-increment rate). Proof sketch: exchange argument — a sensing round moved later
  never reduces terminal FIM (information is additive and time-fungible when Q=0),
  while a learning round moved earlier weakly increases descent (decaying marginal
  utility). Validate empirically (E4 vs E1 oracle grid).
- **T2 (Mission duty cycle).** Mobile targets (Q > 0): optimal policy is periodic
  bursting; burst period p* scales as a function of Q and per-burst FIM injection —
  a "Nyquist rate of the sensing mission" formula linking a radar quantity to rounds
  sacrificed from learning. Derive for scalar case; verify by simulation for the 2-D case.
- **T3 (Dominance).** Conditions under which the schedule's achievable
  (accuracy, avg tracking error) strictly dominates every stationary lambda_s.
  Even a partial/conditions-based result converts the measured static frontier into a
  provable lower bound that the method escapes.
- **T4 (Online regret — upgrades the pending P7 study).** TEMPO-DPP regret against a
  *dynamic* oracle (best schedule in hindsight), replacing the CUCB-vs-static-oracle
  design. Standard drift-plus-penalty analysis adapted to the coupled system.

## 1.5 Experiments

### E-T1 — Oracle-schedule kill test (GATE 1; existing pipeline + tracker only)
- Hand-crafted schedules, no controller:
  - Learn-then-sense: switch at tau in {30, 50, 75, 100, 120}.
  - Sense-then-learn (control condition — expected to be bad; validates T1's direction).
  - Bursting: burst length b in {3, 5, 10} x period p in {15, 25, 50}.
  - Inner selector during learning phase: DivFL-style / f_learn-greedy; during sensing
    phase: existing CRB-only / D-opt selector. (Both already exist as baselines.)
- Settings: CIFAR-10 main point (alpha=0.3), sigma_p in {0, 0.05}; 5 seeds; 150 rounds.
- Metrics: final accuracy; time-averaged and worst-20-round-window tr(P); terminal CRB.
- **GATE 1 (go/no-go):** at least one schedule dominates the measured static frontier by
  >= 1.0 pp accuracy at matched (or better) time-averaged tracking error, with the
  paired 95% CI excluding zero. If no schedule clears this, TEMPO is dead in this
  setting — stop and report.

### E-T2 — Premise checks (run concurrently with E-T1)
- Gradient-norm decay curves at alpha in {0.1, 0.3, 0.5} (does L_t actually decay under
  heavy non-IID? If flat at alpha=0.1, T1's premise fails there — scope claims).
- Tracker sanity: tracking RMSE vs posterior CRB (filter consistency / NEES test).

### E-T3 — Mobility sweep (new axis; also rescues the abandoned JEDI-synergy thread)
- sigma_p in {0, 0.01, 0.02, 0.05, 0.1, 0.2} x {best static baselines, best oracle
  schedule from E-T1}; 5 seeds. Question: where does temporal structure matter?
  Expected shape: at sigma_p=0 the switch policy wins (sense late); as sigma_p grows,
  bursting wins; at extreme sigma_p everything degrades (track unmaintainable) —
  this regime map is a headline figure.

### E-T4 — Main bake-off (only if GATE 1 passes)
- TEMPO-Threshold / -DPP / -MPC vs baselines (below) on CIFAR-10, EMNIST, UCI-HAR;
  10 seeds; 150 rounds; sigma_p in {0, 0.05, 0.1}; both mission forms (terminal,
  sustained).
- Primary evidence: (i) trajectory-frontier plot — accuracy vs time-avg tracking error,
  TEMPO points vs the full static cloud; (ii) per-point strict-dominance counts;
  (iii) paired CIs vs CollabSenseFed, Sensing-Native, and the best E-T1 oracle schedule;
  (iv) accuracy at matched tracking with tracking-bar sweep (audit-style).

### E-T5 — Ablations
- Controller pieces: MPC horizon H; DPP parameter V; constraint level P_max sweep
  (traces the new frontier); mis-specified Q (controller believes 2x/0.5x true sigma_p);
  noisy L_t estimate (robustness of the switch time).
- Inner-selector swap: TEMPO wrapped around SCOUT-FL v2 vs around plain greedy —
  shows the schedule (not the inner selector) carries the gain.

### E-T6 — Online-regret study (absorbs and upgrades pending P7)
- TEMPO-DPP vs dynamic oracle (best schedule in hindsight, computed offline from E-T1
  grid + local search), 300 rounds, 2 configurations. Report empirical regret curve
  against the T4 bound.

## 1.6 Baselines

| Family | Methods | Source |
|---|---|---|
| Stationary frontier (the null) | All 28 existing methods, unchanged | runs/ artifacts (re-scored with tracker where trajectories permit; else re-run top-9) |
| Naive schedules (anti-triviality controls) | Round-robin alternation (1:1); random lambda_s(t); linear anneal 0->1; two-phase 50/50 fixed split | New, trivial to implement |
| Tuned-static | Best constant lambda_s found by grid search (the strongest fair static competitor) | New |
| Oracle schedule | Best hand schedule from E-T1 (upper reference for controllers) | E-T1 |

The naive-schedule controls matter: if random schedules also beat the static frontier,
the story is "any time variation helps" (weaker claim); the controllers must beat the
naive schedules, not just the static methods.

## 1.7 Metrics
Final accuracy; time-averaged tr(P) and worst-window tr(P); terminal final-round CRB;
% rounds constraint violated; tracking RMSE vs ground truth; energy/round;
selection-time overhead. Fairness (Jain) retained for continuity with the old campaign.

## 1.8 Risks and pre-registered failure modes
- R1: Gradient norm non-decaying at alpha=0.1 -> scope T1 to moderate non-IID.
- R2: Dominance gap < 1 pp -> direction dead here; publish nothing (or a short
  negative-result note); pivot budget to CloakFL/GradEcho.
- R3: "Any schedule works" (naive controls also dominate) -> reframe contribution
  around T2/T3 theory + regime map, claims scoped accordingly.
- R4: Tracker inconsistency (NEES fails) -> fix filter before any comparison; tracking
  claims are meaningless otherwise.

---

# Part 2 — CloakFL: Location-Private Client Participation in ISAC-FL

## 2.1 Thesis

An ISAC-FL selection mechanism is a localization engine pointed at its own participants:
the BS must know, and every round refines, each client's range/bearing to compute its
sensing utility. Clients join FL for privacy; the physical layer leaks their position by
design, continuously, at radar precision. Additionally, client uplinks act as
illuminators that any external passive receiver can exploit. No prior work addresses
location privacy inside the ISAC-FL selection loop (the 2026 ISAC-privacy roadmap lists
it as open; FL-privacy work covers model updates only).

**Headline claims to test:** (1) selection can retain most target-sensing utility under
a hard cap on client-position information leaked to the BS ("sense the target, not the
sensor"); (2) zero-sum correlated dithering makes individual client waveforms useless to
an eavesdropper at exactly zero cost to the AirComp aggregate.

## 2.2 Threat model (state on page 1 of the paper)

- **A1 — honest-but-curious BS.** Follows the protocol; accumulates client-position
  Fisher information from returns/CSI/selection side-info. Assumed to already hold
  coarse registration-level location (cell-level, ~100 m); the protected quantity is
  *refinement* to radar precision and *continuous tracking*. This pre-empts the
  "BS already knows where clients are" objection.
- **A2 — external passive eavesdropper(s).** Receives client uplinks; attempts per-client
  matched filtering / bistatic localization of clients (and free-rides on target
  sensing). Knows the protocol but not the shared dither seeds. Extension: 2–3
  colluding receivers.
- Out of scope (declared): malicious BS deviating from protocol; gradient-content
  privacy (delegated to standard SecAgg/DP, orthogonal and composable).

## 2.3 Privacy metric

- Per-round client-position FIM leakage J_k^leak(t) (same rank-structured FIM code as
  targets, evaluated at client k's position); campaign accumulation
  J_k^leak(1..T) = sum_t [selected or otherwise observed contributions].
- Operational privacy level: client-position CRB floor
  r_k = sqrt(tr(J_k^leak(1..T)^-1)) — "the BS cannot localize client k better than
  r_k meters after T rounds." Report median and minimum (worst client) r_k.
- **Composition property (theory T-C3):** leakage adds in Fisher units across rounds —
  gives a per-campaign privacy budget and a DP-style accounting analogy at the
  physical layer.

## 2.4 Method components

1. **M1 — Leakage-capped selection.** Maximize sum_m w_m logdet(J_m(S)) subject to
   |S| <= K, AirComp-MSE budget, and per-client cumulative leakage caps
   J_k^leak(1..t) <= J_max for all k. Implementation: SCOUT-FL greedy with a per-client
   leakage budget (knapsack-like feasibility) + primal–dual handling of the MSE budget
   (all machinery exists). Side effect: caps force participation rotation ->
   interacts with fairness (measure it; likely improves Jain).
2. **M2 — Aggregate-transparent (zero-sum) dithering vs A2.** Selected clients apply
   correlated perturbations delta_k (phase/amplitude/timing dithers) generated from
   pairwise shared seeds with sum_k delta_k = 0 (SecAgg-style pairwise cancellation, but
   analog and at the physical layer). The AirComp aggregate — hence the FL update and
   epsilon_agg — is exactly unchanged; each individual waveform is scrambled, inflating
   A2's per-client CRB. Dither variance sigma_d^2 is the knob.
3. **M3 — Calibrated geometry obfuscation vs A1.** Bounded timing/power jitter and/or
   quantized reported geometry, trading a controlled epsilon_agg increase for reduced
   J^leak. This is the lossy knob (M2 is the lossless one); the M3 trade-off curve is
   itself a result.

## 2.5 Theory targets

- **T-C1 (Sense-the-target-not-the-sensor region).** Characterize the achievable
  (target log-det FIM, client leakage FIM) frontier. Core question: geometric
  separability vs entanglement — target and client information flow through the same
  bistatic geometry; derive conditions (relative bearings/ranges) under which target
  information is extractable with bounded client leakage. Start analytically with
  2 clients / 1 target (paper-and-pencil; the kill test).
  **Fallback that still publishes:** if entanglement is severe in generic geometry, the
  result becomes an impossibility theorem — "ISAC-FL cannot achieve target utility X
  and location privacy Y simultaneously" — arguably the stronger paper.
- **T-C2 (Zero-sum invariance + eavesdropper bound).** (a) Exact aggregate invariance of
  M2 under perfect sync; residual-error bound under sync offset sigma_sync.
  (b) Lower bound on A2's per-client localization CRB as a function of sigma_d^2 and
  the number of colluding receivers.
- **T-C3 (Fisher composition / privacy budget).** Additive leakage accounting across
  rounds; budget-optimal rotation schedules; formal analogy (and disanalogy) to DP
  composition, stated carefully.

## 2.6 Experiments

### E-C1 — Entanglement kill test (GATE 2; analytical + small numeric, no FL training)
- 2 clients / 1 target: sweep geometries (relative bearing 0..180 deg, range ratios);
  compute the exact (target FIM, leakage FIM) frontier per geometry. Then Monte-Carlo
  with N=100 random layouts at the campaign's arena scale.
- **GATE 2:** at leakage caps giving a >= 10 m client CRB floor (vs ~sub-meter
  uncapped), generic geometry retains >= 50% of unconstrained target log-det.
  Pass -> constructive paper. Fail -> pivot to impossibility framing (T-C1 fallback);
  E-C3/E-C4 shrink accordingly.

### E-C2 — Zero-sum dither validation (waveform/aggregation sim; no FL training needed
  for (a)–(b), one short FL run for (c))
- (a) Aggregate invariance: exact under perfect sync; residual epsilon_agg inflation vs
  sync error sigma_sync sweep (this is the honesty experiment — imperfect sync breaks
  exact cancellation and the residual must be quantified).
- (b) Eavesdropper CRB inflation vs sigma_d^2, for 1 and 3 receivers.
- (c) End-to-end: 30-round FL run with M2 on/off — accuracy delta must be ~0 by
  construction; verify.
- **GATE 3:** >= 10x median eavesdropper CRB inflation at a sigma_d^2 costing < 5%
  target-sensing log-det and < 0.5 pp accuracy (should be ~0). Sync-error tolerance
  documented, not assumed.

### E-C3 — Privacy–utility frontier (the headline experiment; after Gates 2–3)
- Sweep J_max over ~6 levels (from uncapped to near-zero leakage); full 150-round runs,
  CIFAR-10 + EMNIST, 10 seeds, stationary targets (mobility optional extension).
- Plot: target log-det AND accuracy AND final-round CRB vs client CRB floor r
  (median and worst-client). Overlay all baselines as points (they sit at r ~ 0).
- Include M1 alone, M1+M2, M1+M2+M3 stacked — attributes the privacy to components.

### E-C4 — "Every existing method localizes its clients" (measurement contribution;
  cheap and high-impact — runs on EXISTING artifacts where per-round selections were
  logged, else re-run top-9 methods once)
- Instrument the leakage accountant across all 28 existing methods; report client CRB
  floor vs rounds for each. Expected killer figure: every published ISAC-FL selector
  localizes its median client to sub-meter precision within tens of rounds; sensing-
  aggressive methods (Asaad, CRB-only) are the worst offenders. This motivates the
  whole paper and is a standalone empirical finding.

### E-C5 — Composition and rotation
- Long-horizon (300–500 rounds) leakage accumulation under M1 caps vs uncapped;
  rotation-induced fairness effects (Jain, per-client participation CDF — reuse the
  audit's Fig-18 tooling).

### E-C6 — Robustness / adversarial stress
- Sync error (already in E-C2a), imperfect CSI at BS, colluding eavesdroppers (2–3),
  BS exploiting selection side-channel only (no returns) — quantifies how much leakage
  is via geometry-of-selection vs via signals (interesting decomposition either way).

## 2.7 Baselines

No prior ISAC-FL privacy baselines exist (that absence is part of the contribution);
comparisons therefore combine existing utility baselines with naive privacy mechanisms:

| Baseline | Purpose |
|---|---|
| All 28 existing methods, unmodified | Utility upper bounds at zero privacy (r ~ 0); populate E-C4 |
| Random selection | Privacy-agnostic control (no preferential refinement, still leaks via returns) |
| DP-Gaussian on geometry | Naive baseline: Gaussian noise on positions/timing sized by a DP budget; expect worse utility at matched r than M1/M3 |
| Coarse-grid quantization | Report positions on a G-meter grid; classic location-privacy baseline |
| SecAgg-style digital masking only | Protects updates, not the physical layer -> demonstrates the gap this paper fills (r unchanged) |
| Uniform-power transmission | Removes power-based inference only; partial-mitigation control |

## 2.8 Metrics
Client-position CRB floor r (median, worst-client); target log-det and final-round CRB;
accuracy; epsilon_agg; eavesdropper per-client CRB; Jain fairness / participation CDF;
overhead (seed agreement, dither generation).

## 2.9 Risks and pre-registered failure modes
- R1: Severe entanglement (Gate 2 fail) -> impossibility paper (T-C1 fallback);
  E-C4 measurement study still publishes.
- R2: Sync sensitivity kills M2 in practice -> report the tolerance envelope honestly;
  M2 becomes a "tight-sync deployments" contribution; M1/M3 unaffected.
- R3: Threat-model pushback ("BS knows locations anyway") -> pre-empted in Sec 2.2
  (coarse-static vs radar-precision-tracked distinction) and quantified in E-C4.
- R4: Leakage caps starve sensing geometry diversity at small K -> visible in E-C3
  frontier; report the knee honestly.

---

# Part 3 — Program plan, gates, and deliverables

## 3.1 Phasing (kill-tests first; cheap before expensive)

| Phase | Work | Cost | Gate |
|---|---|---|---|
| 0 | Shared infra: mobility model, Kalman tracker (+NEES check), trajectory logger, leakage accountant; schema-verified against runs/ | days | Tracker consistent; accountant reproduces hand-computed 2-client case |
| 1a | TEMPO E-T1 + E-T2 (oracle schedules, premise checks) | ~1 wk compute | GATE 1: >=1 pp dominance |
| 1b | CloakFL E-C1 (entanglement, analytic+numeric) + E-C2 (dither) | ~1 wk, mostly no FL | GATES 2–3 |
| 2 | Whichever passed: TEMPO E-T3/E-T4 or CloakFL E-C3/E-C4 (both if both pass and compute allows; E-C4 is cheap — run it regardless) | 2–4 wks | Audit-standard evidence achieved |
| 3 | Ablations, robustness, theory numerics (E-T5/E-T6, E-C5/E-C6) | 2–3 wks | Paper-ready |

## 3.2 Decision rules (pre-registered)
- Both gates pass -> two papers; TEMPO to TWC, CloakFL to TWC or TIFS depending on
  where T-C1 lands (constructive -> TWC; impossibility-led -> either).
- Only one passes -> full resources there; the failed direction gets a 1-page
  negative-result appendix in the lab notes (not silently discarded).
- Neither passes -> both frontiers were real and immovable in this setting; that
  strengthens the GradEcho motivation (the trade-off must be attacked below the
  selection layer) and the pivot is documented.

## 3.3 Deliverables per experiment
Every experiment produces: (i) figures in the existing make_figures.py style
(two-column, log axes for CRB, CI shading); (ii) a machine-readable stats dump
(JSON/CSV) beside each figure; (iii) an entry in analysis/decision_summary.md stating
the gate, the measured value, and pass/fail — written by the code, not by hand.

## 3.4 Statistics standards (inherited from the v2 audit)
Paired tests on shared seeds with effect sizes and 95% CIs; Wilcoxon signed-rank
primary, paired t secondary; 10 seeds for headline claims; threshold/bar sweeps for any
constrained-win claim; dominance counts as primary weight-free evidence; corrected CD
diagrams carry the independence caveat; final-round CRB primary everywhere.
