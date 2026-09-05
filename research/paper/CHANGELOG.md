# SCOUT-FL paper — review-response CHANGELOG

Running log of every change: item → what changed → old value → new value.
Started in response to the 6-priority review. Investigation-only items are marked
[DIAG]; edits are marked [EDIT]; items awaiting user sign-off are [BLOCKED].

---

## Priority 1 — low absolute accuracy (46% CIFAR-10 / 16.3% CIFAR-100)

### [DIAG] Is there a data-retrieval / training bug?  → NO.
- Data path is real: `scout_fl/fl/datasets.py` loads CIFAR-10/100, EMNIST, F-MNIST
  via `torchvision.datasets` (not synthetic). Accuracy rises 16%→46% along a clean
  learning curve; a broken pipeline would sit at chance (~10%). Confirmed real.
- Training loop `scout_fl/fl/client.py` is standard: SGD lr=0.05, momentum=0.9,
  1 local epoch, batch 64, CrossEntropy. No bug.

### [DIAG] Has training converged at T=150?  → YES (plateaued).
- scout_v2, CIFAR-10 main point, mean over 5 seeds:
  r1 16.3% · r25 35.8% · r50 40.3% · r75 43.3% · r100 44.0% · r125 46.1% · r149 46.0%
- Last-25-round mean gain: **−0.06 pp** (flat). Best 47.2% @ round 124.
- Conclusion: NOT undertrained-by-horizon. More rounds will not move 46%.

### [DIAG] Why is absolute accuracy low?  → By design, not by bug.
Driven by the deliberately-constrained operating point (config `campaign_main.yaml`):
- Model `SmallCNN`: conv16→conv32→flatten→**single linear head** (no hidden FC). Tiny.
- `subsample_train: 20000` (of 50k) split over N=100 clients → ~200 images/client.
- Budget K=10 of N=100 → **10% participation** per round.
- non-IID Dirichlet α=0.3.
- **Real AirComp aggregation distortion** ON (physical link budget, tx −15 dBm, cell-edge
  tail) — adds genuine aggregation noise every round.
This is a *client-selection* study; the object of interest is the **relative gap between
selection methods at a fixed, converged, constrained operating point**, not peak accuracy.

### Re-run cost (measured: ~1.3 min/run scout, ~2.0 min/run asaad, 150 rounds):
- Full campaign (26 pts × 32 methods × 5 seeds ≈ 4160 runs): ~100 h single-thread /
  ~7–13 h with 8–16× parallelism.
- Main point only (32 methods × 5 seeds = 160 runs): ~4 h single / ~30 min parallel.

### [DIAG] Bigger-model experiment (user chose partial re-run) — REVEALED A DEEPER PROBLEM.
Smoke-tested an enlarged SmallCNN (32-64-128 conv + 256 FC + dropout) on full 50k CIFAR-10:
- small model (original):  46.0% final / 47.2% best — STABLE, flat tail (converged).
- big model, lr=0.05:      36.5% final / 51.1% best — **DIVERGES** (peaks ~r55, then falls).
- big model, lr=0.02:      39.9% final / 50.3% best — noisy plateau ~40%, NOT stable-higher.
Conclusion: enlarging the model does **not** raise absolute accuracy here; the bottleneck is
the FL regime (non-IID α=0.3 + 10% participation + real AirComp aggregation noise), not model
capacity. Bigger nets are harder to stabilize and end up equal-or-worse than the small one.
→ The full bigger-model re-run would produce WORSE, noisier numbers. Not launched. Reverted
the model change to keep code consistent with the existing (valid, converged) small-model data.
→ Escalated to user: recommend pivot to Path (b).

### RECOMMENDATION: **Path (b) — justify the regime** (see reasons above). A longer
horizon cannot help (plateaued); only a bigger model + more data + more participation
would raise absolute accuracy, which *changes the experiment* and needs a full campaign
re-run for no change to the scientific claim (relative ranking). Optional "Path (a-lite)"
if better optics are wanted: `subsample_train: 50000` + add a hidden FC layer, re-run the
main point + cross-dataset + shown sweeps only.  → **AWAITING USER DECISION.**

---

## Priority 2 — baseline provenance  [EDIT pending in paper.tex]

Provenance from the code docstrings (`scout_fl/selection/baselines.py`). Split into
(i) real published sources, (ii) classical design criteria, (iii) constructed composites.

| Baseline | Source (from code) | Citation status |
|---|---|---|
| Asaad | Asaad, Wang, Tabassum, *IEEE TWC* 2025, arXiv:2501.06334 | REAL — add arXiv id |
| OTA-FL-ISCC | Zheng et al., *IEEE ICC Workshops* 2024 | REAL — needs full details verified |
| Fed-ISCC | Du et al., *IEEE IoT-J* 2024 | REAL — needs full details verified |
| ISCC-Air-FEEL | Wen et al., arXiv:2508.15185, 2025 (selection restriction) | REAL — add caveat |
| CRB-Only | A-optimal design (trace inv-FIM) | Classical — cite Godrich'10 / est. theory |
| Sensing-Only | D-optimal design (log-det FIM) | Classical — cite submodular-sensing |
| CollabSenseFed | multi-objective learn+sensing, equal weights | **COMPOSITE — no single paper; say so** |
| Sensing-Native | gradients reused for sensing, learn+sense | **COMPOSITE — no single paper; say so** |
| Fixed-Weighted | hand-set weighted sum (control) | **CONTROL baseline — no paper** |
| FedAvg-ISCC / FedSGD-ISCC | ISCC throughput-based selection; differ in local rule | **COMPOSITE — cite ISCC concept** |

Action: cite the four REAL sources; for composites/controls, state explicitly in the
text that they are our sensible reconstructions of the stated design principle, with the
closest conceptual source — do NOT fabricate a paper. Full bib details for Zheng/Du/Wen
to be confirmed against published versions (annotated from the code's own provenance).

---

## Priority 4 — (1−1/e) theory-practice gap  [EDIT pending; analysis DONE]

### Correction 1: the learning term was mis-stated in the paper.
- OLD (paper Eq. 9): `U_learn(S)=Σ_k g_{k,t}` — **modular** "gradient-norm + data-mass +
  staleness". **This is wrong.**
- NEW (code `objectives/learning_utility.py`): DivFL-style **facility location**
  `f_learn(S)=Σ_j max_{k∈S} sim(j,k)`, with RBF gradient similarity
  `sim(j,k)=exp(−‖g_j−g_k‖²/2σ²)`, σ = median-of-squared-distances heuristic.
  This is **monotone submodular** (Balakrishnan et al., DivFL, ICLR 2022).

### Correction 2: what guarantee actually holds.
- The composite `U = w_L f_learn + w_S f_sense + w_C f_cov + w_F f_fair` is a nonnegative
  weighted sum of **monotone submodular** terms → **monotone submodular** → greedy gives
  `(1−1/e)` for `U`. RIGOROUS.  (Matches `objectives/total_utility.py`.)
- The gap the reviewer flagged is the **MSE term**. Under channel-inversion AirComp,
  `MSE(S)=σ²/(|S|²·P·min_k g_k)` — a **min-based, set-coupled** penalty, **not submodular**.
  So `U_μ = U − μ(MSE−ε)` is **not** guaranteed submodular/monotone once μ>0, and the
  `(1−1/e)` bound does **not** transfer to `U_μ`.
- Honest statement to put in the paper:
  * μ=0 (constraint slack — our main operating point, realized MSE 2×10⁻⁴ ≪ ε=10⁻³):
    `U_μ=U`, greedy is `(1−1/e)`.
  * μ>0: the MSE penalty is not submodular; we do NOT claim `(1−1/e)` for `U_μ`. The MSE
    budget is enforced by the outer primal–dual update (a separate no-regret mechanism),
    and the greedy skips non-positive marginals, keeping selection in the
    monotone-improvement region. Approximation ratio for the penalized objective in the
    binding regime is left as an open constant (deterministic greedy has no constant-factor
    guarantee for non-monotone submodular; the relevant randomized-greedy bound is 1/e,
    Buchbinder et al. 2014).

---

## Priority 5 — statistical rigor  [EDIT pending; re-run optional]

- Bands overlap at the main point (46.0±2.6 vs 44.7±2.2). The *paired* head-to-head over
  operating points is what carries significance, but operating points are **not
  independent** (OFAT sweep around a shared nominal). Add an explicit caveat sentence.
- Optional: raise seed count for the main point + head-to-head methods for tighter CIs
  (+10 seeds × 32 methods × 1 point ≈ 8 h single / ~40 min parallel). → AWAITING DECISION.

---

## Priority 6 — reproducibility  [EDIT pending; values below]

Config block to surface (from `campaign_main.yaml` + code):
- Learning: facility location, RBF gradient sim, σ = median heuristic (auto per round).
- Sensing FIM constants: `k_range κ_r = 1.0`, `k_angle κ_a = 0.05`, `prior_fim λ0 = 1e-3`.
- Composite weights: `α_learn = 1.0`, `λ_sense = 1.0`, `λ_cov = 0.5`, `λ_fair = 0.3`.
- AirComp budget: `mse_agg_max ε = 1e-3`; dual step `dual_lr η = 0.5`.
- FL: N=100, K=10, 150 rounds, lr 0.05, momentum 0.9, 1 local epoch, batch 64,
  subsample_train 20000 / test 4000, Dirichlet α=0.3 (nominal).
- Physical: 3.5 GHz, NF 7 dB, tx −15 dBm, Rician K=6 dB, pathloss exp 3.0.

---

## Figures / template  [EDIT pending]
- Migrate paper to `Template_IEEE/bare_jrnl_new_sample4.tex`.
- Restyle all figures: full **black** boxed axes (all 4 spines, joined), black ticks/labels,
  drop the grey floating-axis style. (Edit `beautify.py`.)
- Fig. 7: remove the "↑ better" up-arrow; relabel panel (b) y-axis.

---

# APPLIED EDITS (old → new)

## Method / accuracy (P1, user chose Path b + peak accuracy in tables)
- Absolute accuracy: kept small model, **46.0% final** (stable, converged). Added setup
  paragraph ("On absolute accuracy") proving convergence + explaining the regime ceiling.
- Tables II, IV, ablation: added **peak accuracy** beside final for every method
  (e.g. SCOUT-FL 46.0 final → **48.8 peak**). All methods have ~2pp final→peak gap.
- Accuracy std recomputed from raw per-seed: 46.0±2.6 → **46.0±2.3** (consistent w/ peak calc).

## Theory (P4)
- Learning utility Eq.(9): modular "gradient-norm + data-mass + staleness" →
  **facility location** `f_learn(S)=Σ_j max_k s(j,k)`, RBF gradient sim (monotone submodular).
- (1−1/e): single blanket claim → **Prop 2** (holds for composite U at μ=0) + **Prop 3**
  (does NOT transfer to U_μ once μ>0; MSE min-based/non-submodular; dual+skip handle it;
  no constant-factor claim in binding regime, cite Buchbinder'14 1/e).

## Citations (P2)
- Added real cites: Asaad (arXiv 2501.06334), Zheng ICC-W'24, Du IoT-J'24, Wen 2508.15185,
  Balakrishnan DivFL ICLR'22, Wang DELTA, Sun PO-FL, Lai Oort, Nishio FedCS,
  Islam FairEquityFL, Buchbinder SODA'14.
- CollabSenseFed / Sensing-Native / Fixed-Weighted / FedAvg-ISCC / FedSGD-ISCC labelled
  explicitly as **reconstructions/controls** (no fabricated paper). CRB-Only/Sensing-Only
  labelled as classical A-/D-optimal design (Kay, Krause).

## Statistics (P5)
- Added explicit **independence caveat**: OFAT points are correlated, so the paired
  Wilcoxon is descriptive over a correlated sweep, not a formal population test; per-seed
  spread is the complementary independent measure.

## Reproducibility (P6)
- Table I now a config+repro block: κ_r=1.0, κ_a=0.05, λ0=1e-3, weights (α,λ,λ_cov,λ_fair)=
  (1,1,0.5,0.3), ε=1e-3, η=0.5, lr/momentum/epochs/batch, subsample sizes, physical layer.

## Figures
- beautify.py: grey floating axes → **complete black boxed axes** (all spines, black
  ticks/labels, light grid); schematics (fig1, fig2) get a black frame (`save(box=True)`).
- Fig.7: removed "↑ better" arrow; panel (b) y-axis → "sensing information, log det J(S)";
  title → "(b) sensing information (higher is better)".
- NEW **fig9_lambda** (λ sweep) + Section VI "Choosing the trade-off weight λ".
- (Earlier this review: fig6+fig7 all-12 baselines; removed fig CD + algorithm schematic.)

## Template
- `\documentclass[journal]{IEEEtran}` → **`[lettersize,journal]`** on the
  `bare_jrnl_new_sample4` package set; algorithm rewritten in the `algorithmic` package
  (uppercase \STATE/\FOR/\ENDFOR). Compiles: **10 pages, all refs/cites resolved.**

## λ-sweep result (P3, DONE)
7 values × 5 seeds. acc / final-CRB: λ0 47.6%/0.110 · λ0.25 48.8%/0.086 · λ0.5 46.5%/0.075 ·
**λ1 46.5%/0.067 (operating point)** · λ2 45.7%/0.059 · λ4 43.0%/0.062 · λ8 39.1%/0.056.
Clean monotone CRB↓ with gentle accuracy↓; λ=1 at the accuracy-favouring knee. fig9_lambda built.
(Note: fixed a tag bug — `%g` dropped `.0` so whole-number λ dirs weren't matched.)

## Review round 3 (figures 5 & 9, paper typos) — DONE
- **Dropped Fig. 5 (Pareto dominance matrix)**: the 96% number now lives in a sentence in
  the frontier discussion (no matrix figure).
- **Dropped Fig. 9 (rank figure) entirely** (user call): investigated three rankings and
  none honestly gives "SCOUT-FL rank-1 most" over all 12 — a sensing-abandoner (OTA-FL-ISCC,
  raw acc rank-1 in 72%) out-scores it on raw accuracy while sitting off the sensing band;
  joint ISAC score drops SCOUT-FL to ~rank 3. Rather than force a misleading frame, removed
  the comparison. "Consistency and cross-dataset" subsection → "Cross-dataset generalisation".
- **Honesty tighten (surfaced by the above)**: claims of "highest accuracy of *any*
  sensing-aware method" → "highest accuracy among *sensing-competitive* methods" in the
  abstract, contributions, and summary (OTA-FL-ISCC has higher raw acc but off-band).
- **Prop. 3 sentence** smoothed (removed the dangling em-dash "(Proposition 4)" appositive).
- **Algorithm 1 numbering**: verified it renders cleanly 1--15 (the "10 to 106" was a
  viewer glitch; source auto-numbers correctly). No change needed.
- **Added O(KNM) complexity** sentence in the selection-cost paragraph (K greedy steps × N
  candidates × O(M)/target rank-two FIM update; learning term O(KN^2)).
- Figs 1 & 2 → user's new PNGs (sys_model.png full-width; fig_2.png single column).
- Paper: **9 figures, 10 pages, compiles clean, all refs/cites resolved.**

## FINAL STATE — DONE
- paper.tex: **10 pages, compiles clean, all refs/cites resolved.**
- 11 figures, all black-boxed axes; Overleaf zip repackaged (SCOUT-FL_TWC_overleaf.zip).
- All six review priorities + template + peak-accuracy + figure feedback addressed.

---

## 2026-07-06 — Gap-distribution figure, Pareto audit appendix, CRB references

### [DIAG] 96% claim verified from cached campaign data (runs/campaign, 12-method pool, 25 OFAT points)
- final-round CRB: SCOUT-FL non-dominated **24/25 = 96%**; round-mean CRB: **24/25 = 96%**.
- Sole dominated point (both conventions): `A_learning_noniid=0.1`, dominated by Sensing-Native.
- Matches the printed 96% in all four locations (abstract, §I-B contributions, §VI-C, §VI-H
  summary + conclusion). No claim edits needed.
- Field-wide Pareto shares differ between conventions by up to ±16 pp (Fixed-Weighted 36% vs 20%);
  identical for SCOUT-FL. Stated in the new caption.
- Gap-to-best stats (25 pts, seed-mean final acc): SCOUT-FL median 1.29 pp, max 4.66 pp.
  NOTE (flagged, not a paper claim): OTA-FL-ISCC has the smallest median gap (0.00 pp) and sorts
  left of SCOUT-FL in the new figure; narration frames it as leaving the sensing band (Fig. 3).

### [EDIT] paperfigs.py — new `fig_gap_dist()` + `_gap_pareto_data()`
- Renders `figures/fig_gap_dist.{pdf,png}`: 12 box-and-whisker + jittered dots, sorted by median
  gap, y capped at 12 pp with off-scale medians printed, Pareto-% row on top (final-round CRB).
- Hard assert: SCOUT-FL Pareto % must equal 96 under both conventions or the script fails.
- Writes audit trail `figures/stats/gap_pareto.json` + table body
  `figures/stats/gap_pareto_table_body.tex`.

### [EDIT] paper.tex — figure/table/number mapping after insertion
- NEW Fig. 4 = fig_gap_dist (gap distribution + Pareto shares), placed in §VI-C.
- Old Fig. 4 (paired deltas) → Fig. 5; lambda → 6; convergence → 7; threshold → 8;
  runtime → 9; rmse-crb → 10. All via \ref (auto), one hardcoded "Fig. 9" in §VI-J
  fixed to \ref{fig:rmse} (now correctly prints Fig. 10). Hardcoded "Table IV" in §VI-G
  → \ref{tab:cross}.
- NEW Appendix B "Per-Operating-Point Pareto Status" + Table VI (25 rows grouped
  dataset/partition/channel-SNR/geometry; dominated? under both conventions; summary
  row 24/25 = 96% each). Referenced from the Fig. 4 caption.

### [EDIT] Claim sentences touched (old → new)
- §I-B contribution 4: "...under both CRB conventions, and attains..." →
  "...under both CRB conventions (Fig.~\ref{fig:gapdist}), and attains...".
- §VI-C: "non-dominated in 96% of operating points under each CRB convention, a statement..." →
  "... under each CRB convention (Fig.~\ref{fig:gapdist}; the per-point audit is given in
  Appendix~\ref{app:pareto}), a statement..."; plus two new narration sentences introducing Fig. 4.
- §VI-D opener: "To assess robustness beyond a single operating point, Fig.~\ref{fig:paired}
  pairs..." → "Fig.~\ref{fig:gapdist} summarises the field-wide gap distribution;
  Fig.~\ref{fig:paired} complements it by resolving the per-point uncertainty against the two
  strongest rivals, pairing..." (complementary-roles framing; Fig. 5 kept).
- §VI-H summary: added "; across all 25 operating points its accuracy gap to the field's best
  never exceeds 4.7 percentage points (Fig.~\ref{fig:gapdist})" and
  "(Fig.~\ref{fig:gapdist}; Appendix~\ref{app:pareto})" on the 96% sentence.
- §VII conclusion: "in 96% of operating points, and outperforms" →
  "in 96% of operating points (Fig.~\ref{fig:gapdist}), and outperforms".
- Abstract unchanged (no \ref in abstract per IEEE convention).

### [EDIT] References — two additions (Godrich KEPT at Eq. (7))
- Added `wang2024cramer` (Wang/Tao/Sun, TWC 2024) and `huang2025cramer` (Huang/Tang/Wu/Wang,
  T-RS 2025) after hua2023isac → they become [19] and [20]; Kay shifts [29]→[31],
  Godrich [30]→[32] (manual thebibliography, \cite keys renumber automatically).
- Cited in §I-A ISAC co-design sentence ("with recent designs driven directly by the
  Cramér–Rao bound of the sensing task") and in §VI-J CRB-vs-SMI argument ("the CRB remains
  the metric around which contemporary ISAC waveform and beamformer designs are built").
- Kay + Godrich at Eq. (7) untouched.

### [EDIT] Hygiene (§VI-G)
- "(9).The" → "(9). The".
- Comma splice: "scales with dataset complexity, the relative..." → "..., while the relative...".
- "paticular" → "particular"; the dangling one-line paragraph merged into the paragraph above.

### [DIAG] Compile check (pdflatex ×2, TinyTeX; installed sttools for stfloats.sty)
- No undefined references or citations, no overfull hboxes, no errors.
- Pre-existing warnings only: "Unused global option(s)", "No \author given".
- Float placement inspected on pages 7 and 11: Fig. 4 top-left of p.7 beside Table II;
  Table VI top-right of p.11 next to Appendix B text. Clean.

### [EDIT] 2026-07-06 (follow-up) — figure restyle + Appendix B removed
- fig_gap_dist restyled: SCOUT-FL in hero blue, all 11 baselines in muted neutral (#8b939c);
  "XX% Pareto" now sits above each box's top whisker (final-round CRB), 0% methods unlabelled;
  off-scale methods keep the printed median. Sort remains honest ascending-median: OTA-FL-ISCC
  (median 0.00 pp) is leftmost, SCOUT-FL second — NOT reordered to force SCOUT-FL far left.
- Appendix B "Per-Operating-Point Pareto Status" + Table VI DELETED per instruction. The
  audit facts folded into the main text instead: §VI-C now names the single dominated point
  ("the strongly non-IID partition α=0.1, where Sensing-Native prevails"); caption now states
  96% = 24/25 and notes 0% methods carry no label. §VI-H appendix ref removed. Machine audit
  trail remains in figures/stats/gap_pareto.json.
- Recompiled clean: no undefined refs, no overfull hboxes; paper 12 → 11 pages;
  Fig. 4 = gap distribution (single column, p. 7).

---

## 2026-09-05 — TWC scope desk-reject → TCCN retarget, and an AirComp bug

TWC (Paper-TW-Jul-26-2559) desk-rejected on **scope only** — no reviewers, no technical
criticism. The Executive Editorial Committee held that problem (12) carried no explicit
PHY constraints, that the algorithm ignored power control, bandwidth allocation and
interference, and that the numerical study was not tied to the wireless environment.
Retargeted to IEEE TCCN, whose scope the paper already fits.

### [BUG] AirComp aggregation MSE was clamped → 75.4% of campaign rounds invalid
- `sim/aircomp.py` computed `sigma2 / (|S|^2 * max(eta, 1e-12))` with `eta = P * g_min`.
  The `1e-12` floor was written for the normalised convention (P = sigma2 = 1, eta ~ 1).
  Under the physical link budget eta is 1e-14..1e-10 W, so the floor was **always active**:
  MSE saturated at `sigma2/(K^2 * 1e-12)` = **2.0067e-4**, independent of transmit power.
- Reach: **452,354 / 600,000** stored campaign rounds sat exactly on the ceiling (75.4%);
  `runs/campaign_main` 75.4%, `runs_lambda` 99.7%, `runs_tccn` 37.6%.
- Symptoms in the submitted paper: `B_wireless_snr` points -15..-35 dBm bit-identical
  (acc 0.4595, CRB 0.05463 at every one); Table II's MSE column identical (2.0e-4) for
  four methods; the "constraint is slack at the main operating point" claim (Prop 4(i))
  an artifact — the true MSE at -15 dBm is ~1.2e-3, above the budget eps = 1e-3.
- Direction of the bias: the clamp *under*-injects AirComp noise for weak-min-gain sets,
  so it flattered the channel-blind methods (random / DivFL / PO-FL clamped in 96% of
  their rounds) and suppressed SCOUT-FL's channel-awareness. Fixing it should widen the
  margin, but that must be measured, not asserted.
- **[EDIT] Fixed**: guard only the degenerate `eta <= 0` case (-> inf). MSE now scales
  exactly 10x per 10 dB. Two regression tests added in `tests/test_aircomp.py`
  (`test_mse_scales_with_power_in_physical_units`, `test_zero_gain_gives_infinite_mse`).
  98 tests pass, 0 failures.
- **[EDIT]** `run_campaign.py` gained `--tag` so a re-run cannot silently resume the
  clamped units. `scripts/rerun_fixed_aircomp.sh` regenerates the sensing-aware pool
  (13 methods x 25 points x 5 seeds) with the clamped stores archived, not deleted.
- Decisions taken: re-run the **sensing-aware pool only** (it is exactly what the paper
  reports); keep the nominal operating point at **-15 dBm and let the constraint bind**,
  which makes the primal-dual mechanism load-bearing at the headline point.

### [EDIT] Paper: physical-layer content added (answers the desk-reject directly)
- Title -> "Cognitive Client Scheduling for ISAC-Enabled Over-the-Air Federated Learning".
- Abstract and keywords rewritten for TCCN; PHY and power-control claims added.
- **New Sec. II-B "Physical-Layer Model and Link Budget"**: thermal noise sigma^2 = kTFB,
  log-distance path loss, small-scale fading, an interference-inclusive SINR (3), and
  per-round latency/energy (5). Real numbers: sigma^2 = -107 dBm, N0 = -167 dBm/Hz,
  uplink phase 12.8 ms, receive SNR 1.7-32.0 dB across the cell at -15 dBm.
- **New Prop. 1 (power control)**: channel-inversion AirComp with per-client power budgets
  is optimised by rho* = min_k |h_k| sqrt(P_k^max / pi_k) — every active client at full
  power, denoising factor set by the weakest link — so the joint optimisation over
  (S, {b_k}, rho) reduces with no loss of optimality to a set problem. Proof in the
  appendix. This is the direct answer to "no adaptive transmit power control".
- **New Remark 1 (multiple access and bandwidth)**: why the whole band is shared (AirComp
  uplink 12.8 ms vs 0.88 s for orthogonal access on the same 10 clients, a factor of 69),
  and the closed-form optimal bandwidth B* = d / (2(T_max - max_k C_k/f_k)), since B raises
  the noise floor linearly while cutting latency as 1/B.
- **New Remark (interference floor)**: interference enters only through sigma^2_eff, so a
  rise of Delta dB in the interference floor is indistinguishable from cutting eps by
  Delta dB — the loop manages interference without estimating it.
- **Problem (12) restated** with the alignment condition, per-client power budgets, the
  interference-inclusive MSE constraint and a latency deadline, then reduced to (13).
- **New Sec. VI-K "Operation Across the Transmit-Power and SINR Range"** + Fig. 14
  (`fig_power_sweep` in paperfigs.py, 4 panels: accuracy, CRB, realised MSE vs budget,
  settled dual price). Numbers pending the re-run, marked with \TBD.
- Prop. 4 regime scoping rewritten: the main operating point is now BINDING, so the
  primal-dual mechanism, not the greedy constant, carries the guarantee there.
- Related work: new paragraph on AirComp power control / ISAC beamforming
  (zhu2020broadband, cao2022optimized — both verified against source), positioning this
  paper as solving the allocation in closed form and leaving the set as the real problem.
- Table I: added path-loss exponent, bandwidth, noise floor, payload, uplink phase, CPU;
  relabelled the mislabelled "Receive SNR sweep 0..-35 dB" as a transmit-power sweep in dBm.
- Conclusion: dropped "joint power control with the dual price" from future work (it is now
  a result); replaced with multi-cell interference coupling, multi-antenna beam selection,
  and imperfect CSI/synchronisation in the AirComp alignment.
- Hardcoded "Proposition 5" / "Proposition 1" / "(4) and (5)" / "(9)" replaced with \ref.
- `\TBD{..}` marks every number awaiting the re-run; the file header lists them all.

---

## 2026-09-05 (later) — codebase sanity audit, and what it changed

A full pass over the simulator for hardcoded values standing in for physics, for
knobs the config advertises but the code ignores, and for claims in the paper the
code does not support. Ten findings, six of them affecting reported numbers.

### Findings that change results
1. **The channel was frozen for the whole run.** `g` was drawn once per seed and
   reused for all 150 rounds, while the paper writes `h_{k,t}` and describes a
   scheduler that perceives the channel every round. The adaptive dual price had
   nothing to track except the injected epsilon step.
   FIX: `sim/channel.py` split into `large_scale_gain` (fixed by geometry) and
   `small_scale_fading` (redrawn per coherence block). New config
   `channel.fading_per_round` and `channel.coherence_rounds`, off by default so the
   old behaviour is reproducible, on in `campaign_tccn.yaml`. The coherence length is
   now a swept axis, and its longest point recovers the frozen channel exactly.
2. **The sensing SNR did not depend on the transmit power.** `sensing.ref_snr_db: 20.0`
   was a fixed constant, so a transmit-power sweep moved the communication axis while
   holding the sensing axis still. That is not ISAC. The module docstring had flagged
   the placeholder and the replacement never arrived.
   FIX: `sensing_snr` takes `tx_power_dbm` and `ref_tx_power_dbm` and shifts the echo
   SNR by their difference in dB, anchored at -15 dBm so the nominal point is
   bit-identical and only the sweep moves.
3. **The AirComp noise injected into training used a hardcoded 0.5 factor** and
   ignored the per-entry update power pi entirely. Measured pi is 1e-5 to 4e-5, so
   the distortion was over-injected by more than two orders of magnitude, which is
   the likely cause of the 46 percent CIFAR-10 plateau the earlier review flagged.
   FIX: `aircomp.ota_noise_scale: auto` derives the scale as sqrt(pi) from the
   previous round's updates. The constraint itself stays in normalised units, so
   epsilon keeps its meaning as a bound on the relative aggregation error and the
   epsilon grid keeps its interpretation.
4. **Interference did not exist in the code** while the paper's model carries it.
   FIX: `physical.interference_dbm` folds into sigma^2 everywhere, plus a sweep.
5. **Two baselines are the same selector as something else.** Measured over 150
   rounds and 5 seeds, `iscc_air_feel` has mean Jaccard overlap 0.891 with
   `comm_only` and is bit-identical on seed 0; `fedavg_iscc` and `fedsgd_iscc` share
   a selector; `fed_iscc` is identical to `snr_only`. The pool of 32 methods contains
   28 distinct selection trajectories.
   This is not only a bookkeeping problem, it is the paper's own first claim made
   measurable. Where targets sit inside the cell, echo SNR and channel gain are
   monotone in nearly the same distance, so any scalar-SNR sensing criterion ranks
   clients exactly as channel gain alone. A bearing-dependent criterion escapes it.
   NEW: `analysis/baseline_overlap.py` measures it, and a new paper subsection
   reports it instead of asserting it.
6. **Per-client transmit budgets** were a single scalar while the new Proposition 1
   is stated with `P_k^max`. FIX: `aggregation_mse` accepts per-client budgets.

### Paper claims the code did not support
7. The algorithm section claimed lazy evaluation. `penalized_greedy`, which is the
   path SCOUT-FL v2 actually takes, re-evaluates every candidate. That is the correct
   choice, since the priced objective loses submodularity once the dual is positive,
   so cached marginal bounds are unsafe. The text now says so and ties it to the
   measured overhead.
8. Problem (12) carries a latency deadline that nothing enforced or reported.
   `round_latency_s` is now logged per round.

### Dead knobs
9. `physical.sense_power_w`, `physical.sense_time_s` and `selection.use_lazy_greedy`
   were read by nothing. Dropped from the TCCN config.

### Reproducibility
10. Per-unit seeding is correct, so resume order does not change results. MPS runs
    under `warn_only` determinism, so the final numbers should come from CUDA or CPU.

### Delivered
- `scout_fl/configs/campaign_tccn.yaml`, the configuration the paper reports.
- `scripts/schedule_experiments.sh`, ten stages, 2241 runs, resumable and sharded,
  with stage 4 taking its epsilon grid from what stage 1 measured.
- Three new sweeps: interference, bandwidth, channel coherence.
- `analysis/baseline_overlap.py`.
- 9 new tests. 123 pass, 0 fail.
- Paper: new subsections on the scalar-proxy collapse and on the three physical-layer
  sweeps, the fading and normalisation statements, the lazy-evaluation correction.
- Style pass. No colons, semicolons or dashes remain in prose, and the guarantees are
  stated as what is claimed rather than what is not.
