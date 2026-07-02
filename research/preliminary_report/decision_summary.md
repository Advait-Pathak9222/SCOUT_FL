# SCOUT-FL / JEDI(VISMAYA)-FL — Decision Summary (weight-free, honest)

**Question:** are the proposed methods publishable? Every number below is computed directly
from `runs/` artifacts by `decision_analyses.py` (figures `figA1–figA6`, machine-readable
dumps in `figures/stats/`). Accuracy = `objectives.acc` (final-round). CRB is reported for
**both** conventions — round-mean (`objectives.crb`) and final-round (`objectives.crb_final`) —
because the choice materially changes the JEDI verdict. Pairing is on shared `(point, seed)`;
`C_sensing_targets=5` has only 2/5 seeds and is flagged/excluded, never imputed.

> **Headline correction to the preliminary report.** The report benchmarked mainly against
> the *weak* Asaad baseline, used an arbitrary CRB ≤ 0.10 bar, and an equal-weight joint
> score. Against the **strong** baselines (CollabSenseFed, Sensing-Native OTA-FL) and with
> weight-free tests, the proposed methods **do not dominate** — they trade accuracy for
> sensing, and the "wins 21/22" is regime-specific. The core contribution survives, but the
> claims must be re-scoped.

---

## Evidence at a glance

| Test | SCOUT-FL v2 | JEDI/VISMAYA-FL | SCOUT-FL v1 |
|---|---|---|---|
| **A1** Δacc vs CollabSenseFed (per-point, 95% CI) | **+1.6pp** [1.2, 2.1] ✔sig | **+3.6pp** [3.0, 4.2] ✔sig | −0.3pp [−2.0, 1.1] ✗tie |
| **A1** Δacc vs Sensing-Native | **+2.0pp** [1.4, 2.5] ✔sig | **+4.0pp** [3.2, 4.7] ✔sig | +0.1pp ✗tie |
| **A1** ΔCRB vs those baselines (− = better) | **+0.005** ✔sig *worse* | **+0.033** ✔sig *worse* | +0.017 ✗ns *worse* |
| **A3** Pareto-optimal, round-mean CRB | 86% | 91% | 45% |
| **A3** Pareto-optimal, **final-round** CRB | **91%** (robust) | **27%** (collapses) | 50% |
| **A6** joint-score rank (N=21, 1=best) | **1.57** | 4.00 | 3.00 |
| **A2** constrained-acc wins @τ=0.10 / band | 1 / peaks 0.065–0.09 | 19 / peaks 0.10–0.12 | 1 / — |

Legend: ✔sig = both Wilcoxon & paired-t < 0.05 **and** bootstrap 95% CI excludes 0.

---

## Per-method verdict

### SCOUT-FL v2 — **PUBLISH (headline), with re-scoped claims**
- **A1:** significantly beats *both* strong baselines on accuracy (+1.6/+2.0 pp) at a
  **negligible, though significant, CRB cost** (+0.005). This is a favorable, real trade.
- **A3:** robustly **Pareto-optimal in 86–91% of points under *both* CRB conventions** — the
  only proposed method that is convention-robust.
- **A6:** best average joint-score rank (1.57), stable whether or not the underpowered M=5
  point is included (Δrank ≤ 0.03).
- **Caveat (A6):** with only 21 *correlated* points the critical difference is wide (2.62),
  so SCOUT-v2 is **not** statistically separable from CollabSenseFed (2.19), SCOUT-v1, JEDI,
  or even Random. It robustly out-ranks only Asaad/DivFL/FedGCS/Oort.
- **Framing:** "best-balanced, Pareto-robust selector that beats the ISAC SOTA (Asaad) and is
  competitive-to-favorable against the strongest ISAC baselines" — **not** "dominates all."

### JEDI/VISMAYA-FL — **REFRAME (fixable), do not headline as-is**
- **A1:** largest accuracy gains (+3.6/+4.0 pp, significant) but at a **meaningful, significant
  CRB cost** (+0.033) vs the strong baselines — an accuracy↔sensing **trade, not domination**.
- **A3:** Pareto-optimality is **fragile to the CRB definition**: 91% (round-mean) → **27%
  (final-round)**, i.e. under a stricter CRB it is frequently dominated (worst at low-SNR /
  SNR-30). This is a genuine robustness liability.
- **A2:** its "wins 19/22" is **threshold-specific** — it owns the win-count only in
  τ∈[0.10, 0.12]; at τ=0.09 SCOUT-v2 wins, and at τ≥0.13 DivFL/Random take over. The
  report's 0.10 sits right where JEDI looks best.
- **Path to publish:** present JEDI honestly as the *accuracy-maximizing* variant, report
  final-round CRB alongside round-mean, and drop the "wins 21/22" as a robust claim.

### SCOUT-FL v1 — **DEMOTE to ablation (not a standalone contribution)**
- **A1:** accuracy **tie** with the strong baselines (ns) and worse CRB → no net advantage.
- **A3:** Pareto-optimal in only 45–50% of points; heavily dominated at low SNR (18 dominators
  at SNR-30). Its role is to motivate the v2 primal-dual fix.

### Generative-synergy / mobility claim — **HOLD (unproven; data missing)**
- **A4:** a σ_p **sweep does not exist** (only σ_p=0.05; the referenced
  `configs/sweep_mobility_ablation.yaml` is absent). At the single available point the synergy
  term gives **no accuracy benefit** and a large but **statistically insignificant** CRB
  change (Δ≈−2.2, 95% CI crosses 0 across the horizon). Do **not** claim a mobility benefit
  until the sweep (command in `figures/stats/a4_mobility.json`) is run.

### Fairness term — **KEEP as an equity feature (honest trade)**
- **A5:** removing it **starves 16% of clients** (never selected) and drops participation
  equity (Jain 0.43 → 0.74 with it). The accuracy "cost" of keeping it (−8 pp at 30 rounds)
  and CRB cost are **not statistically significant** (n=5). Frame fairness as a
  participation-equity guarantee, not an accuracy feature.

---

## Overall recommendation

**Publish — but reframe.** The defensible core is: *a proposed selector sits on the
learning–sensing Pareto frontier and significantly beats the ISAC state of the art (Asaad)
on the joint objective.* That claim is robust (A1, A3, A6). Ship **SCOUT-FL v2 as the
headline** (best-balanced, CRB-convention-robust, favorable accuracy trade at ~0 sensing
cost). **Reframe JEDI/VISMAYA-FL** as the accuracy-maximizing sibling and fix its
final-round-CRB robustness before headlining it. **Demote SCOUT-FL v1** to an ablation.
**Do not claim** (a) domination of the strong baselines — it is a trade, not domination;
(b) a robust "wins 21/22" — it is threshold- and convention-specific; (c) any mobility/synergy
benefit — the sweep is missing. Statistical significance vs the strong baselines and vs
Random is **not** established by the corrected CD test (correlated points, wide CD); lead with
the weight-free per-point paired trade (A1) and dominator robustness (A3), which are honest
and do hold.

**Artifacts:** `figA1_h2h_{collabsensefed,sensing_native}`, `figA2_threshold_sensitivity`,
`figA3_dominators`, `figA4_mobility_singlepoint`, `figA5_fairness`, `figA6_corrected_cd`
(+ `.png`); dumps in `figures/stats/a{1..6}_*.json`. Regenerate: `python decision_analyses.py`.
