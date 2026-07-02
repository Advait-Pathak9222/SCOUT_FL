# RECA-FL TWC Experiment Plan

## E1. Stationary Wireless ISAC-FEEL Main Bake-Off

Purpose: show RECA is competitive even without non-stationarity.

Metrics:

- test accuracy,
- convergence rounds,
- best accuracy,
- CRB mean,
- CRB-CVaR,
- AirComp MSE mean,
- AirComp MSE-CVaR,
- energy,
- latency,
- constraint violation rate,
- Jain fairness,
- selection runtime,
- total round time.

Baselines: FedAvg, FedProx, FedCS, Oort, FedCor, resource-aware selection,
AirComp-FedAvg, ISCC Air-FEEL, and Asaad-style OTA-FEEL where feasible.

## E2. Non-Stationary Wireless ISAC Shift

Purpose: show RECA recovers faster after wireless/sensing changes.

Shift types:

- target-motion change,
- blockage event,
- channel fading distribution shift,
- sensing clutter shift,
- rare-class region appears,
- coupled rare-class plus sensing-region shift.

Metrics:

- recovery rounds to 95% pre-shift accuracy,
- recovery rounds to pre-shift CRB,
- post-shift accuracy,
- CRB-CVaR,
- AirComp MSE-CVaR,
- outage probability,
- trigger delay,
- adapter creation round.

## E3. Adapter Mechanism Proof

Purpose: prove RECA is not just a clever score.

Compare:

- Full RECA,
- RECA-score-only,
- RECA-no-adapter,
- random trigger,
- periodic trigger,
- oracle trigger,
- spawn-only-no-consolidation,
- no quarantine,
- frozen adapter.

Acceptance:

- Full RECA beats score-only and no-adapter after non-stationarity.
- Oracle trigger acts as an upper bound or near upper bound.
- Random and periodic trigger underperform full RECA.

## E4. World-Model Reliability and Calibration

Purpose: prove the appraisal signals are meaningful.

Evaluate:

- pre-shift calibration,
- immediately post-shift calibration,
- after adapter consolidation calibration.

Metrics:

- prediction RMSE for loss drop,
- prediction RMSE for CRB improvement,
- prediction RMSE for MSE effect,
- ECE/calibration error,
- Spearman/Pearson correlation between predicted and realized progress,
- trigger precision,
- trigger recall,
- false trigger rate,
- missed trigger rate,
- quarantine rate.

Acceptance:

- World-model predictions correlate with realized progress.
- Calibration improves after adapter consolidation.

## E5. Tail-Risk Constrained Wireless Reliability

Purpose: show CVaR/tail-risk appraisal protects against rare but severe
wireless/ISAC failures.

Stressors:

- severe channel fades,
- sensing occlusion,
- target ambiguity,
- blockage probability,
- sudden MSE spikes,
- high-mobility targets.

Compare:

- Full RECA with CVaR,
- RECA with mean-risk appraisal,
- RECA-no-risk,
- RECA-no-overwhelm-control,
- AirComp/ISAC baselines.

Metrics:

- CRB-CVaR,
- AirComp MSE-CVaR,
- worst-cell localization error,
- latency violation probability,
- energy violation probability,
- MSE-threshold violation probability,
- accuracy under rare failures.

## E6. Wireless Resource Trade-Off Curves

Purpose: show RECA's performance under wireless resource budgets.

Sweep:

- transmit power budget,
- selected-client budget,
- bandwidth/resource budget if modeled,
- sensing-resource budget,
- latency budget,
- energy budget,
- adapter-memory budget.

Plots:

- accuracy vs energy,
- accuracy vs latency,
- accuracy vs AirComp MSE,
- CRB vs AirComp MSE,
- CRB-CVaR vs power budget,
- recovery speed vs adapter-memory budget,
- accuracy/CRB trade-off vs selected-client budget.

Acceptance: RECA should offer better recovery/reliability trade-offs, not only
higher accuracy.

## E7. Overhead and Scalability

Purpose: show RECA is practical.

Sweep:

- clients `N = 50, 100, 200, 500`,
- selected-client ratios `5%, 10%, 20%`,
- targets `M = 1, 3, 5, 10`,
- adapters `A = 0, 1, 2, 4, 8`,
- probe dimension,
- regime count.

Metrics:

- selection time,
- world-model update time,
- adapter memory,
- extra FLOPs,
- communication payload,
- total round time,
- total energy,
- convergence rounds.

## E8. Adapter Reuse and Cross-Regime Generalization

Purpose: show adapters are reusable representations.

Setup:

- Regime A appears at round 50.
- RECA creates and consolidates adapter A.
- System returns to normal.
- Similar regime A-prime appears at round 120.

Compare:

- full RECA with adapter memory,
- RECA-no-memory,
- RECA-score-only,
- wrong-adapter reuse,
- random reuse,
- oracle reuse.

Metrics:

- second-shift recovery time,
- adapter reuse rate,
- adapter-match confidence,
- immediate post-second-shift accuracy,
- immediate post-second-shift CRB,
- rounds saved,
- false reuse rate,
- wrong-reuse penalty.

Adapter similarity/matching logs:

- cosine similarity of residual signatures,
- KL divergence between residual distributions,
- distance between gradient/sensing residual embeddings,
- `adapter_match_confidence`.

## E9. SNR, Mobility, and Channel Robustness

Purpose: verify RECA remains stable across wireless regimes.

Sweep:

- uplink SNR,
- sensing SNR,
- fading severity,
- blockage probability,
- target speed,
- client mobility,
- channel coherence time,
- noise variance.

Metrics:

- accuracy,
- CRB-CVaR,
- AirComp MSE,
- trigger precision,
- false reuse rate,
- latency/energy violations,
- recovery rounds.

Acceptance: RECA should not trigger excessive false adapters under noisy
channels and should remain stable across wireless regimes.

## E10. Beamforming / Power-Control Compatibility

Purpose: show RECA is compatible with PHY-layer resource optimization instead
of ignoring it.

Compare RECA on top of:

- fixed power,
- channel inversion,
- OTA-FEEL power control,
- Asaad-style scheduling/beamforming wrapper if feasible.

Metrics:

- AirComp MSE,
- CRB,
- CRB-CVaR,
- energy,
- latency,
- accuracy,
- recovery rounds.

Acceptance: RECA should improve selection/accommodation across multiple PHY
resource-control backends.

## E11. Statistical Significance and Robustness

Apply to all major results.

Use:

- at least 5 seeds,
- mean +/- std,
- 95% confidence intervals,
- paired tests where methods share seeds,
- Wilcoxon signed-rank test if normality is questionable,
- Friedman/Nemenyi-style rank summary for multi-method comparisons.

## Implementation Priority

Phase 1:

- RECAAppraisal,
- WorldModel,
- ContextAdapterBank basic spawn/route/consolidate/quarantine,
- RECASelector,
- E2 sudden shift,
- E3 adapter mechanism proof,
- E4 world-model reliability,
- minimal external baselines: FedAvg, FedProx, FedCS, Oort, FedCor,
  AirComp-FedAvg, Asaad-style wrapper, ISCC-Air-FEEL wrapper if available.

Phase 2:

- E5 tail-risk stress,
- E8 adapter reuse,
- trigger precision/recall,
- wrong-reuse detection,
- adapter-match confidence.

Phase 3:

- E6 resource trade-off curves,
- E7 overhead/scalability,
- E9 SNR/mobility/channel robustness,
- E10 power-control compatibility,
- stronger wrappers for recent TWC/ISAC baselines.

Phase 4:

- theory documentation,
- statistical testing,
- final figure scripts,
- final paper outline.

## TWC-Ready Acceptance Criteria

The final package is TWC-ready only if:

- RECA beats strong external ISAC/FEEL baselines under non-stationary wireless
  shifts.
- RECA-score-only and RECA-no-adapter lose to full RECA after shifts.
- Random/periodic trigger underperform the full trigger.
- Oracle trigger is an upper bound or near upper bound.
- CVaR-based RECA reduces CRB/MSE tail risk compared with mean-risk/no-risk
  variants.
- Adapter reuse improves second-shift recovery over no-memory RECA.
- Overhead is quantified and reasonable.
- Results include at least 5 seeds with confidence intervals and paired or
  non-parametric statistical tests.
- Wireless trade-off curves are present.
- SNR/mobility robustness is present.
- Internal proposed methods are absent from RECA comparison tables.
