# TWC Paper Outline

## Title Idea

Risk-Bounded Context Accommodation for Non-Stationary ISAC Federated Edge
Learning over Wireless Networks

## 1. Introduction

- Wireless ISAC-enabled federated edge learning must jointly manage learning
  progress, OTA aggregation distortion, sensing uncertainty, latency, energy,
  and non-stationary channel/sensing regimes.
- Existing methods typically optimize selection or resource allocation under a
  fixed representation.
- RECA-FL adds a representation-accommodation mechanism: risk-bounded mismatch
  and verified progress trigger adapter spawn, reuse, consolidation, or
  quarantine.

## 2. Related Work

- Wireless FEEL and OTA aggregation.
- ISAC/ISCC resource allocation.
- Client selection in wireless FL.
- Non-stationary and uncertainty-aware FL.

## 3. System Model

- Wireless communication model.
- Sensing model.
- OTA aggregation model.
- Learning model.
- Non-stationarity model.
- Adapter bank and world-model residual signatures.

## 4. Problem Formulation

- Combined FL loss, CRB/sensing uncertainty, AirComp MSE tail risk, adapter
  overhead, energy, and latency objective.
- Constraints on AirComp MSE, CRB, latency, energy, power, bandwidth, sensing
  resources, selected-client budget, and adapter memory.
- Non-convex, mixed-integer, non-stationary nature of the problem.

## 5. RECA-FL Algorithm

- Risk appraisal using CRB/AirComp/latency/energy tail behavior.
- Epistemic mismatch appraisal using world-model residuals.
- Verified progress appraisal using expected learning/sensing/resource gains.
- Trigger rule.
- Adapter lifecycle: spawn, train, consolidate, reuse, quarantine.
- Adapter matching and `adapter_match_confidence`.
- Client/resource selection.

## 6. Theoretical Analysis

- AirComp/sensing-aware convergence bound.
- Accommodation benefit condition.
- Trigger and reuse reliability.
- Complexity and overhead.

## 7. Experiments

- Setup, datasets, wireless/ISAC simulator, and non-stationary regimes.
- External baselines and RECA ablations.
- Metrics and logging schema.
- E1 stationary wireless ISAC-FEEL bake-off.
- E2 non-stationary wireless shift.
- E3 adapter mechanism proof.
- E4 world-model reliability.
- E5 tail-risk reliability.
- E6 wireless resource trade-off curves.
- E7 overhead/scalability.
- E8 adapter reuse.
- E9 SNR/mobility/channel robustness.
- E10 power-control compatibility.
- E11 statistical significance.

## 8. Conclusion

- Summarize wireless reliability, sensing-learning trade-off, and adaptation
  gains.
- State limitations and future extensions to richer beamforming and hardware
  OTA-FEEL deployments.

## Implementation Priority

Phase 1:

- RECAAppraisal.
- WorldModel.
- ContextAdapterBank basic spawn/route/consolidate/quarantine.
- RECASelector.
- E2 sudden shift.
- E3 adapter mechanism proof.
- E4 world-model reliability.
- Minimal external baselines: FedAvg, FedProx, FedCS, Oort, FedCor,
  AirComp-FedAvg, Asaad-style wrapper, ISCC-Air-FEEL wrapper if available.

Phase 2:

- E5 tail-risk stress.
- E8 adapter reuse.
- Trigger precision/recall.
- Wrong-reuse detection.
- Adapter-match confidence.

Phase 3:

- E6 resource trade-off curves.
- E7 overhead/scalability.
- E9 SNR/mobility/channel robustness.
- E10 power-control compatibility.
- Stronger wrappers for recent TWC/ISAC baselines.

Phase 4:

- Theory documentation.
- Statistical testing.
- Final figure scripts.
- Final paper polish.

## Final Acceptance Criteria

- RECA beats strong external ISAC/FEEL baselines under non-stationary wireless
  shifts.
- RECA-score-only and RECA-no-adapter lose to full RECA after shifts.
- Random/periodic trigger underperform full trigger.
- Oracle trigger is an upper bound or near upper bound.
- CVaR-based RECA reduces CRB/MSE tail risk compared with mean-risk/no-risk.
- Adapter reuse improves second-shift recovery over no-memory RECA.
- Overhead is quantified and reasonable.
- Results include at least 5 seeds, confidence intervals, and paired/statistical
  tests.
- Wireless trade-off curves are present.
- SNR/mobility robustness is present.
- RECA comparisons use external baselines and RECA ablations only.
