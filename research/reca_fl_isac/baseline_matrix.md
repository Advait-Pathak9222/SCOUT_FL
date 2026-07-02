# RECA-FL TWC Baseline Matrix

## Baseline Policy

RECA-FL is evaluated against external FL, wireless FL, OTA-FEEL, and ISAC/FEEL
baselines plus RECA ablations. Internal proposed methods from this codebase are
not RECA baselines.

Implementation status labels:

- **implemented**: available in the codebase registry or RECA runner.
- **wrapper needed**: nearby code exists, but a faithful wrapper is still needed.
- **citation only**: use for related-work positioning unless implemented later.

## Core FL Baselines

| Baseline | Category | Implementation Status | Required Signals | Compatible Experiments | Fairness Notes |
|---|---|---|---|---|---|
| Random selection | Core FL | implemented: `random` | client count, budget | E1-E11 | Same budget and seeds; no wireless awareness. |
| FedAvg | Core FL | implemented through random/FedAvg-style runner | client data, aggregation | E1-E5, E9 | Use same local epochs and aggregation settings. |
| FedProx | Core FL | wrapper needed | local objective, proximal coefficient | E1-E5, E9 | Tune proximal coefficient on validation grid. |
| FedCS | Core wireless FL | implemented: `fedcs` | latency, budget | E1, E2, E5-E7, E9 | Same latency model and selected-client budget. |
| Oort | Core FL | implemented: `oort` | loss, latency, RNG | E1-E5, E9 | Same loss probes and exploration setting. |
| FedCor | Core FL | wrapper needed | client correlation or GP features | E1-E4, E9 | Use same gradient/probe embeddings. |
| Loss-based selection | Core FL | implemented: `loss` | per-client loss | E1-E5, E9 | Same loss probe budget. |
| Resource-aware selection | Core wireless FL | implemented: `comm_only`, `fedcs` | channel gain or latency | E1, E5-E7, E9, E10 | No sensing/adapter information. |

## Wireless / AirComp Baselines

| Baseline | Category | Implementation Status | Required Signals | Compatible Experiments | Fairness Notes |
|---|---|---|---|---|---|
| AirComp-FedAvg | Wireless OTA-FL | implemented: `ota_fedavg` | channel gains, AirComp MSE model | E1, E2, E5-E7, E9, E10 | Same OTA distortion model and selected-client budget. |
| OTA-FEEL power-control baseline | Wireless OTA-FEEL | wrapper needed | channel gains, power budget, MSE threshold | E1, E5-E7, E9, E10 | Compare with identical power/bandwidth budgets. |
| Channel-inversion OTA aggregation | Wireless AirComp | implemented approximation: `aircomp_mse_min`, `comm_only` | channel gains, power, noise | E1, E5-E7, E9, E10 | Report when equivalent to strongest-channel selection. |
| Unreliable-channel scheduling | Wireless FL | wrapper needed | outage probability, channel state | E2, E5, E9 | Same fading/blockage traces. |
| TMLCN 2024 Agent Selection Framework | External wireless-FL comparator | wrapper needed | channel/reward/state features | E1, E2, E7, E9 | Use as external comparator only if wrapper is faithful. |
| Zheng et al. OTA-FL client selection in ISCC | ISCC OTA-FL | implemented approximation: `ota_fl_iscc` | learning utility, channel gain, MSE gate | E1, E2, E5, E9, E10 | State approximation clearly; same MSE feasibility gate. |

## ISAC / FEEL Baselines

| Baseline | Category | Implementation Status | Required Signals | Compatible Experiments | Fairness Notes |
|---|---|---|---|---|---|
| Multi-task ISAC-FL resource allocation | ISAC-FL | wrapper needed | learning task, sensing task, resources | E1, E5-E7, E10 | Match resource budgets; cite if not implemented. |
| ISCC OTA-FL / IoT-J 2024 | ISCC OTA-FL | implemented approximation: `fed_iscc` | channel gains, sensing SNR, MSE threshold | E1, E2, E5, E9, E10 | Same sensing/channel realizations. |
| ISCC Air-FEEL / TWC 2025-2026 | ISCC Air-FEEL | implemented approximation: `iscc_air_feel` | sensing SNR, channel gains, power/noise | E1, E2, E5, E9, E10 | Note that current selector is a scheduling wrapper. |
| Integrated sensing-computation-communication FEEL | ISCC FEEL | implemented approximation: `iscc_air_feel`; refine wrapper | sensing, compute, communication resources | E1, E5-E7, E10 | Keep same compute/channel/sensing budgets. |
| Asaad-style OTA-FEEL scheduling/beamforming | ISAC OTA-FEEL | implemented: `asaad` | CRB, AirComp MSE, channel gains | E1, E2, E5, E9, E10 | Primary TWC-style sensing-aware comparator. |
| Sensing-native OTA-FL | Sensing-native OTA-FL | implemented: `sensing_native` | sensing and learning utilities | E1, E2, E5, E9 | Same sensing/learning utility construction. |
| Multi-objective OTA-FEEL / collaborative ISAC | ISAC OTA-FEEL | implemented approximation: `collabsensefed`; stronger wrapper optional | learning and sensing utilities | E1, E5-E7, E10 | Use as wrapper/citation baseline depending on fidelity. |

## RECA Ablations

| Variant | Category | Implementation Status | Required Signals | Compatible Experiments | Fairness Notes |
|---|---|---|---|---|---|
| Full RECA-FL | Proposed method | implemented: `reca` registry + dedicated RECA runners | risk, mismatch, progress, wireless/sensing probes, adapter state | E1-E11 | Same budgets; adapter overhead must be logged. |
| RECA-score-only | Ablation | implemented in RECA runner | risk, mismatch, progress | E2-E4, E8 | Same score without adapter lifecycle. |
| RECA-no-adapter | Ablation | implemented as no-memory/no-adapter route | risk, mismatch, progress | E2-E4, E8 | Tests representation value. |
| RECA-random-trigger | Ablation | wrapper needed | random trigger labels | E3, E4 | Same trigger rate as full RECA where possible. |
| RECA-periodic-trigger | Ablation | wrapper needed | round index, period | E3, E4 | Match number of trigger events. |
| RECA-oracle-trigger | Ablation | wrapper needed | oracle regime labels | E3, E4 | Upper bound; not deployable. |
| RECA-no-quarantine | Ablation | wrapper needed | adapter evidence | E3, E5, E8 | Shows safety value of quarantine. |
| RECA-frozen-adapter | Ablation | wrapper needed | adapter route | E3, E8 | Adapter exists but cannot adapt. |
| RECA-no-memory | Ablation | implemented: `reca_no_memory` | current signature only | E3, E8 | Tests reusable representation. |
| RECA-wrong-reuse | Ablation | implemented: `wrong_reuse` | adapter bank, wrong adapter label | E8 | Measures wrong-reuse penalty. |
| RECA-mean-risk instead of CVaR | Ablation | wrapper needed | mean CRB/MSE risk | E5 | Same thresholds; replaces tail metric. |
| RECA-no-risk | Ablation | wrapper needed | mismatch, progress | E5 | Tests unsafe curiosity. |
| RECA-no-mismatch | Ablation | wrapper needed | risk, progress | E3-E5 | Tests world-model mismatch value. |
| RECA-no-progress | Ablation | wrapper needed | risk, mismatch | E3-E5 | Tests noise chasing. |
| RECA-no-overwhelm-control | Ablation | wrapper needed | risk, mismatch, progress | E5 | Tests bounded-risk safety. |
