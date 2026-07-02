# RECA-FL: Risk-Bounded Context Accommodation for Wireless ISAC/OTA-FEEL

## Core Idea

RECA-FL is a proposed method for non-stationary wireless ISAC-enabled
federated edge learning. It is positioned for IEEE Transactions on Wireless
Communications as a wireless adaptation framework, not as a standalone
machine-learning heuristic.

The central mechanism is:

```text
risk-bounded epistemic mismatch + verified progress
    -> context accommodation
    -> adapter spawn / train / consolidate / reuse / quarantine
```

RECA-FL improves an ISAC/OTA-FEEL system by deciding when a wireless/sensing
regime has changed enough that the BS should adapt its internal representation,
not merely reweight clients.

## TWC Framing

The method operates under:

- over-the-air aggregation and AirComp distortion,
- sensing uncertainty measured by FIM/CRB surrogates,
- CRB and AirComp-MSE tail-risk constraints,
- transmit-power, bandwidth/resource, latency, energy, and sensing-resource
  budgets,
- non-stationary channel, blockage, clutter, mobility, and target regimes.

At each round, the BS/server uses wireless, sensing, and learning probes to
estimate:

- tail risk,
- epistemic mismatch,
- expected/verified progress,
- adapter-match confidence for possible reuse.

## Journal Upgrade Map

- `journal_positioning.md`: TWC positioning, novelty claim, and reviewer-risk
  checklist.
- `system_model_twc.md`: wireless communication, OTA aggregation, sensing, CRB,
  latency, energy, world-model, and adapter-overhead model.
- `problem_formulation_twc.md`: constrained wireless-FL-ISAC optimization
  problem.
- `baseline_matrix.md`: external baselines and RECA ablations only.
- `theory_outline.md`: convergence, accommodation, trigger/reuse reliability,
  and complexity targets.
- `experiment_plan.md`: TWC E1-E11 experiment suite.
- `logging_schema.md`: required per-round and summary logs.
- `paper_outline_twc.md`: manuscript structure and implementation phases.

## Appraisal Signals

### Tail-Risk Appraisal

Risk measures rare but severe wireless/ISAC failure:

```text
Risk(S_t) = CVaR_alpha[
    lambda_crb * violation(CRB_t, epsilon_crb)
  + lambda_mse * violation(MSE_air^t, epsilon_mse)
  + lambda_lat * violation(T_t, T_max)
  + lambda_energy * violation(E_t, E_max)
]
```

### Epistemic Mismatch

Mismatch measures world-model prediction error in gradient, sensing, channel,
and resource effects:

```text
Mismatch(k,t) =
    ||g_k^t - hat{g}_k^t||^2
  + ||s_k^t - hat{s}_k^t||^2
  + ||m_k^t - hat{m}_k^t||^2
  + KL(q_phi(r_t | z_k^t) || p_phi(r_t | history)).
```

### Verified Progress

Progress measures whether acting on mismatch should improve wireless-FL-ISAC
performance:

```text
Progress(S_t) =
  E[Delta accuracy or loss drop
   + beta_s Delta sensing/FIM utility
   - beta_c Delta CRB
   - beta_m Delta MSE_air
   - beta_e Energy
   - beta_l Latency].
```

## RECA Trigger

Accommodation is allowed when mismatch is useful and bounded-risk:

```text
Trigger(S_t) =
  sigmoid(Progress(S_t))
  * log(1 + Mismatch(S_t))
  * exp(-((Risk(S_t) - risk_star)^2) / (2 sigma_r^2)).
```

Extreme risk or excessive unexplained mismatch is routed to a fallback rather
than treated as useful exploration.

## Adapter Lifecycle

When triggered, RECA may:

1. Spawn a context adapter for the suspected regime.
2. Route selected clients through the adapter.
3. Allocate sensing/resource attention to the affected wireless regime.
4. Train and evaluate the adapter using realized loss, CRB, MSE, latency, and
   energy changes.
5. Consolidate if progress is verified.
6. Quarantine if risk increases or the adapter fails to explain the regime.
7. Reuse only if `adapter_match_confidence >= tau_reuse`.

The default reuse threshold is:

```text
tau_reuse = 0.7
```

## Adapter Matching

Each adapter stores a regime signature:

```text
signature = [
  mean gradient residual,
  mean sensing residual,
  world-model residual distribution,
  regime embedding
]
```

RECA computes:

- embedding distance,
- residual cosine similarity,
- residual KL divergence,
- `adapter_match_confidence`.

Safe reuse is a key TWC mechanism because wireless regimes can recur under
similar mobility, blockage, or sensing-clutter conditions.

## Implementation Status

Implemented:

- `RECAAppraisal`,
- `WorldModel`,
- `ContextAdapterBank`,
- `AdapterMatcher`,
- `RECASelector`,
- `reca` registry selector for shared wireless/ISAC bake-offs,
- quick E2/E8 mechanism runners,
- RECA unit tests.

Roadmap:

- faithful wrappers for remaining external TWC/ISAC baselines,
- richer power-control and beamforming compatibility,
- full resource trade-off curves,
- SNR/mobility robustness sweeps,
- statistical testing and final figure scripts.

## Run Scripts

Quick gate:

```bash
bash scripts/reca_twc_quick.sh cuda
```

Full NVIDIA campaign:

```bash
bash scripts/reca_twc_nvidia.sh
```

The full script runs only RECA in the shared FL trainer and then runs the
RECA-specific TWC experiments. It does not rerun the existing internal proposed
methods or external baseline campaigns.
