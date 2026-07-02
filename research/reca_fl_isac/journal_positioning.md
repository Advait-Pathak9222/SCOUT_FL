# RECA-FL Journal Positioning for IEEE Transactions on Wireless Communications

## Target Venue

Target journal: **IEEE Transactions on Wireless Communications (TWC)**.

RECA-FL must be framed as a wireless communications contribution first. The
learning layer is federated edge learning, but the main scientific object is a
non-stationary wireless ISAC/OTA-FEEL system with sensing uncertainty, AirComp
distortion, power and bandwidth limits, latency and energy budgets, and
regime-dependent wireless reliability.

## One-Sentence Novelty Claim

**RECA-FL is a wireless ISAC/OTA-FEEL framework that uses risk-bounded epistemic
mismatch and verified progress to trigger context accommodation, allowing the
system to create and reuse regime-specific adapters under AirComp, CRB, latency,
energy, and sensing-resource constraints.**

## Positioning

RECA-FL is not only a client-selection algorithm and not only an FL algorithm.
It is a wireless resource-selection and representation-accommodation framework
for ISAC-enabled federated edge learning.

At each round, the BS/server observes wireless, sensing, and learning probes:
channel gains, SNR, AirComp distortion estimates, sensing/FIM/CRB summaries,
client losses or gradient sketches, energy/latency state, and regime residuals.
The RECA mechanism asks:

```text
Is the wireless/sensing-learning environment mismatched with the server model,
is the mismatch bounded-risk, and is adapting to it expected to improve FL,
sensing reliability, AirComp distortion, or tail-risk behavior?
```

If yes, RECA triggers **context accommodation**. It may spawn an adapter, route
selected clients through a stored adapter, allocate additional sensing/resource
attention, consolidate useful adapters, or quarantine harmful adapters. Reuse is
permitted only when adapter-regime similarity gives sufficient
`adapter_match_confidence`.

## TWC-Focused System View

The system is an ISAC-enabled BS coordinating wireless clients over shared
communication/sensing resources. The decision includes:

- selected clients `a_k^t in {0,1}`,
- transmit powers `p_k^t`,
- optional bandwidth/resource allocation `b_k^t`,
- optional sensing allocation `r_k^t`,
- adapter route: no adapter, spawn, reuse, consolidate, quarantine.

The performance surface is wireless and ISAC-driven:

- FL convergence and final accuracy,
- over-the-air aggregation MSE,
- sensing Fisher information and CRB,
- CRB-CVaR and AirComp-MSE-CVaR,
- outage/constraint violation probability,
- energy and latency,
- adaptation speed after channel/sensing/target shifts,
- adapter overhead and memory.

## Reviewer-Risk Checklist

| TWC Reviewer Concern | Required Evidence |
|---|---|
| Is this a wireless paper or only an FL score? | Formal wireless/ISAC/OTA-FEEL model with AirComp MSE, CRB, power, bandwidth, latency, energy, and sensing-resource constraints. |
| Does RECA improve wireless reliability? | Tail-risk experiments showing lower CRB-CVaR, AirComp-MSE-CVaR, outage, and constraint violations. |
| Is accommodation more than selection? | RECA-score-only and RECA-no-adapter must lose after non-stationary wireless shifts. |
| Is adapter reuse safe? | Log residual similarity, KL divergence, embedding distance, `adapter_match_confidence`, false reuse, and wrong-reuse penalty. |
| Does the world model remain calibrated under channel/sensing shifts? | Pre-shift, post-shift, and post-consolidation calibration with RMSE/ECE/correlation. |
| What is the PHY/resource trade-off? | Curves versus power, SNR, selected-client budget, sensing budget, latency, energy, and AirComp MSE. |
| Does the method scale? | Runtime, FLOPs, memory, payload, and round-time sweeps in clients, targets, adapters, and probe dimension. |
| Is there theory? | Convergence with AirComp/sensing distortion, accommodation benefit, trigger/reuse reliability, and complexity bounds. |

## Baseline Policy

RECA-FL should be compared only against external FL, wireless FL, OTA-FEEL, and
ISAC/FEEL baselines plus RECA ablations. Other proposed methods in this
codebase should not appear in RECA-FL baseline tables, experiment tables, or
paper-positioning arguments; shared simulator utilities may still be reused.

## Paper Identity

The paper should read as:

```text
Wireless ISAC/OTA-FEEL reliability and adaptation under non-stationary regimes
```

not as:

```text
A new client-selection heuristic for FL
```

Every RECA figure should answer how RECA improves wireless ISAC/OTA-FEEL
reliability, resource efficiency, sensing-learning trade-offs, or adaptation
under non-stationary wireless environments.
