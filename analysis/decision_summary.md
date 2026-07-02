# TEMPO-FL / CloakFL — Decision Summary
_Auto-generated 2026-07-03T02:26:39 from run artifacts. Every value is computed; missing units are listed, never imputed._

## Pre-registered gate verdicts (design §1.5, §2.6)
### GATE 1: GATE1_dominance
- **Verdict:** PENDING ⏳
- **Pre-registered criterion:** >=1 oracle schedule beats the static frontier by >=1 pp accuracy at matched-or-better time-avg tracking, paired 95% CI excludes 0
- note: no E-T1 oracle-schedule units present (run the train stage)

### GATE 2: GATE2_entanglement
- **Verdict:** PASS ✅
- **Pre-registered criterion:** median retained target log-det >= 0.50 at >=10 m client CRB floor
- **Measured:** 0.7812 (threshold 0.5)
- framing: constructive

### GATE 3: GATE3_dither
- **Verdict:** PASS ✅
- **Pre-registered criterion:** >=10x eavesdropper CRB inflation at sigma_d^2 costing <5% sensing log-det & <0.5pp acc
- eavesdropper_inflation_1rx: 10.0
- sensing_logdet_cost_frac: 0.018000000000000002

## Decision rule (design §3.2)
_Provisional (some gates not yet resolved): TEMPO GATE 1 PENDING; CloakFL GATES 2+3 PASSED. The §3.2 decision resolves once the pending Phase-1 units run._

## TEMPO-FL (E-T4)
_No E-T4 bake-off units present (gate not passed, or not yet run)._

## CloakFL (E-C3 privacy–utility frontier, E-C4 measurement)
_No E-C3 frontier units present (gates not passed, or not yet run)._

**E-C4 — every existing method localizes its clients** (worst-exposed client, most-leaky first):

| method | worst-client r (m) | median r (m) |
|---|---|---|
| aircomp_mse_min | 0.09 | 141.42 |
| comm_only | 0.09 | 141.42 |
| fedavg_iscc | 0.09 | 141.42 |
| fedsgd_iscc | 0.09 | 141.42 |
| iscc_air_feel | 0.09 | 141.42 |

## Completeness
- Units enumerated: **2920**; complete: **5**; missing: **2915**.
- First missing (up to 20): `tempo:ET1_cifar10_sp0:oracle_lts_tau30:s0`, `tempo:ET1_cifar10_sp0:oracle_lts_tau30:s1`, `tempo:ET1_cifar10_sp0:oracle_lts_tau30:s2`, `tempo:ET1_cifar10_sp0:oracle_lts_tau30:s3`, `tempo:ET1_cifar10_sp0:oracle_lts_tau30:s4`, `tempo:ET1_cifar10_sp0:oracle_lts_tau50:s0`, `tempo:ET1_cifar10_sp0:oracle_lts_tau50:s1`, `tempo:ET1_cifar10_sp0:oracle_lts_tau50:s2`, `tempo:ET1_cifar10_sp0:oracle_lts_tau50:s3`, `tempo:ET1_cifar10_sp0:oracle_lts_tau50:s4`, `tempo:ET1_cifar10_sp0:oracle_lts_tau75:s0`, `tempo:ET1_cifar10_sp0:oracle_lts_tau75:s1`, `tempo:ET1_cifar10_sp0:oracle_lts_tau75:s2`, `tempo:ET1_cifar10_sp0:oracle_lts_tau75:s3`, `tempo:ET1_cifar10_sp0:oracle_lts_tau75:s4`, `tempo:ET1_cifar10_sp0:oracle_lts_tau100:s0`, `tempo:ET1_cifar10_sp0:oracle_lts_tau100:s1`, `tempo:ET1_cifar10_sp0:oracle_lts_tau100:s2`, `tempo:ET1_cifar10_sp0:oracle_lts_tau100:s3`, `tempo:ET1_cifar10_sp0:oracle_lts_tau100:s4`
- Missing/failed units are reported as missing (design §0.3) — not imputed.
