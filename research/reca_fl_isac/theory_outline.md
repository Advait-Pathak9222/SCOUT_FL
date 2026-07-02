# RECA-FL TWC Theory Outline

## A. AirComp/Sensing-Aware Convergence Bound

Target theorem: under standard smoothness and bounded-variance assumptions, an
OTA-FEEL update with partial participation, sensing uncertainty, world-model
prediction error, and adapter routing satisfies a descent bound of the form

```text
E[F(w_{t+1})] <= E[F(w_t)]
  - eta_t c_1 E[||grad F(w_t)||^2]
  + eta_t^2 c_2 sigma_grad^2
  + eta_t^2 c_3 Gamma_het
  + eta_t^2 c_4 Phi_part(K_t, N)
  + eta_t^2 c_5 E[MSE_air^t]
  + eta_t c_6 E[U_sense^t]
  + eta_t c_7 E[epsilon_wm^t]
  + eta_t c_8 E[C_A^t + P_wrong^t].
```

where:

- `Gamma_het` is client heterogeneity,
- `Phi_part(K_t, N)` is partial-participation error,
- `MSE_air^t` is OTA aggregation distortion,
- `U_sense^t` is CRB or sensing uncertainty,
- `epsilon_wm^t` is world-model prediction error,
- `C_A^t` is adapter overhead,
- `P_wrong^t` is wrong-adapter penalty.

Summing over rounds should yield

```text
min_t E[||grad F(w_t)||^2]
  <= optimization terms
   + AirComp distortion
   + sensing/CRB uncertainty
   + world-model error
   + adapter overhead / wrong reuse.
```

Interpretation for TWC: RECA can improve convergence and reliability only if it
reduces AirComp/sensing/model mismatch terms more than the adapter overhead it
adds.

## B. Accommodation Benefit Condition

Let `A_j` be a candidate context adapter for regime `r_t`. Define the expected
wireless-FL-ISAC benefit:

```text
B(A_j, r_t)
  = E[Delta loss_t]
  + beta_c E[Delta CRB_t]
  + beta_m E[Delta MSE_air^t]
  + beta_r E[tail-risk relief_t]
  - C_train(A_j)
  - C_mem(A_j)
  - C_comm(A_j)
  - C_match(A_j)
  - C_wrong(A_j, r_t).
```

Accommodation is beneficial when

```text
B(A_j, r_t) > 0.
```

Spawn condition:

```text
E[B(A_new, r_t) | trigger_t = 1] > 0.
```

Reuse condition:

```text
E[B(A_j, r_t) | adapter_match_confidence_j^t >= tau_reuse] > 0.
```

This theorem motivates E3 and E8: full RECA should beat score-only/no-adapter,
and adapter reuse should recover faster than no-memory RECA when a similar
regime returns.

## C. Trigger and Reuse Reliability Proposition

Let `T_t` denote the accommodation trigger and `U_t` denote the event that
accommodation is useful over horizon `H`. Suppose the world model has calibration
error `epsilon_cal`, the progress estimator has error `epsilon_prog`, and the
risk estimator has error `epsilon_risk`.

A target proposition is:

```text
P(T_t = 1, U_t = 0)
  <= f_1(epsilon_cal, epsilon_prog, epsilon_risk, margin_trigger),

P(T_t = 0, U_t = 1)
  <= f_2(epsilon_cal, epsilon_prog, epsilon_risk, margin_trigger).
```

For adapter reuse, let `C_j^t` be adapter-match confidence and let `W_j^t` be the
wrong-reuse event:

```text
P(W_j^t | C_j^t >= tau_reuse)
  <= f_3(epsilon_cal, residual_KL, embedding_distance,
         residual_cosine_margin, tau_reuse).
```

This directly motivates logging:

- prediction RMSE and ECE,
- trigger precision/recall,
- false trigger and missed trigger rates,
- embedding distance,
- residual cosine similarity,
- residual KL divergence,
- `adapter_match_confidence`,
- false reuse and wrong reuse.

## D. Complexity and Overhead Bound

Let:

- `N` be number of clients,
- `K` be selected clients,
- `d_p` be probe dimension,
- `d_o` be world-model output dimension,
- `A` be number of stored adapters,
- `d_s` be adapter signature dimension,
- `M` be number of targets,
- `C_FIM(M,K)` be the cost of FIM/CRB computation,
- `C_wm(d_p,d_o)` be world-model update cost.

A per-round bound is:

```text
O(N d_p d_o)                 world-model prediction
+ O(N)                       risk/mismatch/progress appraisal
+ O(N log N)                 client ranking/selection
+ O(A d_s)                   adapter matching
+ O(C_FIM(M,K))              sensing/FIM/CRB computation
+ O(K C_wm(d_p,d_o))         world-model update on selected clients
+ O(K C_adapter)             adapter training/routing
```

Adapter memory is:

```text
O(A d_s + A P_adapter),
```

where `P_adapter` is the adapter parameter count if adapters are implemented as
trainable modules. Communication overhead is the base OTA/FL payload plus any
adapter routing metadata and probe bytes:

```text
Payload_t = Payload_FL_t + O(N d_p) + O(K log A).
```

The TWC scalability experiment must verify that these costs remain practical
under sweeps over `N`, `K/N`, `M`, `A`, probe dimension, and regime count.
