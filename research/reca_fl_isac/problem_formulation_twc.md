# TWC Problem Formulation for RECA-FL

## Decision Variables

At round `t`, the BS chooses:

```text
{a_k^t}_{k=1}^N,      {p_k^t}_{k=1}^N,      {b_k^t}_{k=1}^N,
{r_k^t}_{k=1}^N,      adapter route rho_t.
```

Here `a_k^t` is the binary client-selection decision, `p_k^t` is transmit power,
`b_k^t` is optional communication bandwidth/resource allocation, `r_k^t` is
optional sensing-resource allocation, and `rho_t` chooses no-adapter, spawn,
reuse, consolidate, or quarantine.

## Objective

RECA-FL targets a wireless-FL-ISAC objective over horizon `T`:

```text
min_{a,p,b,r,rho}  E[F(w_T)]
                 + lambda_s sum_t U_sense^t
                 + lambda_c sum_t CVaR_alpha(CRB_t)
                 + lambda_m sum_t CVaR_alpha(MSE_air^t)
                 + lambda_a sum_t C_A^t
                 + lambda_e sum_t E_t
                 + lambda_l sum_t T_t
```

where:

- `F(w_T)` is final/global FL loss,
- `U_sense^t` is sensing uncertainty such as `tr(J_t^{-1})` or `-log det(J_t)`,
- `CVaR_alpha(CRB_t)` measures tail sensing/localization risk,
- `CVaR_alpha(MSE_air^t)` measures tail OTA aggregation risk,
- `C_A^t` is adapter memory/training/matching/communication overhead,
- `E_t` and `T_t` are round energy and latency.

Energy and latency may alternatively be handled purely as constraints by setting
`lambda_e = lambda_l = 0`.

## Constraints

The per-round constraints are:

```text
sum_{k=1}^N a_k^t <= K_t,                  a_k^t in {0,1}
0 <= p_k^t <= P_k^max
sum_{k=1}^N a_k^t b_k^t <= B_t
sum_{k=1}^N a_k^t r_k^t <= R_t
MSE_air^t <= epsilon_mse
CRB_t <= epsilon_crb        or        CVaR_alpha(CRB_t) <= epsilon_cvar
T_t <= T_max
E_t <= E_max
sum_{j=1}^{J_t} mem(A_j) <= B_adapter
```

Adapter reuse must satisfy:

```text
rho_t = reuse(A_j)  =>  adapter_match_confidence(A_j, r_t) >= tau_reuse.
```

Wrong-reuse and quarantine constraints may be written as risk controls:

```text
P(wrong reuse | rho_t = reuse) <= delta_reuse,
P(false trigger) <= delta_trigger.
```

## RECA Online Surrogate

The exact problem is mixed-integer because of binary client selection and
adapter lifecycle decisions. It is non-convex because FL loss, AirComp MSE,
FIM/CRB, power control, and adapter routing interact nonlinearly. It is also
non-stationary because channel distributions, target motion, blockage,
sensing clutter, and data regimes may shift over time.

RECA therefore uses an online approximate policy:

```text
risk-bounded mismatch + verified progress
    -> select clients/resources
    -> spawn/reuse/consolidate/quarantine adapters
```

The online score is not the paper's only claim. The key mechanism is that
bounded wireless/sensing-learning mismatch can trigger a representation change,
and reuse is gated by adapter-regime similarity under wireless constraints.
