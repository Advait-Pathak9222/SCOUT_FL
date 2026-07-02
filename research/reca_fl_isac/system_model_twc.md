# TWC System Model for RECA-FL

## Network and Round Structure

Consider an ISAC-enabled base station (BS) coordinating `N` wireless clients for
federated edge learning over rounds `t = 0, 1, ..., T-1`. Client `k` stores local
data `D_k` and computes a local update for the global model `w_t`. The BS also
uses the same wireless infrastructure to sense `M` targets or regions.

Client selection is represented by

```text
a_k^t in {0,1},              sum_{k=1}^N a_k^t <= K_t,
```

where `K_t` is the selected-client budget. The selected set is

```text
S_t = {k : a_k^t = 1}.
```

RECA also maintains a world model `M_phi^t` and an adapter bank
`A_t = {A_1, ..., A_{J_t}}` for wireless/sensing regimes.

## Wireless Uplink Model

For selected client `k`, the uplink channel at round `t` is

```text
h_k^t = sqrt(beta_k^t) g_k^t,
```

where `beta_k^t` captures path loss and shadowing and `g_k^t` captures small
scale fading. The received signal over an allocated communication resource is

```text
y^t = sum_{k in S_t} h_k^t sqrt(p_k^t) x_k^t + n^t,
```

where `p_k^t` is transmit power, `x_k^t` is the normalized local update symbol,
and `n^t ~ CN(0, sigma_c^2)` is receiver noise. A typical instantaneous SNR is

```text
gamma_k^t = p_k^t |h_k^t|^2 / sigma_c^2.
```

Optional bandwidth/resource variables `b_k^t` satisfy

```text
sum_{k in S_t} b_k^t <= B_t,        0 <= p_k^t <= P_k^max.
```

## OTA Aggregation and AirComp Distortion

The BS estimates the weighted average update

```text
Delta_t = sum_{k in S_t} q_k^t Delta_k^t
```

using over-the-air aggregation. With channel inversion or power control, the
aggregated estimate is

```text
hat{Delta}_t = Delta_t + e_air^t,
```

where `e_air^t` is the AirComp aggregation distortion. A scalar proxy is

```text
MSE_air^t = E[ ||hat{Delta}_t - Delta_t||_2^2 ].
```

For channel-inversion AirComp with common receive scaling, one common bound is

```text
MSE_air^t proportional to sigma_c^2 / (|S_t|^2 P_t min_{k in S_t} |h_k^t|^2),
```

which makes weak selected channels a direct source of aggregation distortion.

## Sensing Model and CRB

The BS receives sensing echoes or client-assisted sensing summaries. For target
state `theta_m^t`, the sensing observation can be abstracted as

```text
z_m^t = q(theta_m^t, {r_k^t, h_k^t}_{k in S_t}) + v_m^t,
```

where `r_k^t` is optional sensing-resource allocation and `v_m^t` is sensing
noise/clutter. The Fisher information matrix (FIM) is

```text
J_t(S_t, r^t) = sum_{k in S_t} J_k^t(r_k^t, h_k^t, theta^t),
```

and the sensing uncertainty is measured by CRB surrogates such as

```text
CRB_t = tr(J_t^{-1}),              or              -log det(J_t).
```

Tail reliability uses `CVaR_alpha(CRB_t)` over targets, regions, or rounds.

## Target Motion and Non-Stationarity

Targets and wireless regimes evolve as

```text
theta_m^{t+1} = f_r(theta_m^t) + xi_m^t,
```

where the latent regime `r_t` can change due to mobility, blockage, clutter,
rare-class appearance, channel fading distribution shift, or sensing geometry
change. A regime shift changes the joint distribution of channel gains, sensing
echo summaries, gradient/loss probes, and expected resource effects:

```text
P_t(h, z, Delta, loss) != P_{t-1}(h, z, Delta, loss).
```

RECA treats such shifts as candidates for context accommodation only when the
mismatch is bounded-risk and verified progress is expected.

## Client Computation Model

Client `k` performs `E_k^t` local steps over mini-batches. With `c_k` CPU cycles
per sample, `n_k^t` processed samples, and CPU frequency `f_k^t`,

```text
T_{cmp,k}^t = c_k n_k^t E_k^t / f_k^t,
E_{cmp,k}^t = kappa_k c_k n_k^t E_k^t (f_k^t)^2.
```

## Latency Model

The round latency includes local computation, uplink transmission, and optional
sensing/action latency:

```text
T_t = max_{k in S_t} (T_{cmp,k}^t + T_{ul,k}^t + T_{sense,k}^t)
```

with

```text
T_{ul,k}^t = L_k^t / (b_k^t log_2(1 + gamma_k^t)).
```

The latency budget is

```text
T_t <= T_max.
```

## Energy Model

Total round energy is

```text
E_t = sum_{k in S_t} (E_{cmp,k}^t + p_k^t T_{ul,k}^t + E_{sense,k}^t).
```

Budgets may be per-round or long-term:

```text
E_t <= E_max,        E_k^t <= E_k^max.
```

## World-Model Prediction Error

The world model predicts per-client effects:

```text
M_phi^t(k) -> [hat{Delta loss}_k^t, hat{Delta CRB}_k^t,
               hat{MSE}_k^t, hat{T}_k^t, hat{E}_k^t].
```

The prediction residual is

```text
epsilon_{wm,k}^t = y_k^t - M_phi^t(k),
```

where `y_k^t` is the realized vector after selection. RECA uses these residuals
to compute epistemic mismatch and adapter signatures.

## Adapter Memory and Overhead

Each context adapter `A_j` stores a regime signature:

```text
psi_j = [mean gradient residual, mean sensing residual,
         world-model residual mean/variance, regime embedding].
```

The adapter cost is

```text
C_A^t = C_train(A_j) + C_mem(A_j) + C_comm(A_j) + C_match(A_j).
```

Reuse is allowed only when

```text
adapter_match_confidence(A_j, r_t) >= tau_reuse.
```

Otherwise RECA spawns a new adapter or falls back to the no-adapter route.
