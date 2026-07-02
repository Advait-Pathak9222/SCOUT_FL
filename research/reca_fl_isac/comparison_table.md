# RECA-FL vs External Method Families

| Method Family | Main Decision | Wireless / ISAC Awareness | Handles Non-Stationarity | Tail-Risk Handling | Representation Accommodation | Difference From RECA-FL |
|---|---|---|---|---|---|---|
| FedAvg / FedProx | Local training and aggregation | No explicit wireless/ISAC model | Limited to optimization robustness | No | No | Baseline FL layer without AirComp/CRB/resource-aware accommodation. |
| FedCS / resource-aware FL | Client selection | Communication latency/resource aware | No explicit regime memory | Resource feasibility only | No | Selects feasible clients but does not adapt world representation. |
| Oort / utility-based FL | Client selection | Usually no ISAC model | Limited exploration | No wireless tail risk | No | Optimizes utility and speed without CRB/AirComp accommodation. |
| FedCor / correlation-aware selection | Client selection | Usually no ISAC model | Some uncertainty modeling | No wireless tail risk | No | Uses statistical correlation but not wireless regime adapters. |
| OTA-FL / AirComp-FedAvg | OTA aggregation | AirComp distortion aware | No context memory | MSE constraints possible | No | Models wireless aggregation but does not trigger representation change. |
| OTA-FEEL power control | Power/resource allocation | Strong wireless awareness | Usually stationary or slowly varying | MSE/resource constraints | No | Optimizes PHY resources under fixed learning representation. |
| ISCC / ISAC-FL resource allocation | Joint sensing/communication/learning resources | Strong ISAC awareness | Usually handled through robust allocation | CRB/MSE constraints possible | No | Optimizes resources but does not spawn/reuse context adapters. |
| Sensing-native OTA-FL | Reuse learning signals for sensing | Strong sensing-learning coupling | Limited | Depends on formulation | No | Couples sensing and learning but lacks confidence-gated adapter reuse. |
| RECA-FL | Client/resource selection + context accommodation | AirComp, CRB, latency, energy, sensing-resource aware | Yes, via mismatch-triggered adapter lifecycle | CVaR and overwhelm control | Yes | Converts bounded wireless/sensing mismatch into reusable regime-specific adapters. |

## Novelty in One Sentence

RECA-FL treats non-stationary wireless ISAC/OTA-FEEL events as candidates for
risk-bounded representation accommodation, and it reuses stored adapters only
when adapter-regime similarity is sufficiently confident.
