# RECA-FL TWC Logging Schema

Every TWC experiment should log enough per-round information to analyze
learning, wireless communication, sensing, RECA mechanism behavior, and
overhead.

## Learning Logs

- `round`
- `method`
- `seed`
- `train_loss`
- `test_loss`
- `test_accuracy`
- `rare_class_accuracy`
- `macro_f1`
- `gradient_norm_proxy`
- `selected_client_ids`
- `local_epochs`
- `aggregation_rule`

## Wireless Logs

- `channel_gains`
- `uplink_snr`
- `sensing_snr`
- `transmit_power`
- `bandwidth_allocation`
- `resource_allocation`
- `aircomp_mse`
- `aircomp_mse_cvar`
- `latency`
- `energy`
- `outage_indicator`
- `mse_violation`
- `latency_violation`
- `energy_violation`
- `power_violation`

## Sensing Logs

- `crb`
- `crb_cvar`
- `fim_logdet`
- `coverage`
- `worst_region_localization_error`
- `target_motion_regime_id`
- `blockage_state`
- `sensing_clutter_state`
- `sensing_resource_allocation`

## RECA Mechanism Logs

- `risk_score`
- `mismatch_score`
- `progress_score`
- `overwhelm_score`
- `trigger_score`
- `trigger_decision`
- `world_model_prediction_error`
- `world_model_rmse`
- `world_model_ece`
- `adapter_id`
- `adapter_spawned`
- `adapter_active`
- `adapter_consolidated`
- `adapter_reused`
- `adapter_quarantined`
- `adapter_embedding_distance`
- `adapter_residual_cosine`
- `adapter_residual_kl`
- `adapter_match_confidence`
- `wrong_reuse_flag`
- `false_reuse_flag`
- `false_trigger_label`
- `missed_trigger_label`

## Overhead Logs

- `probe_time`
- `selection_time`
- `world_model_update_time`
- `adapter_match_time`
- `adapter_update_time`
- `adapter_memory_bytes`
- `extra_probe_bytes`
- `extra_flops`
- `communication_payload_bytes`
- `total_round_time`

## Summary Logs

Each run should additionally produce method-level summaries:

- final accuracy,
- best accuracy,
- convergence rounds,
- mean CRB,
- CRB-CVaR,
- mean AirComp MSE,
- AirComp-MSE-CVaR,
- mean energy,
- mean latency,
- violation rates,
- recovery rounds after shift,
- adapter reuse rate,
- false reuse rate,
- wrong-reuse penalty,
- confidence intervals across seeds.
