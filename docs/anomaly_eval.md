# Anomaly detection evaluation

Injection protocol: 30 known anomalies planted in the real 1096-day series per run, 20 random seeds, averaged. The warehouse has no real labels, so this measures detection power for the stated magnitudes, not performance on unknown real incidents.

| Detector | Precision | Recall | F1 | Days flagged / run | Flags on clean data |
|---|---|---|---|---|---|
| robust_z_residual | 0.80 | 0.79 | 0.79 | 29 | 6 |
| isolation_forest_residual | 0.62 | 0.68 | 0.65 | 33 | 33 |
| legacy_isolation_forest | 0.37 | 0.41 | 0.39 | 33 | 33 |

## Recall by anomaly type

| Detector | volume_spike_1.6x | volume_spike_2.0x | volume_dip_0.5x | returns_spike_3x | returns_spike_4x |
|---|---|---|---|---|---|
| legacy_isolation_forest | 0.23 | 0.58 | 0.17 | 0.44 | 0.64 |
| isolation_forest_residual | 0.56 | 0.94 | 0.98 | 0.33 | 0.58 |
| robust_z_residual | 0.66 | 0.97 | 1.00 | 0.57 | 0.73 |