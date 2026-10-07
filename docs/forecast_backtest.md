# Demand forecast backtest

Walk-forward, 1-step-ahead (predict next month's units per product). Lower is better.
Each fold trains only on months before its test window; the seasonal index is also
estimated from that window only.

| Model | MAE | RMSE | WAPE | Category-month WAPE | Skill vs naive |
|---|---|---|---|---|---|
| level_x_season | 5.79 | 7.64 | 42.6% | 6.3% | +28.5% |
| blend_seasonal_xgb | 5.84 | 7.68 | 43.1% | 6.9% | +27.7% |
| xgb_seasonal | 6.00 | 7.88 | 44.3% | 9.4% | +25.8% |
| moving_avg_12 | 6.32 | 8.86 | 46.8% | 18.5% | +21.8% |
| moving_avg_3 | 7.03 | 9.88 | 52.6% | 22.6% | +13.1% |
| seasonal_naive | 7.70 | 10.29 | 56.8% | 7.3% | +4.7% |
| naive_last_month | 8.09 | 11.29 | 60.5% | 19.0% | +0.0% |
| legacy_xgb | 8.81 | 12.66 | 68.3% | 43.5% | -8.9% |

Best by MAE: **level_x_season**. xgb_seasonal vs the previous model: **+31.8% MAE**.

## Per fold (MAE)

| Fold | naive_last_month | moving_avg_3 | moving_avg_12 | seasonal_naive | level_x_season | xgb_seasonal | blend_seasonal_xgb | legacy_xgb |
|---|---|---|---|---|---|---|---|---|
| 2023-H2 | 8.03 | 6.97 | 6.63 | 8.06 | 6.11 | 6.30 | 6.16 | 7.39 |
| 2024-H1 | 8.06 | 7.07 | 5.72 | 6.77 | 5.02 | 5.32 | 5.12 | 11.64 |
| 2024-H2 | 8.18 | 7.05 | 6.62 | 8.28 | 6.22 | 6.38 | 6.25 | 7.39 |

Product-month sales are noisy counts (mean about 13 units, lag-1 autocorrelation about 0.05),
so a large part of the error is irreducible at the single-product level. The
category-month column shows the error after products are summed, which is the level
planning decisions are made at.