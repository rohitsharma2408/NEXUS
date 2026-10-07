# Forecast prediction intervals

Split-conformal intervals around the blend forecast (0.5 level_x_season + 0.5 xgb_seasonal), calibrated walk-forward on the previous half-year's errors, scaled by sqrt(predicted units). Coverage is measured on the test half-year, so it checks the intervals on data they never saw. The category-month width uses a fixed spread factor K=1.5 relative to independent product errors: errors of products in the same category are positively correlated, and the realised factor was 1.13, 1.38 and about 1.5 in the three completed half-years after the first (the first, 2.47, came from a model with only 12 months of history). Coverage reported for those folds is therefore in-sample for that one constant.

## 80% intervals

Per product-month:

| Test fold | Rows | Coverage | Mean width (units) | Scaled quantile |
|---|---|---|---|---|
| 2023-H2 | 3000 | 78.7% | 18.4 | 2.46 |
| 2024-H1 | 3000 | 82.5% | 16.3 | 2.51 |
| 2024-H2 | 3000 | 76.2% | 18.0 | 2.38 |
| **all** | 9000 | **79.2%** | 17.6 | |

By predicted volume (pooled):

| Volume | Mean forecast | Coverage | Mean width |
|---|---|---|---|
| low | 7.6 | 82.3% | 13.6 |
| mid | 11.0 | 78.3% | 16.2 |
| high | 23.9 | 76.8% | 23.0 |

Per category-month (summed forecast):

| Test fold | Category-months | Coverage | Half-width (% of forecast) |
|---|---|---|---|
| 2023-H2 | 42 | 90.5% | ±12.1% |
| 2024-H1 | 42 | 81.0% | ±14.6% |
| 2024-H2 | 42 | 78.6% | ±12.2% |

## 90% intervals

Per product-month:

| Test fold | Rows | Coverage | Mean width (units) | Scaled quantile |
|---|---|---|---|---|
| 2023-H2 | 3000 | 88.4% | 22.1 | 2.96 |
| 2024-H1 | 3000 | 91.6% | 19.5 | 3.09 |
| 2024-H2 | 3000 | 86.9% | 22.0 | 2.91 |
| **all** | 9000 | **89.0%** | 21.2 | |

By predicted volume (pooled):

| Volume | Mean forecast | Coverage | Mean width |
|---|---|---|---|
| low | 7.6 | 91.4% | 15.9 |
| mid | 11.0 | 89.0% | 19.7 |
| high | 23.9 | 86.6% | 27.9 |

Per category-month (summed forecast):

| Test fold | Category-months | Coverage | Half-width (% of forecast) |
|---|---|---|---|
| 2023-H2 | 42 | 95.2% | ±15.5% |
| 2024-H1 | 42 | 97.6% | ±18.8% |
| 2024-H2 | 42 | 92.9% | ±15.6% |

