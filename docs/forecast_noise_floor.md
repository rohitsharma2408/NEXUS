# Forecast noise floor

Hand-written findings that sit next to the generated backtest in `forecast_backtest.md`. They live in
their own file because `evaluation/forecast_backtest.py` rewrites that one from scratch on every run.
Prediction intervals are in `forecast_intervals.md`.

## Is the per-product error reducible?

An oracle that knows each product's true mean level and the seasonal index reaches 41.0% WAPE in-sample,
against 42.6% for the best walk-forward model, so modelling headroom from level and season is under 2
points. A pure Poisson model would give 21%, so the data is about twice as noisy as sampling noise alone.
Error falls with volume (about 57% for products selling ~7 units a month, 31% for ~24). Price changes
show no measurable effect on units sold (mean surprise after a >2% price cut: 0.000; after a rise: -0.002;
correlation 0.002), so price is not a useful feature, which is consistent with `validate_elasticity.py`
rejecting the static price_elasticity values.

## What else could explain the extra noise?

Checked on the warehouse: there is no shared month or category-month shock (variance explained equals the
chance level), promotions, competitor price and price index have no measurable effect on units sold,
marketing spend is not significant (36 monthly points, correlation about 0.25), `units_sold` equals the
transaction quantity in `fact_sales` exactly (correlation 1.000), and residuals have no month-to-month
carryover (lag-1 autocorrelation -0.024). The remaining error is independent per product-month, and no
data in the warehouse explains it.
