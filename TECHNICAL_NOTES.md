# Technical Notes

## Validation questions

This repository is designed around five questions:

1. **Discrimination:** can the model rank higher-risk borrowers above lower-risk borrowers?
2. **Calibration:** do predicted probabilities resemble realized default frequencies?
3. **Stability:** does the score distribution move materially between development and later vintages?
4. **Benchmarking:** does a structurally different challenger outperform the primary model?
5. **Financial sensitivity:** how does worsening borrower risk translate into expected loss?

## Metrics

### ROC-AUC
Measures rank discrimination. It is insensitive to absolute probability calibration, so it is never treated as sufficient by itself.

### KS statistic
Maximum separation between cumulative score distributions for defaults and non-defaults.

### Brier score
Mean squared error of predicted probabilities. Lower is better and the metric is sensitive to probability quality.

### Calibration by decile
Borrowers are grouped into predicted-risk buckets and mean predicted PD is compared with observed default frequency.

### Population Stability Index (PSI)
Used here as a descriptive score-distribution monitoring statistic. No universal regulatory threshold is assumed.

## Bootstrap confidence intervals

The 2025 OOT sample is resampled with replacement 500 times. Metrics are recomputed for each replicate and percentile intervals are reported. For the primary-vs-challenger comparison, the same bootstrap indices are applied to both models so the AUC difference is paired at borrower level.

## Rolling out-of-time backtest

For each validation year, the logistic model is retrained using only earlier vintages:

- validate 2023 using 2022 development data
- validate 2024 using 2022-2023 development data
- validate 2025 using 2022-2024 development data

This avoids using future outcomes when evaluating earlier vintages.

## Expected credit loss

The project uses the educational identity:

`Expected Credit Loss = PD x EAD x LGD`

PD comes from the logistic model. EAD and LGD are synthetic portfolio attributes generated independently of model training.

## Stress scenarios

### Mild
- annual income -10%
- debt-to-income +10%
- utilization +5 percentage points
- LGD +5 percentage points

### Severe
- annual income -20%
- debt-to-income +22%
- utilization +12 percentage points
- one additional recent inquiry
- LGD +10 percentage points

These scenarios are intended to demonstrate sensitivity analysis. They are not macroeconomic forecasts or regulatory stress scenarios.

## Main model-risk conclusion

The 2025 model retains useful discrimination but underpredicts the aggregate default rate. This illustrates a key distinction: a model can continue ranking borrowers effectively while its absolute PD estimates drift away from realized outcomes. That would motivate recalibration and further validation rather than an automatic production approval.
