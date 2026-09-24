# Model Card - 12-Month Probability of Default

## Intended use
Demonstrate an end-to-end credit-risk model development, validation, monitoring and stress-testing workflow for a technical portfolio.

## Not intended for
Real lending decisions, pricing, adverse action, regulatory reporting, capital calculation, or production deployment.

## Data
12,000 reproducible synthetic borrower observations spanning 2022-2025. Development uses 2022-2024; 2025 is held out for final out-of-time validation. EAD and LGD are synthetic loss-translation fields and are not PD predictors. Age is also excluded from PD inputs and retained only as a non-model synthetic portfolio attribute.

## PD features
- annual income
- employment years
- debt-to-income ratio
- credit utilization
- credit-history years
- delinquencies in the last two years
- inquiries in the last six months

## Target
`default_12m`: binary indicator of simulated default within 12 months.

## Primary model
Scaled logistic regression.

## Challenger
Shallow decision tree (`max_depth=4`, `min_samples_leaf=150`).

## 2025 OOT performance
- ROC-AUC: **0.763**; 95% bootstrap CI **0.739-0.789**
- KS: **0.378**; 95% bootstrap CI **0.347-0.434**
- Brier score: **0.098**; 95% bootstrap CI **0.090-0.107**
- Mean predicted PD: **10.85%**
- Observed default rate: **12.81%**
- Score PSI vs development: **0.003**

## Challenger comparison
- Challenger ROC-AUC: **0.697**
- Primary-minus-challenger AUC: **+0.066**
- Paired-bootstrap 95% CI: **+0.045 to +0.086**

## Temporal backtest
- 2023 AUC: **0.728**
- 2024 AUC: **0.741**
- 2025 AUC: **0.763**

Each year is evaluated using a model trained only on prior vintages.

## Expected-loss sensitivity
- Baseline ECL: **$1.28M**
- Mild stress ECL: **$1.77M (+37.9%)**
- Severe stress ECL: **$2.94M (+129.7%)**

ECL uses the illustrative identity `PD x EAD x LGD`.

## Interpretation
The primary model provides useful rank ordering and materially outperforms the challenger on the 2025 OOT portfolio. However, mean predicted PD remains below the realized default rate, so the model demonstrates a calibration-monitoring issue. The model should not be interpreted as production-ready.

## Limitations
Synthetic data; illustrative loss severity and stress assumptions; limited temporal history; no causal interpretation; no fair-lending or regulatory review; no production governance controls.
