-- Credit-risk portfolio analysis examples (SQLite-compatible)
-- Demonstrates segmentation, concentration, and observed-loss analysis a risk
-- analyst could use before or after model development.

-- 1) Portfolio size, exposure and observed default rate by year
SELECT
    observation_year,
    COUNT(*) AS borrowers,
    ROUND(SUM(exposure_at_default), 0) AS total_ead,
    ROUND(AVG(default_12m) * 100, 2) AS default_rate_pct
FROM credit_portfolio
GROUP BY observation_year
ORDER BY observation_year;

-- 2) Default rate and observed loss by debt-to-income band
SELECT
    CASE
        WHEN debt_to_income < 0.20 THEN '<20%'
        WHEN debt_to_income < 0.40 THEN '20-40%'
        WHEN debt_to_income < 0.60 THEN '40-60%'
        ELSE '60%+'
    END AS dti_band,
    COUNT(*) AS borrowers,
    ROUND(SUM(exposure_at_default), 0) AS total_ead,
    ROUND(AVG(default_12m) * 100, 2) AS default_rate_pct,
    ROUND(SUM(default_12m * exposure_at_default * loss_given_default), 0) AS realized_loss
FROM credit_portfolio
GROUP BY dti_band
ORDER BY AVG(debt_to_income);

-- 3) High-risk behavioural segment
SELECT
    COUNT(*) AS borrowers,
    ROUND(SUM(exposure_at_default), 0) AS total_ead,
    ROUND(AVG(default_12m) * 100, 2) AS default_rate_pct,
    ROUND(AVG(annual_income), 0) AS avg_income
FROM credit_portfolio
WHERE credit_utilization >= 0.75
  AND delinquencies_2y >= 2;

-- 4) Income and utilization cross-segmentation
SELECT
    CASE
        WHEN annual_income < 40000 THEN '<40k'
        WHEN annual_income < 80000 THEN '40k-80k'
        ELSE '80k+'
    END AS income_band,
    CASE
        WHEN credit_utilization < 0.30 THEN 'Low utilization'
        WHEN credit_utilization < 0.70 THEN 'Medium utilization'
        ELSE 'High utilization'
    END AS utilization_band,
    COUNT(*) AS borrowers,
    ROUND(SUM(exposure_at_default), 0) AS total_ead,
    ROUND(AVG(default_12m) * 100, 2) AS default_rate_pct
FROM credit_portfolio
GROUP BY income_band, utilization_band
ORDER BY income_band, utilization_band;

-- 5) Vintage monitoring: portfolio mix, default rate, and realized loss
SELECT
    observation_year,
    ROUND(AVG(debt_to_income), 3) AS avg_dti,
    ROUND(AVG(credit_utilization), 3) AS avg_utilization,
    ROUND(AVG(default_12m) * 100, 2) AS default_rate_pct,
    ROUND(SUM(default_12m * exposure_at_default * loss_given_default), 0) AS realized_loss
FROM credit_portfolio
GROUP BY observation_year
ORDER BY observation_year;
