import numpy as np
import pandas as pd
from src.data import generate_portfolio, FEATURES
from src.stress import apply_stress, stressed_lgd, expected_credit_loss
from src.validation import ks_statistic, bootstrap_confidence_intervals


def test_portfolio_shape_and_ranges():
    df = generate_portfolio(n=500, seed=7)
    assert len(df) == 500
    assert set(FEATURES).issubset(df.columns)
    assert df["default_12m"].isin([0, 1]).all()
    assert df["debt_to_income"].between(0, 1).all()
    assert df["credit_utilization"].between(0, 1).all()
    assert df["loss_given_default"].between(0, 1).all()
    assert (df["exposure_at_default"] > 0).all()


def test_age_is_not_a_pd_feature():
    assert "age" not in FEATURES


def test_stress_moves_risk_drivers_in_expected_direction():
    df = generate_portfolio(n=50, seed=8)[FEATURES]
    mild = apply_stress(df, "mild")
    assert (mild["annual_income"] <= df["annual_income"]).all()
    assert (mild["debt_to_income"] >= df["debt_to_income"]).all()


def test_expected_loss_increases_with_stressed_lgd():
    pd_values = np.array([0.05, 0.10, 0.20])
    ead = np.array([10_000, 12_000, 8_000])
    lgd = np.array([0.35, 0.40, 0.45])
    base = expected_credit_loss(pd_values, ead, stressed_lgd(lgd, "baseline")).sum()
    severe = expected_credit_loss(pd_values, ead, stressed_lgd(lgd, "severe")).sum()
    assert severe > base


def test_ks_is_bounded():
    y = pd.Series([0, 0, 1, 1])
    p = pd.Series([0.1, 0.4, 0.6, 0.9])
    assert 0 <= ks_statistic(y, p) <= 1


def test_bootstrap_ci_contains_point_estimate_for_simple_case():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    p = np.array([0.05, 0.15, 0.25, 0.70, 0.80, 0.90, 0.35, 0.65, 0.20, 0.75])
    ci = bootstrap_confidence_intervals(y, p, n_bootstrap=100, seed=1)
    auc_row = ci.loc[ci["metric"] == "roc_auc"].iloc[0]
    assert auc_row["ci_lower"] <= auc_row["estimate"] <= auc_row["ci_upper"]
