"""Run the complete credit-risk validation workflow.

Usage:
    python run_analysis.py
"""
from pathlib import Path
import json
import pandas as pd

from src.data import save_portfolio, FEATURES, TARGET
from src.model import build_logistic_model, build_challenger_model
from src.validation import (
    evaluate_model,
    calibration_table,
    population_stability_index,
    bootstrap_confidence_intervals,
    bootstrap_auc_difference,
)
from src.stress import apply_stress, stressed_lgd, expected_credit_loss
from src.plots import (
    save_roc,
    save_calibration,
    save_stress,
    save_backtest,
    save_expected_loss,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "credit_portfolio.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("row_count_positive", len(df) > 0, f"rows={len(df):,}"),
        ("customer_id_unique", df["customer_id"].is_unique, f"duplicates={df['customer_id'].duplicated().sum()}"),
        ("no_missing_values", not df.isna().any().any(), f"missing={int(df.isna().sum().sum())}"),
        ("dti_in_range", df["debt_to_income"].between(0, 1).all(), "expected [0,1]"),
        ("utilization_in_range", df["credit_utilization"].between(0, 1).all(), "expected [0,1]"),
        ("lgd_in_range", df["loss_given_default"].between(0, 1).all(), "expected [0,1]"),
        ("ead_positive", (df["exposure_at_default"] > 0).all(), "expected >0"),
        ("binary_target", df[TARGET].isin([0, 1]).all(), "expected {0,1}"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def rolling_backtest(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in [2023, 2024, 2025]:
        train = df[df["observation_year"] < year].copy()
        test = df[df["observation_year"] == year].copy()
        model = build_logistic_model()
        model.fit(train[FEATURES], train[TARGET])
        prob = model.predict_proba(test[FEATURES])[:, 1]
        metrics = evaluate_model(test[TARGET], prob)
        rows.append({
            "validation_year": year,
            "development_rows": int(len(train)),
            "validation_rows": int(len(test)),
            **metrics,
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = save_portfolio(DATA_PATH, n=12000, seed=42)

    dq = data_quality_summary(df)
    dq.to_csv(OUT / "data_quality_checks.csv", index=False)
    if not dq["passed"].all():
        failed = dq.loc[~dq["passed"], "check"].tolist()
        raise ValueError(f"Data quality checks failed: {failed}")

    # Development: 2022-2024. Validation: 2025 (out-of-time).
    train = df[df["observation_year"] <= 2024].copy()
    test = df[df["observation_year"] == 2025].copy()

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    primary = build_logistic_model()
    challenger = build_challenger_model()
    primary.fit(X_train, y_train)
    challenger.fit(X_train, y_train)

    train_prob = primary.predict_proba(X_train)[:, 1]
    primary_prob = primary.predict_proba(X_test)[:, 1]
    challenger_prob = challenger.predict_proba(X_test)[:, 1]

    primary_metrics = evaluate_model(y_test, primary_prob)
    challenger_metrics = evaluate_model(y_test, challenger_prob)
    primary_metrics["score_psi_train_vs_test"] = population_stability_index(train_prob, primary_prob)

    # Statistical uncertainty around the headline validation metrics.
    ci = bootstrap_confidence_intervals(y_test, primary_prob, n_bootstrap=500, seed=42)
    ci.to_csv(OUT / "primary_metric_confidence_intervals.csv", index=False)
    auc_delta = bootstrap_auc_difference(
        y_test,
        primary_prob,
        challenger_prob,
        n_bootstrap=500,
        seed=43,
    )
    pd.DataFrame([auc_delta]).to_csv(OUT / "challenger_auc_difference.csv", index=False)

    cal = calibration_table(y_test.to_numpy(), primary_prob, bins=10)
    cal.to_csv(OUT / "calibration_by_decile.csv", index=False)

    # Rolling temporal backtest: each year is scored using only prior years.
    backtest = rolling_backtest(df)
    backtest.to_csv(OUT / "annual_backtest.csv", index=False)

    # Translate PD movement into expected credit loss: ECL = PD x EAD x LGD.
    scenarios = []
    ead = test["exposure_at_default"].to_numpy()
    base_lgd = test["loss_given_default"].to_numpy()
    for scenario in ["baseline", "mild", "severe"]:
        stressed = apply_stress(X_test, scenario)
        pd_values = primary.predict_proba(stressed[FEATURES])[:, 1]
        scenario_lgd = stressed_lgd(base_lgd, scenario)
        ecl = expected_credit_loss(pd_values, ead, scenario_lgd)
        scenarios.append({
            "scenario": scenario,
            "mean_pd": float(pd_values.mean()),
            "mean_lgd": float(scenario_lgd.mean()),
            "expected_loss": float(ecl.sum()),
            "expected_loss_rate_on_ead": float(ecl.sum() / ead.sum()),
            "relative_change_vs_baseline": None,
            "loss_change_vs_baseline": None,
        })
    baseline_pd = scenarios[0]["mean_pd"]
    baseline_loss = scenarios[0]["expected_loss"]
    for row in scenarios:
        row["relative_change_vs_baseline"] = (row["mean_pd"] / baseline_pd - 1) if baseline_pd else 0.0
        row["loss_change_vs_baseline"] = (row["expected_loss"] / baseline_loss - 1) if baseline_loss else 0.0
    stress_df = pd.DataFrame(scenarios)
    stress_df.to_csv(OUT / "stress_test_results.csv", index=False)

    realized_loss = float(
        (test[TARGET].to_numpy() * ead * base_lgd).sum()
    )

    comparison = pd.DataFrame([
        {"model": "Logistic regression (primary)", **primary_metrics},
        {"model": "Decision tree (challenger)", **challenger_metrics},
    ])
    comparison.to_csv(OUT / "model_comparison.csv", index=False)

    save_roc(y_test, primary_prob, challenger_prob, FIG / "roc_comparison.png")
    save_calibration(cal, FIG / "calibration.png")
    save_stress(stress_df, FIG / "stress_test.png")
    save_backtest(backtest, FIG / "annual_backtest.png")
    save_expected_loss(stress_df, FIG / "expected_loss.png")

    # Scaled-logistic coefficients: sign and relative magnitude are useful for
    # explaining directional relationships. They are not causal effects.
    coefficients = primary.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({"feature": FEATURES, "scaled_coefficient": coefficients})
    coef_df["odds_ratio_per_1sd"] = coef_df["scaled_coefficient"].map(lambda x: float(__import__("math").exp(x)))
    coef_df["abs_coefficient"] = coef_df["scaled_coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient")
    coef_df.to_csv(OUT / "logistic_coefficients.csv", index=False)

    ci_map = {row["metric"]: row for row in ci.to_dict(orient="records")}
    results = {
        "portfolio_rows": int(len(df)),
        "development_rows": int(len(train)),
        "oot_validation_rows": int(len(test)),
        "development_default_rate": float(y_train.mean()),
        "oot_default_rate": float(y_test.mean()),
        "total_oot_ead": float(ead.sum()),
        "realized_oot_loss": realized_loss,
        "primary": primary_metrics,
        "primary_confidence_intervals": ci_map,
        "challenger": challenger_metrics,
        "auc_difference_primary_minus_challenger": auc_delta,
        "stress": scenarios,
        "annual_backtest": backtest.to_dict(orient="records"),
        "data_quality_checks_passed": int(dq["passed"].sum()),
        "data_quality_checks_total": int(len(dq)),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    auc_ci = ci_map["roc_auc"]
    print("\nCREDIT RISK MODEL VALIDATION")
    print("-" * 64)
    print(f"Rows: {len(df):,} | Dev: {len(train):,} | OOT: {len(test):,}")
    print(f"Data-quality checks: {dq['passed'].sum()}/{len(dq)} passed")
    print(f"OOT observed default rate: {y_test.mean():.2%}")
    print("\nPrimary model")
    print(f"  ROC-AUC:    {primary_metrics['roc_auc']:.3f} (95% CI {auc_ci['ci_lower']:.3f}-{auc_ci['ci_upper']:.3f})")
    print(f"  KS:         {primary_metrics['ks_statistic']:.3f}")
    print(f"  Brier:      {primary_metrics['brier_score']:.3f}")
    print(f"  Score PSI:  {primary_metrics['score_psi_train_vs_test']:.3f}")
    print("\nChallenger")
    print(f"  ROC-AUC:    {challenger_metrics['roc_auc']:.3f}")
    print(f"  AUC delta:  {auc_delta['auc_difference']:+.3f} (95% CI {auc_delta['ci_lower']:+.3f} to {auc_delta['ci_upper']:+.3f})")
    print("\nRolling backtest")
    for row in backtest.to_dict(orient="records"):
        print(f"  {int(row['validation_year'])}: AUC {row['roc_auc']:.3f} | KS {row['ks_statistic']:.3f} | Brier {row['brier_score']:.3f}")
    print("\nStress / expected loss")
    for r in scenarios:
        print(
            f"  {r['scenario'].title():8s}: PD {r['mean_pd']:.2%} | "
            f"ECL ${r['expected_loss']:,.0f} ({r['loss_change_vs_baseline']:+.1%} vs baseline)"
        )
    print(f"\nOutputs saved under: {OUT}")


if __name__ == "__main__":
    main()
