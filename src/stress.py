"""Simple borrower-level stress scenarios and expected-loss assumptions."""
import numpy as np
import pandas as pd


def apply_stress(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    stressed = df.copy()
    if scenario == "baseline":
        return stressed
    if scenario == "mild":
        stressed["annual_income"] *= 0.90
        stressed["debt_to_income"] = (stressed["debt_to_income"] * 1.10).clip(upper=0.99)
        stressed["credit_utilization"] = (stressed["credit_utilization"] + 0.05).clip(upper=0.99)
        return stressed
    if scenario == "severe":
        stressed["annual_income"] *= 0.80
        stressed["debt_to_income"] = (stressed["debt_to_income"] * 1.22).clip(upper=0.99)
        stressed["credit_utilization"] = (stressed["credit_utilization"] + 0.12).clip(upper=0.99)
        stressed["inquiries_6m"] = (stressed["inquiries_6m"] + 1).clip(upper=7)
        return stressed
    raise ValueError("scenario must be 'baseline', 'mild', or 'severe'")


def stressed_lgd(base_lgd, scenario: str):
    """Illustrative severity sensitivity used only for ECL translation."""
    base_lgd = np.asarray(base_lgd, dtype=float)
    add = {"baseline": 0.00, "mild": 0.05, "severe": 0.10}
    if scenario not in add:
        raise ValueError("scenario must be 'baseline', 'mild', or 'severe'")
    return np.clip(base_lgd + add[scenario], 0.0, 1.0)


def expected_credit_loss(pd_values, ead, lgd) -> np.ndarray:
    """Account-level expected credit loss: PD x EAD x LGD."""
    return np.asarray(pd_values) * np.asarray(ead) * np.asarray(lgd)
