"""Data generation and loading utilities.

The repository ships with a reproducible synthetic portfolio rather than
customer data. Variables are designed to resemble common unsecured
credit-risk drivers without representing any real borrower.
"""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "annual_income",
    "employment_years",
    "debt_to_income",
    "credit_utilization",
    "credit_history_years",
    "delinquencies_2y",
    "inquiries_6m",
]
TARGET = "default_12m"
LOSS_COLUMNS = ["exposure_at_default", "loss_given_default"]


def generate_portfolio(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    year = rng.choice([2022, 2023, 2024, 2025], size=n, p=[0.22, 0.25, 0.27, 0.26])
    age = np.clip(rng.normal(41, 11, n).round(), 21, 75).astype(int)

    annual_income = np.clip(rng.lognormal(np.log(65000), 0.45, n), 18000, 250000)
    employment_years = np.clip(rng.gamma(2.2, 3.1, n), 0, 35)
    employment_years = np.minimum(employment_years, np.maximum(age - 18, 0))

    debt_to_income = np.clip(rng.beta(2.3, 4.8, n) * 0.95, 0.02, 0.92)
    credit_utilization = np.clip(rng.beta(2.0, 3.0, n), 0.01, 0.99)
    credit_history_years = np.clip(age - 18 - rng.gamma(2.0, 2.3, n), 0.5, 45)
    delinquencies_2y = np.clip(rng.poisson(0.35 + 0.8 * credit_utilization, n), 0, 6)
    inquiries_6m = np.clip(rng.poisson(0.7 + 1.0 * debt_to_income, n), 0, 7)

    # Synthetic exposure and severity fields are kept OUT of the PD model.
    # They let the project translate PD into expected credit loss (ECL).
    credit_limit = np.clip(
        annual_income * rng.uniform(0.08, 0.35, n),
        2_000,
        40_000,
    )
    exposure_at_default = np.clip(
        credit_limit * (0.25 + 0.75 * credit_utilization),
        500,
        40_000,
    )
    loss_given_default = np.clip(
        0.28
        + 0.24 * credit_utilization
        + 0.13 * debt_to_income
        + rng.normal(0, 0.05, n),
        0.15,
        0.85,
    )

    # Slightly more difficult macro environment in 2025. This makes the
    # out-of-time validation meaningful without creating unrealistically
    # separable data.
    macro = np.where(year == 2025, 0.28, np.where(year == 2024, 0.08, 0.0))

    income_k = annual_income / 1000
    logit = (
        -4.35
        + 2.8 * debt_to_income
        + 2.35 * credit_utilization
        + 0.48 * delinquencies_2y
        + 0.22 * inquiries_6m
        - 0.020 * credit_history_years
        - 0.010 * (income_k - 60)
        + macro
    )
    p_default = 1 / (1 + np.exp(-logit))
    default_12m = rng.binomial(1, p_default)

    df = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
        "observation_year": year,
        "age": age,
        "annual_income": annual_income.round(2),
        "employment_years": employment_years.round(1),
        "debt_to_income": debt_to_income.round(4),
        "credit_utilization": credit_utilization.round(4),
        "credit_history_years": credit_history_years.round(1),
        "delinquencies_2y": delinquencies_2y.astype(int),
        "inquiries_6m": inquiries_6m.astype(int),
        "exposure_at_default": exposure_at_default.round(2),
        "loss_given_default": loss_given_default.round(4),
        "default_12m": default_12m.astype(int),
    })
    return df.sort_values(["observation_year", "customer_id"]).reset_index(drop=True)


def save_portfolio(path: str | Path, n: int = 12000, seed: int = 42) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_portfolio(n=n, seed=seed)
    df.to_csv(path, index=False)
    return df


def load_portfolio(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
