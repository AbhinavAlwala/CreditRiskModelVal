"""Model validation metrics used in the project."""
from collections.abc import Callable
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve


def ks_statistic(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def calibration_table(y_true, y_prob, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": y_true, "predicted": y_prob}).copy()
    frame["decile"] = pd.qcut(frame["predicted"], q=bins, duplicates="drop")
    result = (
        frame.groupby("decile", observed=True)
        .agg(
            borrowers=("actual", "size"),
            predicted_pd=("predicted", "mean"),
            observed_default_rate=("actual", "mean"),
        )
        .reset_index(drop=True)
    )
    result["calibration_gap"] = result["predicted_pd"] - result["observed_default_rate"]
    return result


def population_stability_index(expected, actual, bins: int = 10) -> float:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def evaluate_model(y_true, y_prob) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "ks_statistic": ks_statistic(y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "mean_predicted_pd": float(np.mean(y_prob)),
        "observed_default_rate": float(np.mean(y_true)),
    }


def bootstrap_confidence_intervals(
    y_true,
    y_prob,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired bootstrap confidence intervals for core validation metrics.

    Sampling is performed at the observation level. Replicates containing a
    single target class are skipped because ROC-AUC and KS are undefined.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    metric_fns: dict[str, Callable] = {
        "roc_auc": roc_auc_score,
        "ks_statistic": ks_statistic,
        "brier_score": brier_score_loss,
    }
    samples = {name: [] for name in metric_fns}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if np.unique(y_b).size < 2:
            continue
        for name, fn in metric_fns.items():
            samples[name].append(float(fn(y_b, p_b)))

    alpha = 1 - confidence
    rows = []
    point = evaluate_model(y_true, y_prob)
    for name, values in samples.items():
        arr = np.asarray(values)
        rows.append({
            "metric": name,
            "estimate": point[name],
            "ci_lower": float(np.quantile(arr, alpha / 2)),
            "ci_upper": float(np.quantile(arr, 1 - alpha / 2)),
            "bootstrap_replicates": int(len(arr)),
        })
    return pd.DataFrame(rows)


def bootstrap_auc_difference(
    y_true,
    primary_prob,
    challenger_prob,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 43,
) -> dict:
    """Paired-bootstrap CI for AUC(primary) - AUC(challenger)."""
    y_true = np.asarray(y_true)
    primary_prob = np.asarray(primary_prob)
    challenger_prob = np.asarray(challenger_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if np.unique(y_b).size < 2:
            continue
        deltas.append(
            roc_auc_score(y_b, primary_prob[idx])
            - roc_auc_score(y_b, challenger_prob[idx])
        )

    alpha = 1 - confidence
    deltas = np.asarray(deltas)
    point = roc_auc_score(y_true, primary_prob) - roc_auc_score(y_true, challenger_prob)
    return {
        "auc_difference": float(point),
        "ci_lower": float(np.quantile(deltas, alpha / 2)),
        "ci_upper": float(np.quantile(deltas, 1 - alpha / 2)),
        "bootstrap_replicates": int(len(deltas)),
    }
