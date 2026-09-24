"""Plotting helpers."""
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


def save_roc(y_true, primary_prob, challenger_prob, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    RocCurveDisplay.from_predictions(y_true, primary_prob, name="Logistic (primary)", ax=ax)
    RocCurveDisplay.from_predictions(y_true, challenger_prob, name="Decision tree (challenger)", ax=ax)
    ax.set_title("Out-of-time ROC comparison")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_calibration(cal_table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(cal_table["predicted_pd"], cal_table["observed_default_rate"], marker="o", label="Observed")
    lim = max(cal_table["predicted_pd"].max(), cal_table["observed_default_rate"].max()) * 1.08
    ax.plot([0, lim], [0, lim], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability of default")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration by predicted-risk decile")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_stress(stress_df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(stress_df["scenario"], stress_df["mean_pd"] * 100)
    ax.set_ylabel("Portfolio mean predicted PD (%)")
    ax.set_title("Borrower-level stress test")
    for i, value in enumerate(stress_df["mean_pd"] * 100):
        ax.text(i, value + 0.15, f"{value:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_backtest(backtest_df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = backtest_df["validation_year"].astype(str)
    ax.plot(x, backtest_df["roc_auc"], marker="o", label="ROC-AUC")
    ax.plot(x, backtest_df["ks_statistic"], marker="o", label="KS")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric value")
    ax.set_xlabel("Validation vintage")
    ax.set_title("Rolling out-of-time backtest")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_expected_loss(stress_df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    values = stress_df["expected_loss"] / 1_000_000
    ax.bar(stress_df["scenario"], values)
    ax.set_ylabel("Expected credit loss ($ millions)")
    ax.set_title("Expected-loss sensitivity")
    for i, value in enumerate(values):
        ax.text(i, value + max(values) * 0.015, f"${value:.2f}M", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
