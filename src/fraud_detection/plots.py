"""Static report figures (PNG, rendered into README / report.md)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
STATUS = {"stable": "#0ca30c", "moderate": "#fab219", "significant": "#d03b3b"}

# Hue = model family, line style = feature set; the legend carries both.
MODEL_STYLE = {
    "lgbm_velocity": dict(color=BLUE, ls="-", label="LightGBM + velocity"),
    "lgbm_base": dict(color=BLUE, ls=(0, (4, 2)), label="LightGBM, no velocity"),
    "logreg_velocity": dict(color=ORANGE, ls="-", label="LogReg + velocity"),
    "logreg_base": dict(color=ORANGE, ls=(0, (4, 2)), label="LogReg, no velocity"),
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 1,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        st = MODEL_STYLE.get(name, dict(color=MUTED, ls="-", label=name))
        ap = np.sum(np.diff(r[::-1]) * p[::-1][1:])
        ax.plot(r, p, color=st["color"], ls=st["ls"], label=f"{st['label']}  (PR-AUC {ap:.3f})")
    ax.axhline(y.mean(), color=MUTED, lw=1)
    ax.text(
        0.01,
        y.mean(),
        f"fraud rate {y.mean():.1%}",
        color=INK_2,
        ha="left",
        va="bottom",
        fontsize=8,
    )
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.set_title("Precision-recall on the test month")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.08), fontsize=8.5)
    _save(fig, path)


def recall_vs_budget(
    y: np.ndarray, scores: dict[str, np.ndarray], budget: float, path: Path
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    budgets = np.linspace(0.001, 0.10, 100)
    n, total = len(y), max(y.sum(), 1)
    for name, s in scores.items():
        ranked = y[np.argsort(-s, kind="stable")].cumsum()
        rec = [ranked[max(1, int(round(b * n))) - 1] / total for b in budgets]
        st = MODEL_STYLE.get(name, dict(color=MUTED, ls="-", label=name))
        ax.plot(budgets * 100, rec, color=st["color"], ls=st["ls"], label=st["label"])
        if name == "lgbm_velocity":
            r_at = ranked[max(1, int(round(budget * n))) - 1] / total
            ax.plot(
                [budget * 100], [r_at], "o", ms=8, color=st["color"], mec=SURFACE, mew=2, zorder=5
            )
            ax.annotate(
                f"{r_at:.0%} of fraud\nin a {budget:.0%} queue",
                (budget * 100, r_at),
                xytext=(12, -18),
                textcoords="offset points",
                fontsize=9,
                color=INK,
            )
    ax.axvline(budget * 100, color=AXIS, lw=1)
    ax.set(
        xlim=(0, 10),
        ylim=(0, 1),
        xlabel="Review budget (% of transactions sent to analysts)",
        ylabel="Recall (share of fraud caught)",
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Fraud caught vs. manual-review capacity")
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, path)


def importance_bars(imp: pd.DataFrame, velocity: set[str], path: Path, top: int = 20) -> None:
    _style()
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 0.28 * top + 1.2))
    colors = [BLUE if f in velocity else ORANGE for f in d["feature"]]
    ax.barh(d["feature"], d["mean_abs_shap"], height=0.62, color=colors)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title(f"Top {top} features by mean |SHAP|, test month")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(
        handles, ["velocity feature", "raw / static feature"], loc="lower right", fontsize=8.5
    )
    ax.tick_params(axis="y", labelsize=8.5)
    _save(fig, path)


def psi_bars(rep: pd.DataFrame, warn: float, alert: float, path: Path, top: int = 15) -> None:
    _style()
    d = rep.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 0.3 * top + 1.2))
    ax.barh(d["feature"], d["psi"], height=0.62, color=[STATUS[s] for s in d["status"]])
    for x, lab in ((warn, f"{warn:g} moderate"), (alert, f"{alert:g} significant")):
        ax.axvline(x, color=MUTED, lw=1)
        ax.text(x, len(d) - 0.4, f" {lab}", color=INK_2, fontsize=8, va="bottom")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PSI vs. validation month")
    ax.set_title("Largest distribution shifts on the test month")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATUS.values()]
    ax.legend(handles, [f"{k}" for k in STATUS], loc="lower right", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=8.5)
    _save(fig, path)


def daily_alert_rate(
    day: np.ndarray, alert: np.ndarray, budget: float, test_start: int, path: Path
) -> None:
    _style()
    s = pd.Series(alert.astype(float)).groupby(day).mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(s.index - s.index.min(), s.values, color=BLUE)
    ax.axhline(budget, color=MUTED, lw=1)
    ax.text(0, budget, f" budget {budget:.0%}", color=INK_2, fontsize=8, va="bottom")
    ax.axvline(test_start - s.index.min(), color=AXIS, lw=1)
    ax.text(
        test_start - s.index.min(),
        ax.get_ylim()[1],
        " test month →",
        color=INK_2,
        fontsize=8,
        va="top",
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set(xlabel="Day (from start of validation month)", ylabel="Share of transactions flagged")
    ax.set_title("Daily alert rate at the threshold frozen on validation")
    _save(fig, path)


def shap_plots(
    values: np.ndarray, base: np.ndarray, X: pd.DataFrame, row: int, out_dir: Path
) -> list[str]:
    """Beeswarm (global) and waterfall (one alert) using the shap package, if installed."""
    try:
        import shap
    except ImportError:
        return []
    _style()
    exp = shap.Explanation(
        values=values, base_values=base, data=X.to_numpy(), feature_names=list(X.columns)
    )
    shap.plots.beeswarm(exp, max_display=20, show=False, plot_size=(8, 7))
    plt.gcf().savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    shap.plots.waterfall(exp[row], max_display=12, show=False)
    plt.gcf().savefig(out_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    return ["shap_beeswarm.png", "shap_waterfall.png"]
