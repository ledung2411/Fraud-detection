"""Markdown report generated from the metrics of a training run."""

from __future__ import annotations

from pathlib import Path

from .explain import describe

NAMES = {
    "logreg_base": "Logistic Regression - no velocity",
    "logreg_velocity": "Logistic Regression + velocity",
    "lgbm_base": "LightGBM - no velocity",
    "lgbm_velocity": "**LightGBM + velocity**",
}


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def _ci(c: list | tuple, pct: bool = False) -> str:
    lo, hi = c
    return f"[{_pct(lo)}, {_pct(hi)}]" if pct else f"[{lo:.3f}, {hi:.3f}]"


def write_report(s: dict, path: Path, velocity: set[str]) -> None:
    b = s["review_budget"]
    tag = f"{b * 100:g}pct"
    res, unc = s["results"], s["bootstrap_95ci"]
    sp = s["split"]
    L = [f"# Fraud detection report - `{s['data_source']}` data", ""]
    if s["data_source"] == "synthetic":
        L += [
            "> **Synthetic data.** These numbers come from the IEEE-CIS-shaped generator in",
            "> `src/fraud_detection/synthetic.py`; they show that the pipeline works end to end,",
            "> not how the model performs on real transactions. Run `make train-ieee` for that.",
            "",
        ]
    L += [
        f"{s['rows']:,} transactions, fraud rate {_pct(s['fraud_rate'], 2)}. "
        f"Review budget: **{_pct(b, 1)} of transactions** can be checked by analysts.",
        "",
        "## Out-of-time split",
        "",
        "| Period | Days | Rows | Fraud rate | Used for |",
        "|---|---|---|---|---|",
    ]
    usage = {
        "train": "fitting models and encoders",
        "valid": "early stopping, choosing the alert threshold, drift reference",
        "test": "all reported numbers",
    }
    for k in ("train", "valid", "test"):
        d0, d1 = sp[f"{k}_days"]
        fr = res["lgbm_velocity"][k]["fraud_rate"] if k != "train" else None
        L.append(
            f"| {k} | {d0}-{d1} | {sp[k]:,} | {_pct(fr, 2) if fr is not None else '-'} | {usage[k]} |"
        )

    L += [
        "",
        "## Results on the test month",
        "",
        f"| Model | PR-AUC [95% CI] | ROC-AUC | Recall @ {_pct(b, 0)} [95% CI] | Precision @ {_pct(b, 0)} "
        f"| Fraud $ caught @ {_pct(b, 0)} |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in res.items():
        t = r["test"]
        L.append(
            f"| {NAMES.get(name, name)} | {t['pr_auc']:.3f} {_ci(unc['pr_auc'][name])} | {t['roc_auc']:.3f} "
            f"| {_pct(t[f'recall@{tag}'])} {_ci(unc['recall_at_budget'][name], pct=True)} "
            f"| {_pct(t[f'precision@{tag}'])} | {_pct(t[f'amount_recall@{tag}'])} |"
        )
    fr = res["lgbm_velocity"]["test"]["fraud_rate"]
    L += [
        "",
        f"Ceiling: with {_pct(fr, 2)} fraud in the test month, a {_pct(b, 0)} queue can hold at most "
        f"{_pct(min(1.0, b / fr))} of all fraud (if every reviewed transaction were fraud), so recall @ "
        f"{_pct(b, 0)} must be read together with precision @ {_pct(b, 0)}.",
        "",
        "Confidence intervals: bootstrap that resamples whole **days** of the test month "
        "(fraud arrives in bursts, so rows are not independent). Paired differences "
        "(same resamples for both models):",
        "",
    ]
    for k in ("lgbm_velocity - logreg_velocity", "lgbm_velocity - lgbm_base"):
        L.append(
            f"- {k}: PR-AUC {_ci(unc['pr_auc'][k])}, recall @ {_pct(b, 0)} "
            f"{_ci(unc['recall_at_budget'][k], pct=True)}"
        )
    L += [
        "",
        "### Recall at other review budgets",
        "",
        "| Model | "
        + " | ".join(
            k.split("@")[1].replace("pct", "%")
            for k in res["lgbm_velocity"]["test"]
            if k.startswith("recall@")
        )
        + " |",
        "|---|" + "---|" * sum(k.startswith("recall@") for k in res["lgbm_velocity"]["test"]),
    ]
    for name, r in res.items():
        vals = [_pct(v) for k, v in r["test"].items() if k.startswith("recall@")]
        L.append(f"| {NAMES.get(name, name)} | " + " | ".join(vals) + " |")

    t = res["lgbm_velocity"]["test"]
    L += [
        "",
        "### Operating the threshold",
        "",
        f"A real deployment cannot rank the test month in hindsight: the cut-off is fixed on the "
        f"validation month (score >= {t['threshold']:.4f} flags {_pct(b, 0)} there) and then applied to new "
        f"traffic. On the test month that frozen threshold flags **{_pct(t['alert_rate_at_threshold'], 2)}** "
        f"of transactions, catching {_pct(t['recall_at_threshold'])} of fraud at "
        f"{_pct(t['precision_at_threshold'])} precision. A flag rate drifting away from the budget is the "
        "first, label-free sign that the queue will be over- or under-staffed.",
        "",
        "![PR curves](figures/pr_curves.png)",
        "![Recall vs budget](figures/recall_vs_budget.png)",
        "",
        "## What drives the score (SHAP)",
        "",
        "| # | Feature | Meaning | mean abs SHAP |",
        "|---|---|---|---|",
    ]
    for i, row in enumerate(s["shap_top"][:15], 1):
        f = row["feature"]
        mark = " (velocity)" if f in velocity else ""
        L.append(f"| {i} | `{f}`{mark} | {describe(f)} | {row['mean_abs_shap']:.3f} |")
    n_vel = sum(r["feature"] in velocity for r in s["shap_top"][:20])
    L += [
        "",
        f"{n_vel} of the top 20 features are velocity features.",
        "",
        "![SHAP importance](figures/shap_importance.png)",
    ]
    if "shap_beeswarm.png" in s["figures"]:
        L += [
            "![SHAP beeswarm](figures/shap_beeswarm.png)",
            "",
            "Explanation of the highest-scored fraud in the sample:",
            "",
            "![SHAP waterfall](figures/shap_waterfall.png)",
        ]
    L += [
        "",
        "## Drift: test month vs. validation month",
        "",
        "| Feature | PSI | Status | Missing (ref -> test) |",
        "|---|---|---|---|",
    ]
    for row in s["drift_top"][:10]:
        L.append(
            f"| `{row['feature']}` | {row['psi']:.3f} | {row['status']} | "
            f"{_pct(row['missing_ref'])} -> {_pct(row['missing_cur'])} |"
        )
    L += [
        "",
        "Week by week (PSI of the model score, and the share flagged at the frozen threshold):",
        "",
        "| Week | Rows | Score PSI | Alert rate | Features with PSI > 0.25 |",
        "|---|---|---|---|---|",
    ]
    for w in s["drift_weekly"]:
        L.append(
            f"| {w['week']} | {w['rows']:,} | {w['score_psi']:.3f} | {_pct(w['alert_rate'], 2)} "
            f"| {w['n_features_significant']} |"
        )
    L += [
        "",
        "![PSI](figures/psi_drift.png)",
        "![Daily alert rate](figures/daily_alert_rate.png)",
        "",
        f"_Generated by `python -m fraud_detection.train` in {s['runtime_seconds']:.0f}s._",
        "",
    ]
    path.write_text("\n".join(L))
