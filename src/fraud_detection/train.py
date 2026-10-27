"""End-to-end training and evaluation.

    python -m fraud_detection.train --config configs/ieee.yaml

1. load data, sort by time, derive D1n / uid
2. velocity features over the full stream (past transactions only, no labels)
3. out-of-time split: train | valid (early stopping, threshold) | test (reported)
4. four models: {LogReg, LightGBM} x {without, with velocity features}
5. PR-AUC, recall/precision/amount-recall at review budgets, day-block bootstrap CIs
6. SHAP on the test month, PSI drift test vs. validation
7. save the serving bundle (model, encoders, velocity state, drift reference)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, plots
from .artifacts import Bundle, save_bundle
from .config import Config
from .data import AMOUNT_COL, DAY_COL, TARGET, TIME_COL, add_time_columns, load_dataset
from .drift import DriftMonitor
from .explain import global_importance, shap_values
from .features import FeatureBuilder
from .metrics import block_bootstrap, ci, evaluate, recall_at_budget, threshold_for_budget
from .models import fit_lightgbm, fit_logreg, predict
from .report import write_report
from .split import time_split
from .velocity import OnlineVelocityStore, compute_velocity

log = logging.getLogger("fraud_detection")
MAIN = "lgbm_velocity"


def build_features(cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    df = add_time_columns(load_dataset(cfg.data))
    log.info(
        "data: %d rows, %d columns, fraud rate %.2f%%",
        len(df),
        df.shape[1],
        100 * df[TARGET].mean(),
    )
    t = time.perf_counter()
    vel = compute_velocity(df, cfg.features.entities, cfg.features.windows)
    log.info("velocity features: %d columns in %.1fs", vel.shape[1], time.perf_counter() - t)
    return pd.concat([df, vel], axis=1), list(vel.columns)


def run(cfg: Config) -> dict:
    t_start = time.perf_counter()
    report_dir = Path(cfg.output.report_dir)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df, velocity_cols = build_features(cfg)
    split = time_split(df, cfg.split)
    log.info(
        "split (rows): %s",
        {k: v for k, v in split.summary().items() if k in ("train", "valid", "test")},
    )

    fc = cfg.features
    builder = FeatureBuilder(velocity_cols, fc.freq_encode, fc.min_category_count, fc.drop)
    builder.fit(df[split.train])
    X = builder.transform(df)
    y = df[TARGET].to_numpy()
    amount = df[AMOUNT_COL].to_numpy(dtype=float)
    parts = {k: getattr(split, k) for k in ("train", "valid", "test")}
    log.info(
        "model matrix: %d features (%d velocity, %d categorical)",
        len(builder.feature_names),
        len(velocity_cols),
        len(builder.cat_cols),
    )

    ev = cfg.evaluation
    variants = {
        "logreg_base": ("logreg", builder.base_features),
        "logreg_velocity": ("logreg", builder.feature_names),
        "lgbm_base": ("lgbm", builder.base_features),
        MAIN: ("lgbm", builder.feature_names),
    }
    models, scores, results = {}, {}, {}
    for name, (kind, feats) in variants.items():
        t = time.perf_counter()
        Xtr, ytr = X.loc[parts["train"], feats], y[parts["train"]]
        if kind == "logreg":
            model = fit_logreg(Xtr, ytr, builder.cat_cols, **cfg.models.logreg)
        else:
            model = fit_lightgbm(
                Xtr,
                ytr,
                X.loc[parts["valid"], feats],
                y[parts["valid"]],
                builder.cat_cols,
                **cfg.models.lightgbm,
            )
        s_valid = predict(model, X.loc[parts["valid"], feats])
        s_test = predict(model, X.loc[parts["test"], feats])
        thr = threshold_for_budget(s_valid, ev.review_budget)
        results[name] = {
            "valid": evaluate(y[parts["valid"]], s_valid, amount[parts["valid"]], ev.budgets),
            "test": evaluate(
                y[parts["test"]], s_test, amount[parts["test"]], ev.budgets, threshold=thr
            ),
            "n_features": len(feats),
            "train_seconds": round(time.perf_counter() - t, 1),
        }
        if kind == "lgbm":
            results[name]["best_iteration"] = int(model.best_iteration)
        models[name], scores[name] = model, {"valid": s_valid, "test": s_test}
        log.info(
            "%-16s test PR-AUC %.4f  recall@%g%% %.3f  (%.0fs)",
            name,
            results[name]["test"]["pr_auc"],
            ev.review_budget * 100,
            recall_at_budget(y[parts["test"]], s_test, ev.review_budget),
            time.perf_counter() - t,
        )

    # Uncertainty: resample whole days of the test month (fraud is bursty).
    y_te, day_te = y[parts["test"]], df.loc[parts["test"], DAY_COL].to_numpy()
    test_scores = {k: v["test"] for k, v in scores.items()}
    boot = {
        "pr_auc": block_bootstrap(y_te, test_scores, day_te, _ap, ev.bootstrap, ev.seed),
        "recall_at_budget": block_bootstrap(
            y_te,
            test_scores,
            day_te,
            lambda yy, ss: recall_at_budget(yy, ss, ev.review_budget),
            ev.bootstrap,
            ev.seed,
        ),
    }
    uncertainty = {}
    for metric, samples in boot.items():
        uncertainty[metric] = {name: ci(s) for name, s in samples.items()}
        for other in ("logreg_velocity", "lgbm_base"):
            uncertainty[metric][f"{MAIN} - {other}"] = ci(samples[MAIN] - samples[other])

    # Explanations on a random sample of the test month.
    booster = models[MAIN]
    X_te = X.loc[parts["test"], builder.feature_names]
    rng = np.random.default_rng(ev.seed)
    sample = np.sort(rng.choice(len(X_te), min(cfg.explain.sample_size, len(X_te)), replace=False))
    importance = global_importance(booster, X_te.iloc[sample])
    importance.to_csv(report_dir / "shap_importance.csv", index=False)
    # Waterfall for the highest-scored true fraud in the sample.
    s_sample, y_sample = scores[MAIN]["test"][sample], y_te[sample]
    row = int(np.argmax(np.where(y_sample == 1, s_sample, -1)))
    vals, base = shap_values(booster, X_te.iloc[sample])
    figures = plots.shap_plots(vals, base, X_te.iloc[sample], row, fig_dir)

    # Drift: reference = validation month (where the threshold was calibrated).
    dc = cfg.drift
    ref = X.loc[parts["valid"], builder.feature_names].assign(score=scores[MAIN]["valid"])
    monitor = DriftMonitor.fit(
        ref, builder.cat_cols, dc.n_bins, dc.max_categories, dc.warn, dc.alert
    )
    drift_rep = monitor.report(X_te.assign(score=scores[MAIN]["test"]))
    drift_rep.to_csv(report_dir / "drift_test_vs_valid.csv", index=False)
    weekly = []
    for w, idx in pd.Series(np.arange(len(day_te))).groupby((day_te - day_te.min()) // 7):
        cur = X_te.iloc[idx.to_numpy()].assign(score=scores[MAIN]["test"][idx.to_numpy()])
        summ = monitor.summary(cur)
        weekly.append(
            {
                "week": int(w) + 1,
                "rows": len(idx),
                "score_psi": summ["score_psi"],
                "alert_rate": float((cur["score"] >= results[MAIN]["test"]["threshold"]).mean()),
                "n_features_significant": summ["n_features_significant"],
            }
        )

    # Figures.
    plots.pr_curves(y_te, test_scores, fig_dir / "pr_curves.png")
    plots.recall_vs_budget(y_te, test_scores, ev.review_budget, fig_dir / "recall_vs_budget.png")
    plots.importance_bars(importance, set(velocity_cols), fig_dir / "shap_importance.png")
    plots.psi_bars(drift_rep, dc.warn, dc.alert, fig_dir / "psi_drift.png")
    vt = parts["valid"] | parts["test"]
    plots.daily_alert_rate(
        df.loc[vt, DAY_COL].to_numpy(),
        np.r_[scores[MAIN]["valid"], scores[MAIN]["test"]] >= results[MAIN]["test"]["threshold"],
        ev.review_budget,
        split.test_days[0],
        fig_dir / "daily_alert_rate.png",
    )
    figures += [
        "pr_curves.png",
        "recall_vs_budget.png",
        "shap_importance.png",
        "psi_drift.png",
        "daily_alert_rate.png",
    ]

    # Serving bundle: velocity state as of the start of the test month, so the test
    # month can be replayed through the API as if it were live traffic.
    test_start_dt = int(df.loc[parts["test"], TIME_COL].min())
    store = OnlineVelocityStore.from_frame(
        df[df[TIME_COL] < test_start_dt], fc.entities, fc.windows
    )
    metadata = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "package_version": __version__,
        "data_source": cfg.data.source,
        "features": builder.feature_names,
        "categorical_features": builder.cat_cols,
        "velocity_features": velocity_cols,
        "entities": fc.entities,
        "windows": fc.windows,
        "review_budget": ev.review_budget,
        "threshold": results[MAIN]["test"]["threshold"],
        "split_days": {k: list(getattr(split, f"{k}_days")) for k in ("train", "valid", "test")},
        "state_as_of_transaction_dt": test_start_dt,
        "test_metrics": results[MAIN]["test"],
        "top_k_reasons": cfg.explain.top_k_reasons,
    }
    save_bundle(cfg.output.artifact_dir, Bundle(booster, builder, store, monitor, metadata))

    summary = {
        "data_source": cfg.data.source,
        "rows": int(len(df)),
        "fraud_rate": float(y.mean()),
        "split": split.summary(),
        "review_budget": ev.review_budget,
        "results": results,
        "bootstrap_95ci": uncertainty,
        "shap_top": importance.head(20).to_dict(orient="records"),
        "drift_top": drift_rep.head(15).to_dict(orient="records"),
        "drift_weekly": weekly,
        "figures": figures,
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    (report_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    write_report(summary, report_dir / "report.md", set(velocity_cols))
    log.info(
        "done in %.0fs - report: %s, artifacts: %s",
        summary["runtime_seconds"],
        report_dir / "report.md",
        cfg.output.artifact_dir,
    )
    return summary


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if y.any() else float("nan")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main() -> None:
    p = argparse.ArgumentParser(description="Train and evaluate the fraud models.")
    p.add_argument("--config", default="configs/synthetic.yaml")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(Config.load(a.config))


if __name__ == "__main__":
    main()
