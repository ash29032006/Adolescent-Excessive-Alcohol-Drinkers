"""Run every experiment and save results/*.csv, figures/*.png and the demo model.

    python run_experiments.py                          # everything (slow part: tangent)
    python run_experiments.py --experiment connectivity
    python run_experiments.py --subset 60              # quick smoke test on 60 people

Experiments (each run under stratified AND site-grouped CV):
  baseline     paper reproduction: range features, demographics, both  (+ dummy)
  connectivity correlation connectivity, with and without demographics
  tangent      tangent connectivity (fit inside CV), with and without demographics
  deep         small 1D CNN on raw time series
Rows of an experiment replace older rows in results/results.csv, so experiments can be run separately.
"""
import argparse
import random
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures only, no windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.data import load_abide
from src.deep import cnn_oof_probs
from src.evaluate import evaluate, fold_scores, make_splits, oof_predict, summarize
from src.features import feature_matrix, make_features
from src.models import MODELS, SEED, make_model
from src.plots import best_per_features, plot_ablation, plot_comparison, plot_confusion, plot_connectivity, plot_timeseries

warnings.filterwarnings("ignore")
RESULTS, FIGURES, MODELS_DIR = Path("results"), Path("figures"), Path("models")
EXPERIMENTS = {"baseline": ["demo", "range", "range+demo"],
               "connectivity": ["correlation", "correlation+demo"],
               "tangent": ["tangent", "tangent+demo"],
               "deep": ["timeseries"]}
PER_PERSON = ("range", "correlation", "demo")  # features the demo can compute for one new person


def run_sklearn(feature_sets, ts, y, splits, tables):
    """Evaluate every model on every feature set under every CV scheme."""
    rows = []
    for cv_name, sp in splits.items():
        for fs in feature_sets:
            for m in MODELS:
                if m == "dummy" and fs != "demo":  # the dummy ignores features: run it once
                    continue
                t0 = time.time()
                n_jobs = 1 if "tangent" in fs else -1  # tangent cache lives in this process
                res = evaluate(make_model(m, make_features(fs, ts, tables)), np.arange(len(y))[:, None], y, sp, n_jobs)
                rows.append({"features": fs, "model": m, "cv": cv_name, **res})
                print(f"{cv_name:10s} {fs:18s} {m:10s} acc {res['accuracy_mean']:.3f}  "
                      f"f1 {res['f1_mean']:.3f}  auc {res['roc_auc_mean']:.3f}   ({time.time() - t0:.0f}s)", flush=True)
    return rows


def run_deep(ts, y, splits):
    """CNN under both CV schemes."""
    rows = []
    for cv_name, sp in splits.items():
        res = summarize(fold_scores(y, cnn_oof_probs(ts, y, sp), sp))
        rows.append({"features": "timeseries", "model": "cnn", "cv": cv_name, **res})
        print(f"{cv_name:10s} timeseries         cnn        acc {res['accuracy_mean']:.3f}  auc {res['roc_auc_mean']:.3f}", flush=True)
    return rows


def finalize(df, ts, y, pheno, tables, splits):
    """Figures, confusion matrix of the best model, and the model used by demo.py."""
    plot_comparison(df, FIGURES / "model_comparison.png")
    plot_ablation(df, FIGURES / "age_ablation.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_timeseries(ts[0], axes[0])
    plot_connectivity(ConnectivityMeasure(kind="correlation").fit_transform([ts[0]])[0], axes[1])
    fig.tight_layout()
    fig.savefig(FIGURES / "example_subject.png", dpi=200)
    plt.close(fig)

    sk = df[df.model != "cnn"]
    best = best_per_features(sk).iloc[-1]  # best sklearn model overall, stratified CV
    model = make_model(best.model, make_features(best.features, ts, tables))
    pred = oof_predict(model, np.arange(len(y))[:, None], y, splits["stratified"], 1 if "tangent" in best.features else -1)
    cm = confusion_matrix(y, pred)
    pd.DataFrame(cm, index=["true_control", "true_autism"], columns=["pred_control", "pred_autism"]).to_csv(
        RESULTS / "confusion_matrix.csv")
    (RESULTS / "best_model.txt").write_text(f"{best.features},{best.model}\n")
    plot_confusion(cm, f"{best.features} + {best.model}, out-of-fold (stratified CV)", FIGURES / "confusion_matrix.png")

    # Demo model: best per-person feature set, trained on 80%, the held-out 20% is shown in the demo.
    demo = best_per_features(sk[sk.features.map(lambda f: set(f.split("+")) <= set(PER_PERSON))]).iloc[-1]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    clf = make_model(demo.model).fit(tables_for(demo.features, tables)[tr], y[tr])
    joblib.dump({"model": clf, "features": demo.features, "name": demo.model, "test_idx": pheno["row"].to_numpy()[te]}, MODELS_DIR / "demo_model.joblib")
    print(f"best: {best.features} + {best.model} | demo model: {demo.features} + {demo.model}")


def tables_for(name, tables):
    """Concatenate precomputed blocks, e.g. 'correlation+demo'."""
    return np.hstack([tables[b] for b in name.split("+")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="all", choices=["all", *EXPERIMENTS])
    ap.add_argument("--subset", type=int, default=0, help="use a random N people (smoke test)")
    args = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    for d in (RESULTS, FIGURES, MODELS_DIR):
        d.mkdir(exist_ok=True)

    ts, y, pheno = load_abide()
    pheno["row"] = np.arange(len(y))  # original row, so demo.py can find held-out people in the full data
    if args.subset:
        keep = np.sort(np.random.default_rng(SEED).choice(len(y), args.subset, replace=False))  # spans all sites
        ts, y, pheno = [ts[i] for i in keep], y[keep], pheno.iloc[keep].reset_index(drop=True)
    print(f"{len(y)} people | autism {y.sum()} | control {(y == 0).sum()} | sites {pheno.SITE_ID.nunique()}")
    splits = {cv: make_splits(cv, y, pheno.SITE_ID.to_numpy()) for cv in ("stratified", "site")}
    tables = {b: feature_matrix(b, ts, pheno) for b in PER_PERSON}

    todo = EXPERIMENTS if args.experiment == "all" else {args.experiment: EXPERIMENTS[args.experiment]}
    rows = []
    for name, fsets in todo.items():
        print(f"\n== {name}")
        rows += run_deep(ts, y, splits) if name == "deep" else run_sklearn(fsets, ts, y, splits, tables)

    new = pd.DataFrame(rows)
    path = RESULTS / "results.csv"
    if path.exists():  # keep rows from experiments that were not re-run
        old = pd.read_csv(path)
        new = pd.concat([old[~old.features.isin(new.features.unique())], new], ignore_index=True)
    new["n_subjects"] = len(y)
    new.round(4).to_csv(path, index=False)
    pd.DataFrame([{"n_subjects": len(y), "n_autism": int(y.sum()), "n_control": int((y == 0).sum()),
                   "n_sites": pheno.SITE_ID.nunique(), "min_timepoints": min(len(t) for t in ts),
                   "max_timepoints": max(len(t) for t in ts), "age_min": pheno.AGE_AT_SCAN.min(),
                   "age_max": pheno.AGE_AT_SCAN.max()}]).to_csv(RESULTS / "dataset_summary.csv", index=False)
    finalize(new, ts, y, pheno, tables, splits)
    print(f"\nSaved {path}, figures/ and models/demo_model.joblib")


if __name__ == "__main__":
    main()
