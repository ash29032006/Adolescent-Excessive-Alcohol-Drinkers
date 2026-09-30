"""Live demo (offline, < 2 min). Needs data/ (download_data.py) and models/ (run_experiments.py).

    python demo.py              # a random held-out person
    python demo.py --person 3   # the 4th held-out person
    python demo.py --no-plot    # terminal only
"""
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure

from src.data import load_abide
from src.features import feature_matrix
from src.plots import plot_connectivity, plot_timeseries

LABELS = {0: "control", 1: "autism"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", type=int, default=None, help="index into the held-out people")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    for f in ("models/demo_model.joblib", "results/results.csv"):
        if not Path(f).exists():
            raise SystemExit(f"Missing {f}. Run `python run_experiments.py` first.")
    ts, y, pheno = load_abide()  # exits with a clear message if data/ is missing
    saved = joblib.load("models/demo_model.joblib")

    print("=" * 60, "\nDATASET  ABIDE preprocessed (CPAC, Craddock 200 regions)")
    print(f"people {len(y)} | autism {y.sum()} | control {(y == 0).sum()} | sites {pheno.SITE_ID.nunique()}")
    print(f"timepoints per scan {min(map(len, ts))}-{max(map(len, ts))} | age {pheno.AGE_AT_SCAN.min():.1f}-{pheno.AGE_AT_SCAN.max():.1f}")

    # The model never saw these people during training.
    te = saved["test_idx"]
    X = feature_matrix(saved["features"], ts, pheno, te)
    if X.shape[1] != saved["model"][-1].n_features_in_:
        raise SystemExit("Saved model does not match data/. Re-run `python run_experiments.py`.")
    prob = saved["model"].predict_proba(X)[:, 1] if hasattr(saved["model"], "predict_proba") else None
    pred = saved["model"].predict(X)
    print(f"\nMODEL  {saved['features']} + {saved['name']}  (trained on 80%, tested on {len(te)} held-out people)")
    print(f"held-out accuracy {np.mean(pred == y[te]):.3f}   (always guessing the majority class: {max(np.mean(y[te]), 1 - np.mean(y[te])):.3f})")

    k = np.random.default_rng().integers(len(te)) if args.person is None else args.person % len(te)
    i = te[k]
    row = pheno.iloc[i]
    conf = f" (P(autism) = {prob[k]:.2f})" if prob is not None else ""
    print(f"\nPERSON  #{i}  site {row.SITE_ID} | age {row.AGE_AT_SCAN:.1f} | sex {'F' if row.SEX == 2 else 'M'} | scan {ts[i].shape}")
    print(f"predicted: {LABELS[int(pred[k])]}{conf}   true: {LABELS[int(y[i])]}   ->  {'CORRECT' if pred[k] == y[i] else 'WRONG'}")

    df = pd.read_csv("results/results.csv")
    cols = ["features", "model", "cv", "accuracy_mean", "f1_mean", "roc_auc_mean"]
    print("\nRESULTS (5-fold CV, top 10 by ROC AUC per CV scheme)")
    for cv in ("stratified", "site"):
        print(f"\n[{cv}]\n" + df[df.cv == cv].sort_values("roc_auc_mean", ascending=False)[cols].head(10).to_string(index=False))

    if not args.no_plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        plot_timeseries(ts[i], axes[0])
        plot_connectivity(ConnectivityMeasure(kind="correlation").fit_transform([ts[i]])[0], axes[1])
        fig.suptitle(f"Person #{i}: predicted {LABELS[int(pred[k])]}, true {LABELS[int(y[i])]}", fontsize=13)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
