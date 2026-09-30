"""Cross-validation: fold definitions, metrics (mean and std over folds), out-of-fold predictions."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict, cross_validate

SEED = 42
METRICS = ["accuracy", "f1", "roc_auc"]


def make_splits(cv_name, y, groups):
    """'stratified' = normal 5-fold; 'site' = 5-fold where no site is in both train and test."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED) if cv_name == "stratified" else GroupKFold(5)
    return list(cv.split(np.zeros(len(y)), y, groups))


def summarize(scores):
    """{metric: per-fold list} -> {metric_mean, metric_std}."""
    return {f"{m}_{f}": float(getattr(np, f)(scores[m])) for m in METRICS for f in ("mean", "std")}


def evaluate(model, X, y, splits, n_jobs=1):
    """Outer CV of a (nested) sklearn model."""
    s = cross_validate(model, X, y, cv=splits, scoring=METRICS, n_jobs=n_jobs)
    return summarize({m: s[f"test_{m}"] for m in METRICS})


def fold_scores(y_true, prob, splits):
    """Per-fold metrics from out-of-fold probabilities (used for the deep model)."""
    out = {m: [] for m in METRICS}
    for _, te in splits:
        pred = (prob[te] > 0.5).astype(int)
        out["accuracy"].append(accuracy_score(y_true[te], pred))
        out["f1"].append(f1_score(y_true[te], pred, zero_division=0))
        out["roc_auc"].append(roc_auc_score(y_true[te], prob[te]))
    return out


def oof_predict(model, X, y, splits, n_jobs=1):
    """Out-of-fold class predictions, used for the confusion matrix."""
    return cross_val_predict(model, X, y, cv=splits, n_jobs=n_jobs)
