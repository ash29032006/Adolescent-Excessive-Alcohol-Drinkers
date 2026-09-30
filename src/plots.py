"""Figures for the write-up, slides and demo. All functions take data and either an axis or a path."""
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical order: blue, orange, aqua
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "font.size": 10})


def best_per_features(df, cv="stratified"):
    """For every feature set keep its best model (by ROC AUC) under the given CV scheme."""
    d = df[(df.cv == cv) & (df.model != "dummy")]
    return d.loc[d.groupby("features").roc_auc_mean.idxmax()].sort_values("roc_auc_mean")


def _bars(ax, labels, series, names, xlabel):
    """Horizontal grouped bars with std error bars and a chance line at 0.5."""
    h = 0.8 / len(series)
    for k, ((mean, std), name) in enumerate(zip(series, names)):
        ax.barh(np.arange(len(labels)) + (k - (len(series) - 1) / 2) * h, mean, h * 0.9, xerr=std,
                color=COLORS[k], label=name, error_kw={"elinewidth": 0.8, "ecolor": MUTED})
    ax.axvline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")


def plot_comparison(df, path):
    """Best model per feature set: normal CV vs site-grouped CV (ROC AUC)."""
    best = best_per_features(df)
    site = df[df.cv == "site"].set_index(["features", "model"])
    keys = list(zip(best.features, best.model))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(best) + 1.5))
    _bars(ax, [f"{f}  ({m})" for f, m in keys],
          [(best.roc_auc_mean, best.roc_auc_std),
           ([site.roc_auc_mean.get(k, np.nan) for k in keys], [site.roc_auc_std.get(k, np.nan) for k in keys])],
          ["Stratified 5-fold CV", "Site-grouped 5-fold CV"], "ROC AUC (mean ± std over folds, dashed = chance)")
    ax.set_title("Best model per feature set", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ablation(df, path):
    """Brain only vs demographics only vs both, best model each, stratified CV (ROC AUC)."""
    best = best_per_features(df).set_index("features")
    brains = [b for b in ("range", "correlation", "tangent") if b in best.index]
    get = lambda name, col: best[col].get(name, np.nan)
    series = [([get(b, "roc_auc_mean") for b in brains], [get(b, "roc_auc_std") for b in brains]),
              ([get("demo", "roc_auc_mean")] * len(brains), [get("demo", "roc_auc_std")] * len(brains)),
              ([get(f"{b}+demo", "roc_auc_mean") for b in brains], [get(f"{b}+demo", "roc_auc_std") for b in brains])]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _bars(ax, brains, series, ["Brain only", "Demographics only", "Brain + demographics"], "ROC AUC (stratified CV)")
    ax.set_title("Age / demographics ablation", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(cm, title, path):
    """2x2 confusion matrix with counts."""
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else INK, fontsize=13)
    ax.set_xticks([0, 1], ["Control", "Autism"])
    ax.set_yticks([0, 1], ["Control", "Autism"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title(title, loc="left", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_timeseries(t, ax, n=5):
    """First n region signals of one person, offset vertically so they do not overlap."""
    z = (t - t.mean(0)) / (t.std(0) + 1e-8)
    for k in range(n):
        ax.plot(z[:, k] + 4 * k, lw=1, color=COLORS[k % 3])
    ax.set_yticks([4 * k for k in range(n)], [f"region {k + 1}" for k in range(n)])
    ax.set_xlabel("Timepoint")
    ax.set_title("Resting-state BOLD signal (z-scored)", loc="left", color=INK)


def plot_connectivity(mat, ax):
    """200 x 200 region correlation matrix on a diverging scale centred at 0."""
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_title("Functional connectivity (correlation)", loc="left", color=INK)
    ax.set_xlabel("Region")
    ax.set_ylabel("Region")
    plt.colorbar(im, ax=ax, fraction=0.046)
