"""Build the deliverables from results/ and figures/ (run after run_experiments.py):

    report/writeup.pdf   two-page write-up
    report/slides.pptx   slide deck (via pandoc; report/slides.md is always written)
    README.md            results table between the RESULTS markers

Every number is read from results/, nothing is typed by hand.
"""
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

TEAM = "Team 35: Ashwin C H (PES1UG24AM900) and Chidivilas Adi (PES1UG24AM802)"
TITLE = "Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data"
SUB = "UE24CS352A Machine Learning mini project, PES University. Pipeline demonstrated on ABIDE resting-state fMRI"
R, OUT = Path("results"), Path("report")
NAMES = {"logreg": "Logistic regression", "linear_svm": "Linear SVM", "rbf_svm": "RBF SVM", "dummy": "Dummy (majority)", "cnn": "1D CNN"}


def load():
    """Collect every number the documents use into one dict."""
    df = pd.read_csv(R / "results.csv")
    ds = pd.read_csv(R / "dataset_summary.csv").iloc[0]
    cm = pd.read_csv(R / "confusion_matrix.csv", index_col=0).to_numpy()
    get = lambda f, m, cv="stratified": df[(df.features == f) & (df.model == m) & (df.cv == cv)].iloc[0]
    strat = df[(df.cv == "stratified") & ~df.model.isin(["dummy"])]
    best_of = lambda f: strat[strat.features == f].sort_values("roc_auc_mean").iloc[-1]
    bf, bm = (R / "best_model.txt").read_text().strip().split(",")
    brains = [f for f in ("range", "correlation", "tangent") if f in set(strat.features)]
    top_brain = max(brains, key=lambda f: best_of(f).roc_auc_mean)
    return dict(df=df, ds=ds, cm=cm, get=get, best_of=best_of, bf=bf, bm=bm, top_brain=top_brain,
                has_cnn="cnn" in set(df.model))


def table(k):
    """Best model per feature set, stratified and site-grouped CV side by side."""
    df, rows = k["df"], []
    order = ["demo", "range", "range+demo", "correlation", "correlation+demo", "tangent", "tangent+demo", "timeseries"]
    for f in [f for f in order if f in set(df.features)] + ["dummy"]:
        s = df[(df.cv == "stratified") & (df.model == "dummy")] if f == "dummy" else \
            df[(df.cv == "stratified") & (df.features == f) & (df.model != "dummy")]
        b = s.sort_values("roc_auc_mean").iloc[-1]
        g = df[(df.cv == "site") & (df.features == b.features) & (df.model == b.model)].iloc[0]
        rows.append({"Features": "none (dummy)" if f == "dummy" else f, "Model": NAMES[b.model],
                     "Acc": f"{b.accuracy_mean:.3f} ± {b.accuracy_std:.3f}", "F1": f"{b.f1_mean:.3f}",
                     "AUC": f"{b.roc_auc_mean:.3f} ± {b.roc_auc_std:.3f}",
                     "Site acc": f"{g.accuracy_mean:.3f}", "Site AUC": f"{g.roc_auc_mean:.3f}"})
    return pd.DataFrame(rows)


def md_table(t):
    """DataFrame -> markdown pipe table (avoids the optional tabulate dependency)."""
    return "\n".join(["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in t.values])


def findings(k):
    """Result sentences, each computed from the saved numbers."""
    ds, get, best_of = k["ds"], k["get"], k["best_of"]
    b, bs = get(k["bf"], k["bm"]), get(k["bf"], k["bm"], "site")
    dummy = get("demo", "dummy")
    pr, pb = get("range+demo", "logreg"), get("range", "logreg")
    tb, d = best_of(k["top_brain"]), best_of("demo")
    tbd = best_of(k["top_brain"] + "+demo")
    corr, rng = best_of("correlation") if "correlation" in set(k["df"].features) else None, best_of("range")
    tn, fp, fn, tp = k["cm"].ravel()
    drop = b.roc_auc_mean - bs.roc_auc_mean
    out = [
        f"Best model: {k['bf']} features with {NAMES[k['bm']]}, accuracy {b.accuracy_mean:.3f} ± {b.accuracy_std:.3f} and "
        f"ROC AUC {b.roc_auc_mean:.3f} ± {b.roc_auc_std:.3f} (stratified 5-fold CV), against {dummy.accuracy_mean:.3f} accuracy "
        f"and AUC 0.5 for always guessing the majority class. Out of fold it caught {tp}/{tp + fn} autism cases "
        f"(sensitivity {tp / (tp + fn):.2f}) and {tn}/{tn + fp} controls (specificity {tn / (tn + fp):.2f}).",
        f"Paper reproduction: range features + demographics with logistic regression gave accuracy {pr.accuracy_mean:.3f} "
        f"(AUC {pr.roc_auc_mean:.3f}); range features alone gave {pb.accuracy_mean:.3f} (AUC {pb.roc_auc_mean:.3f}). "
        f"The paper reported 0.713 with age and 0.538 without on NCANDA (different task, so not directly comparable).",
    ]
    if corr is not None:
        out.append(f"Better features: correlation connectivity {'beats' if corr.roc_auc_mean > rng.roc_auc_mean else 'does not beat'} "
                   f"the paper's range features (best AUC {corr.roc_auc_mean:.3f} vs {rng.roc_auc_mean:.3f}).")
    brain_led = tb.roc_auc_mean > d.roc_auc_mean and abs(tbd.roc_auc_mean - tb.roc_auc_mean) < 0.03
    out.append(f"Age ablation: demographics alone reach AUC {d.roc_auc_mean:.3f}; {k['top_brain']} brain features alone reach "
               f"{tb.roc_auc_mean:.3f} and {tbd.roc_auc_mean:.3f} with demographics added. " +
               ("Unlike the paper, the prediction is carried by the brain features, not by age."
                if brain_led else "Demographics contribute a large part of the performance, as in the paper."))
    out.append(f"Site effect: holding out whole sites moves the best model's AUC from {b.roc_auc_mean:.3f} to "
               f"{bs.roc_auc_mean:.3f} ({-drop:+.3f}). " +
               ("The model transfers to unseen scanners." if drop < 0.03 else
                "Part of the normal-CV score comes from site-specific signal, so site-grouped CV is the honest estimate."))
    if k["has_cnn"]:
        c = get("timeseries", "cnn")
        out.append(f"Deep learning: the 1D CNN on raw time series reached accuracy {c.accuracy_mean:.3f}, AUC {c.roc_auc_mean:.3f}"
                   f"{', below the linear models on connectivity: as in the paper, simple models on derived features win on small fMRI data' if c.roc_auc_mean < b.roc_auc_mean else ', matching or beating the linear models'}.")
    return out


def sections(k):
    """Static prose plus computed findings, shared by the write-up and the slides."""
    ds = k["ds"]
    return {
        "Problem statement": [
            "Excessive alcohol use in adolescence changes brain structure, but its effect on functional connectivity is unclear. "
            "The task, following Kim, Liu and Noh (Stanford CS229), is binary classification of a person from their resting-state "
            "fMRI: brain activity (BOLD signal) recorded at rest, summarised as one time series per brain region. We reproduce the "
            "paper's baseline, test stronger connectivity features, and check whether predictions rely on age or on scanner site."],
        "Dataset": [
            "The paper used NCANDA (715 adolescents). Access needs an NIAAA Data Use Certification and a committee review of up to "
            "30 days, which does not fit a one-week project. We therefore use ABIDE Preprocessed (CPAC pipeline, band-pass filtered, "
            "quality checked), the closest public match: resting-state fMRI, the same Craddock atlas family (200 regions), one time series "
            f"per region, a multi-site imbalanced sample. It has {int(ds.n_subjects)} people ({int(ds.n_autism)} autism, "
            f"{int(ds.n_control)} control) from {int(ds.n_sites)} sites, {int(ds.min_timepoints)} to {int(ds.max_timepoints)} "
            f"timepoints per scan, aged {ds.age_min:.1f} to {ds.age_max:.1f}. The label is autism vs control instead of heavy "
            "drinking. The pipeline is dataset agnostic: using NCANDA only needs a new loader returning the same arrays."],
        "Approach": [
            "Features: (1) the paper's derived feature, max minus min of each region's signal (200 values); (2) functional "
            "connectivity, the correlation between every pair of regions (19,900 values); (3) tangent-space connectivity, which "
            "measures each person relative to a group mean learned on the training fold; (4) demographics: age, sex, one-hot site "
            "(the equivalent of the paper's scanner type). Each brain feature set is run alone, with demographics, and demographics "
            "alone is run separately (age ablation).",
            "Models: L2 logistic regression, linear SVM and RBF SVM with balanced class weights (all people are kept, no downsampling), "
            "a majority-class dummy baseline, and a small 1D CNN on z-scored raw time series (random 64-step crops, global average pooling).",
            "Evaluation: 5-fold stratified CV and 5-fold site-grouped CV (no site appears in both train and test). Anything learned from "
            "data (tangent reference, scaler) is fit on training folds only, and C is tuned by an inner 3-fold grid search. We report "
            "the mean and standard deviation of accuracy, F1 and ROC AUC. Seed 42 everywhere."],
        "Implementation": [
            "Python with nilearn, scikit-learn and PyTorch. src/data.py loads data, src/features.py builds features, src/models.py builds "
            "the nested pipelines, src/evaluate.py runs CV, src/deep.py holds the CNN and src/plots.py the figures. run_experiments.py "
            "writes every number to results/ and figures/; demo.py runs offline and predicts people the demo model never saw; "
            "make_report.py builds this document and the slides from those files."],
        "Results": findings(k),
    }


def conclusions(k):
    """Short takeaways; the conditional parts come from the saved numbers."""
    b, bs = k["get"](k["bf"], k["bm"]), k["get"](k["bf"], k["bm"], "site")
    d = k["get"]("demo", "dummy")
    return [f"Resting-state connectivity separates the classes above chance ({b.accuracy_mean:.3f} vs {d.accuracy_mean:.3f} accuracy), "
            f"and site-grouped CV ({bs.accuracy_mean:.3f}) is the honest number to quote.",
            "Whole-brain correlation features are a stronger and simpler starting point than per-region signal range.",
            "Always report a dummy baseline, a demographics-only model and site-held-out CV: without them a model can look good "
            "for the wrong reasons (age or scanner).",
            "Next step: run the same pipeline on NCANDA once the data use agreement is approved (only src/data.py changes)."]


def writeup(k, path):
    """Two A4 pages rendered with matplotlib (no LaTeX needed)."""
    items = [("title", TITLE), ("sub", TEAM), ("sub", SUB)]
    for h, paras in sections(k).items():
        items += [("h", h)] + [("b" if h == "Results" else "p", p) for p in paras]
    items += [("table", table(k)), ("img", "figures/model_comparison.png"), ("h", "Conclusions")]
    items += [("b", c) for c in conclusions(k)]
    items += [("p", "Reference: Y. Kim, C. Liu, J. Noh, Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data, CS229, Stanford.")]
    style = {"title": (14, "bold", 95), "sub": (8.5, "normal", 120), "h": (10.5, "bold", 100), "p": (8.3, "normal", 118), "b": (8.3, "normal", 114)}
    with PdfPages(path) as pdf:
        fig, y = plt.figure(figsize=(8.27, 11.69)), 0.955
        for kind, val in items:
            need = 0.25 if kind == "img" else 0.03 * (len(val) + 1) + 0.02 if kind == "table" else 0.03
            if y - need < 0.04:
                pdf.savefig(fig), plt.close(fig)
                fig, y = plt.figure(figsize=(8.27, 11.69)), 0.955
            if kind == "img":
                im = plt.imread(val)
                h = 0.72 * im.shape[0] / im.shape[1] * 8.27 / 11.69
                fig.add_axes([0.14, y - h, 0.72, h]).imshow(im), fig.axes[-1].axis("off")
                y -= h + 0.01
            elif kind == "table":
                h = 0.017 * (len(val) + 1)
                ax = fig.add_axes([0.07, y - h, 0.86, h])
                ax.axis("off")
                t = ax.table(cellText=val.values, colLabels=val.columns, loc="center", cellLoc="center")
                t.auto_set_font_size(False), t.set_fontsize(7.2), t.scale(1, 1.0)
                for (r, _), c in t.get_celld().items():
                    c.set_edgecolor("#d0cfca"), r == 0 and c.set_text_props(weight="bold")
                y -= h + 0.012
            else:
                size, weight, width = style[kind]
                text = ("• " if kind == "b" else "") + val
                lines = textwrap.wrap(text, width, subsequent_indent="  " if kind == "b" else "")
                y -= 0.006 if kind == "h" else 0
                for line in lines:
                    fig.text(0.07, y, line, fontsize=size, weight=weight, va="top", color="#0b0b0b")
                    y -= size / 72 / 11.69 * 1.38
                y -= 0.005
        pdf.savefig(fig), plt.close(fig)


def slides(k, path):
    """Markdown deck converted to .pptx by pandoc (## = one slide)."""
    f = findings(k)
    pick = lambda prefix: next(x for x in f if x.startswith(prefix))
    md = f"""---
title: "{TITLE}"
subtitle: "{SUB}"
author: "{TEAM}"
---

## Problem statement

- Can a classifier tell groups apart from **resting-state fMRI** alone?
- Paper (Kim, Liu, Noh, CS229): heavy adolescent drinkers vs others, NCANDA, 715 people
- Their best model: logistic regression on max minus min per region + demographics: **0.713** accuracy, **0.538** without age
- Our goals: reproduce the baseline, find better features, check reliance on **age** and **scanner site**

## Dataset: why ABIDE

- NCANDA needs an NIAAA Data Use Certification + committee review (up to 30 days): not possible in one week
- **ABIDE Preprocessed** (CPAC, band-pass, quality checked): same data type, same Craddock atlas family (200 regions)
- {int(k['ds'].n_subjects)} people: {int(k['ds'].n_autism)} autism / {int(k['ds'].n_control)} control, {int(k['ds'].n_sites)} sites, {int(k['ds'].min_timepoints)} to {int(k['ds'].max_timepoints)} timepoints
- Label: autism vs control. Pipeline is dataset agnostic: NCANDA = one new loader

## One person's data

![](figures/example_subject.png)

## Approach

- **Range** (paper): max minus min per region, 200 features
- **Correlation connectivity**: every region pair, 19,900 features
- **Tangent connectivity**: relative to a group mean learned on training folds only
- **Demographics**: age, sex, site; run alone, with brain features, and brain only
- Models: logistic regression, linear SVM, RBF SVM, dummy baseline, small 1D CNN

## Evaluation without leakage

- 5-fold **stratified** CV and 5-fold **site-grouped** CV (unseen sites at test time)
- Learned steps (tangent mean, scaler) fit on training folds only
- Nested tuning: inner 3-fold grid search over C
- Balanced class weights instead of dropping people
- Accuracy, F1, ROC AUC as mean ± std; seed 42

## Results

{md_table(table(k))}

## Model comparison

![](figures/model_comparison.png)

## Age ablation

![](figures/age_ablation.png)

- {pick('Age ablation')}

## Site effect and confusion matrix

![](figures/confusion_matrix.png)

- {pick('Site effect')}

## Findings

""" + "\n".join(f"- {x}" for x in f) + """

## Conclusions

""" + "\n".join(f"- {x}" for x in conclusions(k)) + """

## Code and live demo

- `download_data.py` → `run_experiments.py` → `make_report.py`, `demo.py`
- `src/`: data, features, models, evaluate, deep, plots
- Demo: offline, predicts people the model never saw, shows their signal and connectivity

## Thank you

Questions?
"""
    path.with_suffix(".md").write_text(md)
    if shutil.which("pandoc"):
        subprocess.run(["pandoc", str(path.with_suffix(".md")), "-o", str(path), "--resource-path", "."], check=True)
    else:
        print("pandoc not found: wrote report/slides.md only")


def readme(k):
    """Replace the block between the RESULTS markers in README.md."""
    p = Path("README.md")
    block = "<!-- RESULTS START -->\n" + md_table(table(k)) + "\n\n" + \
            "\n".join(f"- {x}" for x in findings(k)) + "\n<!-- RESULTS END -->"
    p.write_text(re.sub(r"<!-- RESULTS START -->.*<!-- RESULTS END -->", lambda _: block, p.read_text(), flags=re.S))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    k = load()
    writeup(k, OUT / "writeup.pdf")
    slides(k, OUT / "slides.pptx")
    readme(k)
    print("Wrote report/writeup.pdf, report/slides.pptx, README.md results")
