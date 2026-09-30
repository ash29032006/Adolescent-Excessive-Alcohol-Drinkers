---
title: "Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data"
subtitle: "UE24CS352A Machine Learning mini project, PES University. Pipeline demonstrated on ABIDE resting-state fMRI"
author: "Team 35: Ashwin C H (PES1UG24AM900) and Chidivilas Adi (PES1UG24AM802)"
---

## Problem statement

- Can a classifier tell groups apart from **resting-state fMRI** alone?
- Paper (Kim, Liu, Noh, CS229): heavy adolescent drinkers vs others, NCANDA, 715 people
- Their best model: logistic regression on max minus min per region + demographics: **0.713** accuracy, **0.538** without age
- Our goals: reproduce the baseline, find better features, check reliance on **age** and **scanner site**

## Dataset: why ABIDE

- NCANDA needs an NIAAA Data Use Certification + committee review (up to 30 days): not possible in one week
- **ABIDE Preprocessed** (CPAC, band-pass, quality checked): same data type, same Craddock atlas family (200 regions)
- 871 people: 403 autism / 468 control, 20 sites, 78 to 316 timepoints
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

| Features | Model | Acc | F1 | AUC | Site acc | Site AUC |
|---|---|---|---|---|---|---|
| demo | Logistic regression | 0.532 ± 0.017 | 0.476 | 0.532 ± 0.047 | 0.481 | 0.499 |
| range | RBF SVM | 0.583 ± 0.027 | 0.547 | 0.620 ± 0.025 | 0.569 | 0.602 |
| range+demo | Logistic regression | 0.563 ± 0.022 | 0.537 | 0.594 ± 0.029 | 0.569 | 0.596 |
| correlation | Linear SVM | 0.668 ± 0.017 | 0.648 | 0.733 ± 0.020 | 0.673 | 0.737 |
| correlation+demo | Linear SVM | 0.667 ± 0.017 | 0.646 | 0.733 ± 0.020 | 0.673 | 0.737 |
| tangent | Linear SVM | 0.677 ± 0.023 | 0.660 | 0.740 ± 0.025 | 0.674 | 0.743 |
| tangent+demo | Linear SVM | 0.675 ± 0.021 | 0.658 | 0.740 ± 0.024 | 0.664 | 0.741 |
| timeseries | 1D CNN | 0.634 ± 0.028 | 0.613 | 0.705 ± 0.040 | 0.643 | 0.695 |
| none (dummy) | Dummy (majority) | 0.537 ± 0.003 | 0.000 | 0.500 ± 0.000 | 0.537 | 0.500 |

## Model comparison

![](figures/model_comparison.png)

## Age ablation

![](figures/age_ablation.png)

- Age ablation: demographics alone reach AUC 0.532; tangent brain features alone reach 0.740 and 0.740 with demographics added. Unlike the paper, the prediction is carried by the brain features, not by age.

## Site effect and confusion matrix

![](figures/confusion_matrix.png)

- Site effect: holding out whole sites moves the best model's AUC from 0.740 to 0.741 (+0.001). The model transfers to unseen scanners.

## Findings

- Best model: tangent+demo features with Linear SVM, accuracy 0.675 ± 0.021 and ROC AUC 0.740 ± 0.024 (stratified 5-fold CV), against 0.537 accuracy and AUC 0.5 for always guessing the majority class. Out of fold it caught 273/403 autism cases (sensitivity 0.68) and 315/468 controls (specificity 0.67).
- Paper reproduction: range features + demographics with logistic regression gave accuracy 0.563 (AUC 0.594); range features alone gave 0.583 (AUC 0.597). The paper reported 0.713 with age and 0.538 without on NCANDA (different task, so not directly comparable).
- Better features: correlation connectivity beats the paper's range features (best AUC 0.733 vs 0.620).
- Age ablation: demographics alone reach AUC 0.532; tangent brain features alone reach 0.740 and 0.740 with demographics added. Unlike the paper, the prediction is carried by the brain features, not by age.
- Site effect: holding out whole sites moves the best model's AUC from 0.740 to 0.741 (+0.001). The model transfers to unseen scanners.
- Deep learning: the 1D CNN on raw time series reached accuracy 0.634, AUC 0.705, below the linear models on connectivity: as in the paper, simple models on derived features win on small fMRI data.

## Conclusions

- Resting-state connectivity separates the classes above chance (0.675 vs 0.537 accuracy), and site-grouped CV (0.664) is the honest number to quote.
- Whole-brain correlation features are a stronger and simpler starting point than per-region signal range.
- Always report a dummy baseline, a demographics-only model and site-held-out CV: without them a model can look good for the wrong reasons (age or scanner).
- Next step: run the same pipeline on NCANDA once the data use agreement is approved (only src/data.py changes).

## Code and live demo

- `download_data.py` → `run_experiments.py` → `make_report.py`, `demo.py`
- `src/`: data, features, models, evaluate, deep, plots
- Demo: offline, predicts people the model never saw, shows their signal and connectivity

## Thank you

Questions?
