"""Model pipelines: features -> GridSearchCV(scaler -> classifier).

The feature step is fit on the outer training fold only. The scaler and the classifier's
C are tuned by an inner 3-fold CV on that same training fold, so test folds are never seen.
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

SEED = 42
MODELS = ["dummy", "logreg", "linear_svm", "rbf_svm"]
GRIDS = {"logreg": [1e-3, 1e-2, 1e-1, 1],       # L2 strength, smaller C = stronger regularisation
         "linear_svm": [1e-4, 1e-3, 1e-2, 1e-1],
         "rbf_svm": [0.1, 1, 10, 100]}


def make_classifier(name):
    """Unfitted classifier; class_weight='balanced' handles the class imbalance."""
    return {"dummy": DummyClassifier(strategy="most_frequent"),
            "logreg": LogisticRegression(class_weight="balanced", max_iter=5000),
            "linear_svm": LinearSVC(class_weight="balanced", max_iter=20000),
            "rbf_svm": SVC(kernel="rbf", class_weight="balanced")}[name]


def make_model(name, features="passthrough"):
    """Full leak-free pipeline. `features` maps X to a feature matrix ('passthrough' if X already is one)."""
    inner = Pipeline([("scale", StandardScaler()), ("clf", make_classifier(name))])
    if name in GRIDS:
        inner = GridSearchCV(inner, {"clf__C": GRIDS[name]}, scoring="roc_auc",
                             cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
    return Pipeline([("features", features), ("model", inner)])
