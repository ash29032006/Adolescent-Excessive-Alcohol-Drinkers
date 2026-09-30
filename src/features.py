"""Feature extraction.

Per-person features (range, correlation connectivity, demographics) use no information
from other people, so they are computed once up front. Tangent connectivity learns a
group reference from the training people, so it is a transformer fit inside CV.

Models receive subject *indices* as X; the Lookup / TangentFeatures transformers turn
indices into feature rows. This keeps one pipeline shape for every feature type.
"""
import hashlib

import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion


def range_features(ts):
    """The paper's derived feature: max minus min of every region's signal -> (n, 200)."""
    return np.stack([t.max(0) - t.min(0) for t in ts])


def connectivity_features(ts, kind="correlation"):
    """Correlation between every pair of regions (upper triangle) -> (n, 19900)."""
    cm = ConnectivityMeasure(kind=kind, vectorize=True, discard_diagonal=True)
    return np.nan_to_num(cm.fit_transform(ts))


def demographic_features(pheno):
    """Age, sex (1 = female) and one-hot site, the ABIDE equivalent of the paper's scanner type."""
    sites = pd.get_dummies(pheno["SITE_ID"], prefix="site", dtype=float)
    return np.column_stack([pheno["AGE_AT_SCAN"], (pheno["SEX"] == 2).astype(float), sites])


def feature_matrix(name, ts, pheno, idx=None):
    """Build a precomputed feature set such as "correlation+demo" for the people in idx."""
    idx = np.arange(len(ts)) if idx is None else np.asarray(idx)
    sub = [ts[i] for i in idx]
    blocks = {"range": lambda: range_features(sub),
              "correlation": lambda: connectivity_features(sub),
              "demo": lambda: demographic_features(pheno)[idx]}
    return np.hstack([blocks[b]() for b in name.split("+")]).astype(np.float32)


class Lookup(BaseEstimator, TransformerMixin):
    """Return precomputed rows for the given subject indices. Nothing is learned."""

    def __init__(self, table=None):
        self.table = table

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.table[np.ravel(X)]


_TANGENT_CACHE = {}  # fitting is slow (~1 min), so reuse it when the same training fold comes back


class TangentFeatures(BaseEstimator, TransformerMixin):
    """Tangent-space connectivity. The group mean covariance is learned from the training
    subjects only, which is why this must live inside the CV pipeline. Each person is then
    projected independently, so all people are projected once per fit and cached."""

    def __init__(self, ts=None):
        self.ts = ts

    def fit(self, X, y=None):
        idx = np.ravel(X)
        self.key_ = hashlib.md5(np.sort(idx).tobytes() + str(len(self.ts)).encode()).hexdigest()
        if self.key_ not in _TANGENT_CACHE:
            cm = ConnectivityMeasure(kind="tangent", vectorize=True, discard_diagonal=True).fit([self.ts[i] for i in idx])
            _TANGENT_CACHE[self.key_] = np.nan_to_num(cm.transform(self.ts)).astype(np.float32)
        return self

    def transform(self, X):
        return _TANGENT_CACHE[self.key_][np.ravel(X)]


def make_features(name, ts, tables):
    """Transformer for a feature set name; `tables` holds the precomputed per-person blocks."""
    blocks = [TangentFeatures(ts) if b == "tangent" else Lookup(tables[b]) for b in name.split("+")]
    return blocks[0] if len(blocks) == 1 else FeatureUnion([(f"f{i}", b) for i, b in enumerate(blocks)])
