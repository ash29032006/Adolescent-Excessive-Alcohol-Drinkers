"""Dataset loaders. Every loader returns the same triple (timeseries, labels, phenotype),
so the rest of the pipeline never needs to know which dataset it is working on."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))  # override for quick tests on another folder


def load_abide(data_dir=DATA_DIR):
    """Load ABIDE as saved by download_data.py.

    Returns
    -------
    ts    : list of float arrays, one per person, shape (timepoints, 200 regions)
    y     : int array, 1 = autism, 0 = control
    pheno : DataFrame with at least SITE_ID, AGE_AT_SCAN, SEX (1 male, 2 female), DX_GROUP
    """
    files = [Path(data_dir) / f for f in ("timeseries.npy", "labels.npy", "phenotypic.csv")]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"Missing data files: {missing}\nRun `python download_data.py` first.")
    ts = [np.asarray(t, dtype=np.float64) for t in np.load(files[0], allow_pickle=True)]
    y = np.load(files[1]).astype(int)
    pheno = pd.read_csv(files[2])
    assert len(ts) == len(y) == len(pheno), "timeseries, labels and phenotype are out of sync"
    return ts, y, pheno


def load_ncanda(data_dir):
    """Placeholder for the paper's dataset. It must return the same triple as load_abide
    (label 1 = heavy drinker) and needs an NIAAA data use agreement to obtain."""
    raise NotImplementedError("NCANDA requires an NIAAA Data Use Certification.")
