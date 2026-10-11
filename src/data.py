
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


def load_abide(data_dir=DATA_DIR):
    files = [Path(data_dir) / f for f in ("timeseries.npy", "labels.npy", "phenotypic.csv")]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"Missing data files: {missing}\nRun `python download_data.py` first.")
    ts = [np.asarray(t, dtype=np.float64) for t in np.load(files[0], allow_pickle=True)]
    y = np.load(files[1]).astype(int)
    pheno = pd.read_csv(files[2])
    assert len(ts) == len(y) == len(pheno), "timeseries, labels and phenotype are out of sync"
    return ts, y, pheno


def load_ncanda(data_dir):"
    raise NotImplementedError("NCANDA requires an NIAAA Data Use Certification.")
