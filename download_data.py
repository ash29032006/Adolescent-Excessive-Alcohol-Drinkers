from pathlib import Path
import numpy as np
from nilearn.datasets import fetch_abide_pcp

DATA_DIR = Path("data")
N_SUBJECTS = None        # test with 5 first, then change to None for everyone

abide = fetch_abide_pcp(
    data_dir=DATA_DIR,
    n_subjects=N_SUBJECTS,
    derivatives=["rois_cc200"],   # ONLY region time series (Craddock 200 regions)
    pipeline="cpac",
    band_pass_filtering=True,     # keeps the slow brain rhythms, removes noise
    global_signal_regression=False,
    quality_checked=True,         # drops scans that failed quality review
)

# each person = one array of shape (timepoints, 200 regions)
ts = [np.loadtxt(t) if isinstance(t, (str, Path)) else t for t in abide.rois_cc200]
pheno = abide.phenotypic

# label: DX_GROUP 1 = autism, 2 = control  →  convert to 1 / 0
y = (pheno["DX_GROUP"] == 1).astype(int).to_numpy()

print("people:", len(ts))
print("first person shape:", ts[0].shape)
print("class counts:", np.bincount(y))

np.save(DATA_DIR / "timeseries.npy", np.array(ts, dtype=object), allow_pickle=True)
np.save(DATA_DIR / "labels.npy", y)
pheno.to_csv(DATA_DIR / "phenotypic.csv", index=False)