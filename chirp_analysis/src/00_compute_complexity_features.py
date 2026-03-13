"""
00_compute_complexity_features.py
==================================
Computes complexity (FRCMSE-based) features for all subjects in metadata.csv
and writes chirp_analysis/data/complexity_features.csv.

Features are derived from the 45-scale FRCMSE curve stored per subject in the
entropy H5 files (pre-computed by the Julia pipeline).

Signal source
-------------
  <subject>_chirp_entropy.h5 → FRCMSE/0.2/electrode_mean/snr_7/curve  (shape: 45,)

  The curve is the Refined Composite Multiscale Sample Entropy of the SNR-filtered
  electrode-mean chirp signal, at scales 1–45 and tolerance r = 0.2 × std(signal).

Features (8 + 1 alias)
-----------------------
  nAUC_15   mean of curve[0:15]    (low-scale complexity)
  nAUC_30   mean of curve[15:30]   (mid-scale complexity)
  nAUC_45   mean of curve[30:45]   (high-scale complexity)
  nAUC_all  mean of curve[0:45]    (overall complexity)
  LRS_15    linregress slope of curve[0:15]
  LRS_30    linregress slope of curve[15:30]
  LRS_45    linregress slope of curve[30:45]
  LRS_all   linregress slope of curve[0:45]
  Chirp_Complexity  = nAUC_all  (alias used by ML scripts)

Usage (from chirp_analysis/ folder):
    python src/00_compute_complexity_features.py

Data directory (USB or local path to entropy H5 files):
    Edit ENTROPY_DIR below if your H5 files are in a different location.
"""

import os
import numpy as np
import pandas as pd
import h5py
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)

META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
OUT_CSV     = os.path.join(ROOT, 'data', 'complexity_features.csv')

# Path to per-subject entropy H5 files: <subject>_chirp_entropy.h5
ENTROPY_DIR = '/Volumes/USB/retina/chirp_analysis/entropy_data'


# ── Feature computation ────────────────────────────────────────────────────────
def load_curve(subject: str) -> np.ndarray | None:
    """Load 45-scale FRCMSE curve from entropy H5. Returns None if missing."""
    path = os.path.join(ENTROPY_DIR, f"{subject}_chirp_entropy.h5")
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, 'r') as f:
            key = 'FRCMSE/0.2/electrode_mean/snr_7/curve'
            if key not in f:
                return None
            return f[key][:]
    except Exception as e:
        print(f"  [warn] {subject}: {e}")
        return None


def compute_features(curve: np.ndarray) -> dict:
    """Derive nAUC and LRS features from 45-scale FRCMSE curve."""
    feats = {}

    # nAUC = mean of segment (equivalent to trapz(seg) / len(seg))
    for name, sl in [('nAUC_15', slice(0, 15)),
                     ('nAUC_30', slice(15, 30)),
                     ('nAUC_45', slice(30, 45)),
                     ('nAUC_all', slice(0, 45))]:
        seg = curve[sl]
        feats[name] = float(np.trapz(seg) / len(seg))

    # LRS = linear regression slope over the segment
    for name, sl in [('LRS_15', slice(0, 15)),
                     ('LRS_30', slice(15, 30)),
                     ('LRS_45', slice(30, 45)),
                     ('LRS_all', slice(0, 45))]:
        seg = curve[sl]
        slope, *_ = stats.linregress(np.arange(len(seg)), seg)
        feats[name] = float(slope)

    # Alias used by 07_complexity_ml.py
    feats['Chirp_Complexity'] = feats['nAUC_all']

    return feats


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df_meta = pd.read_csv(META_CSV)
    df_meta['Subject'] = df_meta['Subject'].str.strip()
    subjects = sorted(df_meta['Subject'].tolist())

    print(f"Computing complexity features for {len(subjects)} subjects from metadata.csv")
    print(f"Entropy H5 dir : {ENTROPY_DIR}")
    print(f"Output         : {OUT_CSV}\n")

    rows = []
    skipped = []
    for subj in subjects:
        curve = load_curve(subj)
        if curve is None:
            print(f"  SKIP  {subj}  (no snr_7 FRCMSE curve)")
            skipped.append(subj)
            continue
        feats = compute_features(curve)
        feats['Subject'] = subj
        rows.append(feats)
        print(f"  OK    {subj}  nAUC_all={feats['nAUC_all']:.6f}")

    col_order = ['Subject',
                 'nAUC_15', 'nAUC_30', 'nAUC_45', 'nAUC_all',
                 'LRS_15', 'LRS_30', 'LRS_45', 'LRS_all',
                 'Chirp_Complexity']
    df_out = pd.DataFrame(rows)[col_order]
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\n✓ Wrote {len(df_out)} subjects → {OUT_CSV}")
    if skipped:
        print(f"  Skipped {len(skipped)} (no snr_7 curve): {skipped}")


if __name__ == '__main__':
    main()
