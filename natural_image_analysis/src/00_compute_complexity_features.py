"""
00_compute_complexity_features.py  (Natural Image)
====================================================
Computes complexity (FRCMSE-based) features for all subjects in metadata.csv
and writes natural_image_analysis/data/complexity_features.csv.

Signal source
-------------
  <subject>_natural_images_entropy.h5 → FRCMSE/0.2/electrode_mean/snr_7/curve  (shape: 45,)

Features (8)
------------
  nAUC_15   mean of curve[0:15]
  nAUC_30   mean of curve[15:30]
  nAUC_45   mean of curve[30:45]
  nAUC_all  mean of curve[0:45]
  LRS_15    linregress slope of curve[0:15]
  LRS_30    linregress slope of curve[15:30]
  LRS_45    linregress slope of curve[30:45]
  LRS_all   linregress slope of curve[0:45]

Usage (from natural_image_analysis/ folder):
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

ENTROPY_DIR = '/Volumes/USB/retina/natural_image_analysis/entropy_data'


# ── Feature computation ────────────────────────────────────────────────────────
def load_curve(subject: str) -> np.ndarray | None:
    """Load 45-scale FRCMSE curve from NI entropy H5. Returns None if missing."""
    for fname in [f"{subject}_natural_images_entropy.h5",
                  f"{subject}_ni_entropy.h5"]:
        path = os.path.join(ENTROPY_DIR, fname)
        if os.path.exists(path):
            try:
                with h5py.File(path, 'r') as f:
                    key = 'FRCMSE/0.2/electrode_mean/snr_7/curve'
                    if key not in f:
                        return None
                    return f[key][:]
            except Exception as e:
                print(f"  [warn] {subject}: {e}")
                return None
    return None


def compute_features(curve: np.ndarray) -> dict:
    """Derive nAUC and LRS features from 45-scale FRCMSE curve."""
    feats = {}
    for name, sl in [('nAUC_15', slice(0, 15)),
                     ('nAUC_30', slice(15, 30)),
                     ('nAUC_45', slice(30, 45)),
                     ('nAUC_all', slice(0, 45))]:
        seg = curve[sl]
        feats[name] = float(np.trapz(seg) / len(seg))

    for name, sl in [('LRS_15', slice(0, 15)),
                     ('LRS_30', slice(15, 30)),
                     ('LRS_45', slice(30, 45)),
                     ('LRS_all', slice(0, 45))]:
        seg = curve[sl]
        slope, *_ = stats.linregress(np.arange(len(seg)), seg)
        feats[name] = float(slope)

    return feats


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df_meta = pd.read_csv(META_CSV)
    df_meta['Subject'] = df_meta['Subject'].str.strip()
    subjects = sorted(df_meta['Subject'].tolist())

    print(f"Computing NI complexity features for {len(subjects)} subjects from metadata.csv")
    print(f"Entropy H5 dir : {ENTROPY_DIR}")
    print(f"Output         : {OUT_CSV}\n")

    rows = []
    skipped = []
    for subj in subjects:
        curve = load_curve(subj)
        if curve is None:
            print(f"  SKIP  {subj}  (no snr_7 FRCMSE curve or H5 not found)")
            skipped.append(subj)
            continue
        feats = compute_features(curve)
        feats['Subject'] = subj
        rows.append(feats)
        print(f"  OK    {subj}  nAUC_all={feats['nAUC_all']:.6f}")

    col_order = ['Subject',
                 'nAUC_15', 'nAUC_30', 'nAUC_45', 'nAUC_all',
                 'LRS_15', 'LRS_30', 'LRS_45', 'LRS_all']
    df_out = pd.DataFrame(rows)[col_order]
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\n✓ Wrote {len(df_out)} subjects → {OUT_CSV}")
    if skipped:
        print(f"  Skipped {len(skipped)}: {skipped}")


if __name__ == '__main__':
    main()
