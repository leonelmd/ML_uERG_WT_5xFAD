"""
00_compute_hc_features.py  (Natural Image)
==========================================
Computes hand-crafted features for all subjects in metadata.csv and writes
natural_image_analysis/data/hand_crafted_features.csv.

Features are extracted from electrode_mean/snr_7/data in each subject's
natural-image-processed H5 file (SNR-filtered, trial-averaged electrode mean).
Subjects whose H5 lacks snr_7 are skipped.

Signal source
-------------
  electrode_mean/snr_7/data  (shape 25000, fs=250 Hz, 100 s natural image)
  (10 trials × 10 s @ 250 Hz = 25000 samples)

Features
--------
Temporal:
  Signal_Max, Signal_Min, Signal_P2P, Signal_RMS, Signal_Std

Spectral (Welch PSD, 2-s window, on full signal):
  Power_Total  (0.5 – 100 Hz)
  Power_Low    (0.5 –   4 Hz)  ~ delta
  Power_Mid    (  4 –   8 Hz)  ~ theta
  Power_High   (  8 –  16 Hz)  ~ alpha

Usage (from natural_image_analysis/ folder):
    python src/00_compute_hc_features.py

Data directory (USB or local path to processed H5 files):
    Edit DATA_DIR below if your H5 files are in a different location.
"""

import os
import numpy as np
import pandas as pd
import h5py
from scipy.signal import welch

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)

META_CSV = os.path.join(ROOT, 'data', 'metadata.csv')
OUT_CSV  = os.path.join(ROOT, 'data', 'hand_crafted_features.csv')

# Path to the processed H5 files: <subject>_natural_images_processed.h5
DATA_DIR = '/Volumes/USB/retina/natural_image_analysis/processed_data'

FS = 250.0


# ── Feature computation ────────────────────────────────────────────────────────
def load_signal(subject: str) -> np.ndarray | None:
    """Load electrode_mean/snr_7/data from subject's NI H5. Returns None if missing."""
    # Try both naming conventions
    for fname in [f"{subject}_natural_images_processed.h5",
                  f"{subject}_ni_processed.h5"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            try:
                with h5py.File(path, 'r') as f:
                    key = 'electrode_mean/snr_7/data'
                    if key not in f:
                        return None
                    return f[key][:]
            except Exception as e:
                print(f"  [warn] {subject}: {e}")
                return None
    return None


def compute_features(sig: np.ndarray) -> dict:
    """Compute all 9 NI HC features from the full NI signal."""
    feats = {
        'Signal_Max': float(np.max(sig)),
        'Signal_Min': float(np.min(sig)),
        'Signal_P2P': float(np.max(sig) - np.min(sig)),
        'Signal_RMS': float(np.sqrt(np.mean(sig ** 2))),
        'Signal_Std': float(np.std(sig)),
    }

    freqs, psd = welch(sig, FS, nperseg=int(FS * 2))
    dx = freqs[1] - freqs[0]
    bands = {
        'Power_Total': (0.5, 100),
        'Power_Low':   (0.5,   4),
        'Power_Mid':   (4.0,   8),
        'Power_High':  (8.0,  16),
    }
    for name, (f_lo, f_hi) in bands.items():
        idx = (freqs >= f_lo) & (freqs <= f_hi)
        feats[name] = float(np.trapz(psd[idx], dx=dx))

    return feats


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df_meta = pd.read_csv(META_CSV)
    df_meta['Subject'] = df_meta['Subject'].str.strip()
    subjects = sorted(df_meta['Subject'].tolist())

    print(f"Computing NI HC features for {len(subjects)} subjects from metadata.csv")
    print(f"H5 data dir : {DATA_DIR}")
    print(f"Output      : {OUT_CSV}\n")

    rows = []
    skipped = []
    for subj in subjects:
        sig = load_signal(subj)
        if sig is None:
            print(f"  SKIP  {subj}  (no snr_7 signal or H5 not found)")
            skipped.append(subj)
            continue
        feats = compute_features(sig)
        feats['Subject'] = subj
        rows.append(feats)
        print(f"  OK    {subj}  Signal_RMS={feats['Signal_RMS']:.4f}")

    col_order = ['Subject',
                 'Signal_Max', 'Signal_Min', 'Signal_P2P', 'Signal_RMS', 'Signal_Std',
                 'Power_Total', 'Power_Low', 'Power_Mid', 'Power_High']
    df_out = pd.DataFrame(rows)[col_order]
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\n✓ Wrote {len(df_out)} subjects → {OUT_CSV}")
    if skipped:
        print(f"  Skipped {len(skipped)}: {skipped}")


if __name__ == '__main__':
    main()
