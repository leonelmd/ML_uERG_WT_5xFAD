"""
00_compute_hc_features.py
=========================
Computes hand-crafted features for all subjects in metadata.csv and writes
chirp_analysis/data/hand_crafted_features.csv.

Features are extracted from electrode_mean/snr_7/data in each subject's
chirp-processed H5 file (SNR-filtered, trial-averaged electrode mean).
Subjects whose H5 lacks snr_7 (no electrode passed SNR >= 7) are skipped.

Signal source
-------------
  electrode_mean/snr_7/data  (shape 8750, fs=250 Hz, 35 s chirp)

Segment definitions (samples @ 250 Hz)
---------------------------------------
  Flash         :   0 – 1875
  Chirp Freq    : 1875 – 6000
  Chirp Amp     : 6000 – 8750

Features
--------
Temporal:
  Flash_Peak_Max, Flash_Peak_Min, Flash_Peak_P2P, Flash_RMS
  ChirpFreq_RMS, ChirpFreq_Std
  ChirpAmp_RMS, ChirpAmp_Max, ChirpAmp_P2P

Spectral (Welch PSD, 2-s window, on full 8750-sample signal):
  Power_Total  (0.5 – 100 Hz)
  Power_Low    (0.5 –   4 Hz)  ~ delta
  Power_Mid    (  4 –   8 Hz)  ~ theta
  Power_High   (  8 –  16 Hz)  ~ alpha

Usage (from chirp_analysis/ folder):
    python src/00_compute_hc_features.py

Data directory (USB or local path to processed H5 files):
    Edit DATA_DIR below if your H5 files are in a different location.
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
from scipy.signal import welch

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)          # chirp_analysis/

META_CSV = os.path.join(ROOT, 'data', 'metadata.csv')
OUT_CSV  = os.path.join(ROOT, 'data', 'hand_crafted_features.csv')

# Path to the processed H5 files (one per subject: <subject>_chirp_processed.h5)
DATA_DIR = '/Volumes/USB/retina/chirp_analysis/processed_data'

FS = 250.0   # Hz


# ── Feature computation ────────────────────────────────────────────────────────
def load_signal(subject: str) -> np.ndarray | None:
    """Load electrode_mean/snr_7/data from subject's chirp H5. Returns None if missing."""
    path = os.path.join(DATA_DIR, f"{subject}_chirp_processed.h5")
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, 'r') as f:
            key = 'electrode_mean/snr_7/data'
            if key not in f:
                return None
            return f[key][:]
    except Exception as e:
        print(f"  [warn] {subject}: {e}")
        return None


def compute_features(sig: np.ndarray) -> dict:
    """Compute all 13 HC features from an 8750-sample signal."""
    s_flash = sig[0:1875]
    s_freq  = sig[1875:6000]
    s_amp   = sig[6000:8750]

    feats = {
        'Flash_Peak_Max':  float(np.max(s_flash)),
        'Flash_Peak_Min':  float(np.min(s_flash)),
        'Flash_Peak_P2P':  float(np.max(s_flash) - np.min(s_flash)),
        'Flash_RMS':       float(np.sqrt(np.mean(s_flash ** 2))),
        'ChirpFreq_RMS':   float(np.sqrt(np.mean(s_freq ** 2))),
        'ChirpFreq_Std':   float(np.std(s_freq)),
        'ChirpAmp_RMS':    float(np.sqrt(np.mean(s_amp ** 2))),
        'ChirpAmp_Max':    float(np.max(s_amp)),
        'ChirpAmp_P2P':    float(np.max(s_amp) - np.min(s_amp)),
    }

    # Spectral power via Welch (2-s window = 500 samples at 250 Hz)
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

    print(f"Computing HC features for {len(subjects)} subjects from metadata.csv")
    print(f"H5 data dir : {DATA_DIR}")
    print(f"Output      : {OUT_CSV}\n")

    rows = []
    skipped = []
    for subj in subjects:
        sig = load_signal(subj)
        if sig is None:
            print(f"  SKIP  {subj}  (no snr_7 signal)")
            skipped.append(subj)
            continue
        feats = compute_features(sig)
        feats['Subject'] = subj
        rows.append(feats)
        print(f"  OK    {subj}  Flash_RMS={feats['Flash_RMS']:.4f}")

    col_order = ['Subject',
                 'Flash_Peak_Max', 'Flash_Peak_Min', 'Flash_Peak_P2P', 'Flash_RMS',
                 'ChirpFreq_RMS', 'ChirpFreq_Std',
                 'ChirpAmp_RMS', 'ChirpAmp_Max', 'ChirpAmp_P2P',
                 'Power_Total', 'Power_Low', 'Power_Mid', 'Power_High']
    df_out = pd.DataFrame(rows)[col_order]
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\n✓ Wrote {len(df_out)} subjects → {OUT_CSV}")
    if skipped:
        print(f"  Skipped {len(skipped)} (no snr_7): {skipped}")


if __name__ == '__main__':
    main()
