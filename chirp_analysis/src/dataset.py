"""
ERG Chirp Dataset
=================
Loads chirp-processed H5 files into PyTorch Dataset objects.
Supports 4-class labels (WT-Young, WT-Adult, 5xFAD-Young, 5xFAD-Adult)
and binary labels (WT=0, 5xFAD=1).

Segment options:
  'full'          - 0:8750  (full 35s chirp @ 250 Hz)
  'flash'         - 0:1875  (initial flash response)
  'frequency'     - 1875:6000 (frequency sweep)
  'amplitude'     - 6000:8750 (amplitude sweep)
  'amplitude_norm'- 6000:8750, normalized by flash peak amplitude
"""

import os
import re
import torch
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Map metadata.csv Group strings → plots.jl letter codes (used internally)
_GROUP_STR_TO_CODE = {
    'WT young':    'A',
    'WT adult':    'B',
    '5xFAD young': 'C',
    '5xFAD adult': 'D',
}

# ── Group → (genotype, age-class, 4-class label) ──────────────────────────────
GROUP_INFO = {
    'A': {'genotype': 'WT',    'age': 'young', 'label': 0},  # WT-Young
    'B': {'genotype': 'WT',    'age': 'adult', 'label': 1},  # WT-Adult
    'C': {'genotype': '5xFAD', 'age': 'young', 'label': 2},  # 5xFAD-Young
    'D': {'genotype': '5xFAD', 'age': 'adult', 'label': 3},  # 5xFAD-Adult
}

SEGMENT_SLICES = {
    'full':           (0,    8750),
    'flash':          (0,    1875),
    'frequency':      (1875, 6000),
    'amplitude':      (6000, 8750),
    'amplitude_norm': (6000, 8750),   # same slice, but normalised by flash peak
}


class ERGChirpDataset(Dataset):
    """
    Chirp ERG dataset with 4-class labels and flexible segmentation.

    Parameters
    ----------
    data_dir : str
        Directory containing ``<subject>_chirp_processed.h5`` files.
    plots_jl_path : str
        Path to ``plots.jl`` that lists subjects per group.
    segment : str
        One of the keys in SEGMENT_SLICES.
    cache_dir : str, optional
        If given, cache the processed dataset here as a ``.pt`` file.
    force_reprocess : bool
        If True, ignore the cached file and reprocess.

    SNR policy
    ----------
    Only subjects whose H5 contains ``electrode_mean/snr_7/data`` are loaded.
    This key is written by the Julia pipeline when at least one electrode passes
    SNR >= 7; its absence means the recording failed quality control.
    Per-trial signals are then averaged across all 252 electrodes — the
    subject-level gate is sufficient.
    """

    def __init__(self, data_dir, plots_jl_path, segment='amplitude_norm',
                 cache_dir=None, force_reprocess=False):
        self.data_dir       = data_dir
        self.segment        = segment

        # Storage
        self.signals  = []   # list of (1, L) float32 tensors
        self.labels   = []   # 4-class int
        self.subjects = []   # "<orig_subject>_trial_<n>"
        self.orig_subjects = []  # "<orig_subject>"

        # Cache
        cache_file = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f'chirp_{segment}.pt')

        if cache_file and os.path.exists(cache_file) and not force_reprocess:
            print(f"  [Dataset] Loading from cache: {cache_file}")
            c = torch.load(cache_file, weights_only=False)
            self.signals, self.labels, self.subjects, self.orig_subjects = (
                c['signals'], c['labels'], c['subjects'], c['orig_subjects'])
        else:
            self._build(plots_jl_path)   # accepts .csv or .jl
            if cache_file and len(self.signals) > 0:
                torch.save({'signals': self.signals, 'labels': self.labels,
                            'subjects': self.subjects,
                            'orig_subjects': self.orig_subjects}, cache_file)

    # ── build ──────────────────────────────────────────────────────────────────
    def _build(self, plots_jl_path):
        group_subjects = self._parse_metadata(plots_jl_path)
        seg   = self.segment.replace('_norm', '')   # strip suffix for slice lookup
        start, end = SEGMENT_SLICES[self.segment]

        for grp, info in GROUP_INFO.items():
            label = info['label']
            for subject in group_subjects.get(grp, []):
                h5_path = os.path.join(self.data_dir, f"{subject}_chirp_processed.h5")

                if not os.path.exists(h5_path):
                    continue

                # Quality gate: only load subjects whose Julia pipeline produced a
                # SNR-filtered electrode mean (snr_7).  Subjects without this key
                # failed the SNR threshold and must not be used.
                try:
                    with h5py.File(h5_path, 'r') as _f:
                        if 'electrode_mean/snr_7/data' not in _f:
                            print(f"  [Dataset] Skipping {subject}: no electrode_mean/snr_7/data")
                            continue
                except Exception:
                    continue

                good_elec = self._get_good_electrodes()

                try:
                    with h5py.File(h5_path, 'r') as f:
                        for trial in range(1, 11):
                            seg_signals, flash_peaks = [], []
                            for e in good_elec:
                                path = f"electrode_{e}/event_{trial}/normalized/data"
                                if path in f:
                                    full = f[path][:]
                                    seg_signals.append(full[start:end])
                                    if self.segment == 'amplitude_norm':
                                        flash_peaks.append(np.max(np.abs(full[1:1875])))

                            if not seg_signals:
                                continue

                            avg = np.mean(seg_signals, axis=0)
                            if self.segment == 'amplitude_norm' and flash_peaks:
                                fp = np.mean(flash_peaks)
                                if fp > 1e-6:
                                    avg = avg / fp

                            self.signals.append(
                                torch.tensor(avg, dtype=torch.float32).unsqueeze(0))
                            self.labels.append(label)
                            self.subjects.append(f"{subject}_trial_{trial}")
                            self.orig_subjects.append(subject)
                except Exception as e:
                    print(f"  [Dataset] Skipping {subject}: {e}")

        print(f"  [Dataset] Loaded {len(self.signals)} trials from "
              f"{len(set(self.orig_subjects))} subjects "
              f"(segment='{self.segment}')")

    def _get_good_electrodes(self):
        # All 252 electrodes (0-indexed) are included in the per-trial average.
        # Subject-level quality is enforced upstream via the electrode_mean/snr_7/data
        # existence check: only subjects whose Julia pipeline found enough high-SNR
        # electrodes to produce that key are loaded at all.
        return list(range(252))

    @staticmethod
    def _parse_metadata(path):
        """Load subject lists per group from either metadata.csv or plots.jl.

        Returns dict mapping group code (A/B/C/D) → list of subject IDs.
        Accepts:
          - <anything>.csv  : reads 'Subject' and 'Group' columns
          - <anything>.jl   : legacy Julia parser (requires plots.jl on disk)
        """
        if path.endswith('.csv'):
            df = pd.read_csv(path)
            groups = {}
            for _, row in df.iterrows():
                code = _GROUP_STR_TO_CODE.get(str(row['Group']).strip())
                if code:
                    groups.setdefault(code, []).append(str(row['Subject']).strip())
            return groups

        # Legacy: parse plots.jl
        groups = {}
        current = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                m = re.search(r'grouped_datasets\["([A-H])"\]\s*=\s*\[', line)
                if m:
                    current = m.group(1); groups[current] = []
                elif current and line.startswith('"') and line.endswith('",'):
                    groups[current].append(line.strip('",'))
                elif current and line.startswith(']'):
                    current = None
        return groups

    # Keep old name for any direct callers
    _parse_plots_jl = _parse_metadata

    # ── Pytorch Dataset protocol ───────────────────────────────────────────────
    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx], self.subjects[idx]

    # ── Convenience helpers ────────────────────────────────────────────────────
    def get_binary_label(self, idx):
        """Return 0 (WT) or 1 (5xFAD)."""
        return 1 if self.labels[idx] >= 2 else 0

    def get_age_binary(self, idx):
        """Return 0.0 (young) or 1.0 (adult) from 4-class label."""
        return float(self.labels[idx] % 2 == 1)
