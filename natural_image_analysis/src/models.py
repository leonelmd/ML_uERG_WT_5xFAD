"""
Natural Image Dataset & CNN Architecture
=========================================
NaturalImageDataset: loads H5 processed files.
Multi-scale CNN with Metadata support.
"""

import os
import re
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Maps metadata.csv Group strings → internal A/B/C/D codes
_NI_GROUP_INFO = {
    'WT young':    {'genotype': 'WT',    'age': 3.0},
    'WT adult':    {'genotype': 'WT',    'age': 6.0},
    '5xFAD young': {'genotype': '5xFAD', 'age': 3.0},
    '5xFAD adult': {'genotype': '5xFAD', 'age': 6.0},
}

def parse_ni_metadata(metadata_path: str) -> dict:
    """Return per-subject metadata dict.

    Accepts either a metadata.csv (preferred, self-contained) or the legacy
    plots.jl file (requires the Julia source to be present on disk).

    CSV columns used: Subject, Group, Age (Days), Sex, Label.
    """
    if metadata_path.endswith('.csv'):
        df = pd.read_csv(metadata_path)
        metadata = {}
        for _, row in df.iterrows():
            sub  = str(row['Subject']).strip()
            grp  = str(row['Group']).strip()
            info = _NI_GROUP_INFO.get(grp, {})
            if not info:
                continue
            sex = str(row.get('Sex', '')).strip().lower()
            age = float(row.get('Age (Days)', info['age'] * 30)) / 180.0  # normalise to [0,1]
            metadata[sub] = {
                'genotype': info['genotype'],
                'age':      info['age'],
                'sex':      sex,
                'label':    int(row['Label']),
                'age_norm': min(age, 1.0),
                'sex_bin':  1.0 if sex == 'female' else 0.0,
            }
        return metadata

    # Legacy: parse plots.jl
    GROUP_INFO = {
        'A': {'genotype': 'WT',    'age': 3.0, 'male_idx': (0, 9)},
        'B': {'genotype': 'WT',    'age': 6.0, 'male_idx': (0, 4)},
        'C': {'genotype': '5xFAD', 'age': 3.0, 'male_idx': (0, 2)},
        'D': {'genotype': '5xFAD', 'age': 6.0, 'male_idx': (0, 7)},
    }
    grouped = {}
    with open(metadata_path) as f:
        content = f.read()
    for grp in ['A', 'B', 'C', 'D']:
        m = re.search(r'grouped_datasets\["' + grp + r'"\]\s*=\s*\[(.*?)\]',
                      content, re.DOTALL)
        if m:
            grouped[grp] = re.findall(r'"(.*?)"', m.group(1))

    metadata = {}
    for grp, info in GROUP_INFO.items():
        for i, sub in enumerate(grouped.get(grp, [])):
            sex = 'male' if info['male_idx'][0] <= i < info['male_idx'][1] else 'female'
            metadata[sub] = {
                'genotype': info['genotype'], 'age': info['age'], 'sex': sex,
                'label': 0 if info['genotype'] == 'WT' else 1,
                'age_norm': info['age'] / 6.0, 'sex_bin': 1.0 if sex == 'female' else 0.0,
            }
    return metadata

class NaturalImageDataset(Dataset):
    """
    SNR policy
    ----------
    Only subjects whose H5 contains ``electrode_mean/snr_7/data`` are loaded.
    This key is written by the Julia pipeline when at least one electrode passes
    SNR >= 7; its absence means the recording failed quality control.
    Per-trial signals are then averaged across all 252 electrodes — the
    subject-level gate is sufficient.
    """
    def __init__(self, data_dir, metadata, cache_path=None):
        self.data_dir = data_dir
        self.metadata = metadata
        all_signals, all_meta, all_labels, all_subjects = [], [], [], []
        if cache_path and os.path.exists(cache_path):
            c = torch.load(cache_path, weights_only=False)
            all_signals, all_meta, all_labels, all_subjects = c['data'], c['meta'], c['labels'], c['subjects']
        else:
            for sub, meta in sorted(metadata.items()):
                h5 = os.path.join(data_dir, f"{sub}_natural_images_processed.h5")
                if not os.path.exists(h5): continue
                # Quality gate: skip subjects without SNR-filtered electrode mean
                try:
                    with h5py.File(h5, 'r') as _f:
                        if 'electrode_mean/snr_7/data' not in _f:
                            print(f"  [Dataset] Skipping {sub}: no electrode_mean/snr_7/data")
                            continue
                except Exception:
                    continue
                reps = self._load_repetitions(h5)
                all_signals.append(torch.tensor(reps, dtype=torch.float32))
                all_meta.append(torch.tensor([meta['age_norm'], meta['sex_bin']], dtype=torch.float32))
                all_labels.append(torch.tensor(meta['label'], dtype=torch.long))
                all_subjects.append(sub)
            if cache_path:
                torch.save({'data': all_signals, 'meta': all_meta, 'labels': all_labels, 'subjects': all_subjects}, cache_path)
        self.all_data = all_signals; self.all_meta = all_meta; self.all_labels = all_labels; self.all_subjects = all_subjects

    def _load_repetitions(self, h5_path):
        reps = []
        with h5py.File(h5_path, 'r') as f:
            for ev in range(1, 11):
                trial_sigs = []
                for e in range(252):  # all 252 electrodes; subject passes SNR gate above
                    key = f"electrode_{e}/event_{ev}/normalized/data"
                    if key in f: trial_sigs.append(f[key][:])
                reps.append(np.mean(trial_sigs, axis=0) if trial_sigs else np.zeros(2500))
        return np.array(reps, dtype=np.float32)

    def __len__(self): return len(self.all_subjects)
    def __getitem__(self, idx): return (self.all_data[idx], self.all_meta[idx], self.all_labels[idx], self.all_subjects[idx])

class NaturalImageCNN_NoAge(nn.Module):
    def __init__(self, input_ch=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_ch, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        feat = self.conv(x).view(x.size(0), -1)
        return self.fc(feat)

class NaturalImageCNN_AgeMetadata(nn.Module):
    # Left as placeholder for backwards compatibility if needed
    def __init__(self, input_ch=10):
        super().__init__()
        self.backbone = NaturalImageCNN_NoAge(input_ch)
        self.fc_head  = nn.Sequential(
            nn.Linear(64 + 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x, m):
        feat = self.backbone.conv(x).view(x.size(0), -1)
        combined = torch.cat([feat, m], dim=1)
        return self.fc_head(combined)

class DualInputCNN_NoAge(nn.Module):
    def __init__(self, ni_ch=10, c_ch=1):
        super().__init__()
        self.ni_branch = NaturalImageCNN_NoAge(ni_ch)
        self.c_branch  = NaturalImageCNN_NoAge(c_ch)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, ni_x, c_x):
        n = self.ni_branch.conv(ni_x).view(ni_x.size(0), -1)
        c = self.c_branch.conv(c_x).view(c_x.size(0), -1)
        return self.fc(torch.cat([n, c], dim=1))

class ImprovedNICNN_NoAge(nn.Module):
    """
    Improved NI CNN — same architectural fixes as ImprovedBinaryCNN:
      1. InstanceNorm1d at input per-sample normalisation.
      2. TemporalStatPool: [mean, max, std] instead of AdaptiveAvgPool1d(1).
      3. Smaller channel counts (8/16/32 vs 16/32/64).
      4. Stronger dropout (0.5).
    Input: (B, 10, 2500) — 10 repetitions as channels, same as original.
    """
    def __init__(self, input_ch=10):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(input_ch, affine=True)
        self.conv = nn.Sequential(
            nn.Conv1d(input_ch, 8,  kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(8),  nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(8,        16, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16,       32, kernel_size=7,  stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
        )
        # [mean, max, std] → 3 × 32 = 96 features
        self.fc = nn.Sequential(
            nn.Linear(96, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.input_norm(x)
        feat = self.conv(x)
        pooled = torch.cat([feat.mean(dim=-1),
                            feat.amax(dim=-1),
                            feat.std(dim=-1, unbiased=False)], dim=1)
        return self.fc(pooled)


class DualInputCNN_AgeMetadata(nn.Module):
    def __init__(self, ni_ch=10, c_ch=1):
        super().__init__()
        self.ni_branch = NaturalImageCNN_NoAge(ni_ch)
        self.c_branch  = NaturalImageCNN_NoAge(c_ch)
        self.fc = nn.Sequential(
            nn.Linear(128 + 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, ni_x, c_x, m):
        n = self.ni_branch.conv(ni_x).view(ni_x.size(0), -1)
        c = self.c_branch.conv(c_x).view(c_x.size(0), -1)
        return self.fc(torch.cat([n, c, m], dim=1))
