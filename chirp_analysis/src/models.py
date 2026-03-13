"""
CNN Architectures for Chirp ERG Classification
===============================================

CANONICAL / DEFAULT MODEL
--------------------------
  Class   : ImprovedBinaryCNN
  Segment : full  (0:8750 samples of the chirp)
  Weights : results/models/12_improved_full_fold_{1-5}.pt
  Pooled AUC (5-fold subject-disjoint CV): 0.590

Use load_default_ensemble(mod_dir, device) to load all five folds at once.

Architecture rationale vs legacy BinaryCNN_NoAge:
  1. InstanceNorm1d at input  — removes between-preparation amplitude drift.
  2. TemporalStatPool [mean, max, std] — preserves temporal structure
     that AdaptiveAvgPool1d(1) destroys.
  3. Smaller channels (8/16/32 vs 16/32/64) — less overfitting on N~46.
  4. Dropout 0.5 vs 0.3 — stronger regularisation.

DO NOT use BinaryCNN_NoAge or the weights from scripts 02/05 for new
analyses — those models perform near chance (pooled AUC ≈ 0.529).
"""

import os
import torch
import torch.nn as nn

# ── Canonical model registry ───────────────────────────────────────────────────
DEFAULT_MODEL_CLASS  = None   # set after class definition (see bottom of file)
DEFAULT_MODEL_PREFIX = '12_improved'   # filename prefix in results/models/
DEFAULT_SEGMENT      = 'full'          # chirp segment the default was trained on
DEFAULT_N_FOLDS      = 5


def load_default_ensemble(mod_dir, device=None):
    """
    Load the canonical ImprovedBinaryCNN ensemble (all 5 folds).

    Parameters
    ----------
    mod_dir : str
        Path to results/models/ directory.
    device : torch.device or None
        Target device.  Defaults to MPS → CUDA → CPU.

    Returns
    -------
    list[ImprovedBinaryCNN]
        Five models in eval mode with gradients frozen.

    Example
    -------
    >>> from models import load_default_ensemble
    >>> models = load_default_ensemble('results/models')
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    ensemble = []
    for fold in range(1, DEFAULT_N_FOLDS + 1):
        fname = f'{DEFAULT_MODEL_PREFIX}_{DEFAULT_SEGMENT}_fold_{fold}.pt'
        path  = os.path.join(mod_dir, fname)
        m = ImprovedBinaryCNN().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        ensemble.append(m)
    return ensemble

class BinaryCNN_NoAge(nn.Module):
    def __init__(self, signal_length=None):
        super().__init__()
        # Standard lightweight 1D CNN
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Handles any sequence length smoothly
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)

class BinaryCNN_AgeMetadata(nn.Module):
    # Left as placeholder for backwards compatibility if needed in step 3
    def __init__(self, signal_length=None):
        super().__init__()
        self.backbone = BinaryCNN_NoAge(signal_length)
        self.fc_head = nn.Sequential(
            nn.Linear(64 + 1, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x, age):
        if x.dim() == 2: x = x.unsqueeze(1)
        feat = self.backbone.conv(x).view(x.size(0), -1)
        if age.dim() == 1: age = age.unsqueeze(1)
        combined = torch.cat([feat, age], dim=1)
        return self.fc_head(combined)


class TemporalStatPool(nn.Module):
    """
    Multi-scale temporal aggregation: concatenates [mean, max, std] over the
    time dimension, replacing AdaptiveAvgPool1d(1).

    Rationale: global average pooling discards all temporal structure.
    Providing mean + peak (max) + variability (std) preserves amplitude,
    extremal, and dispersion information while keeping a fixed-size output.
    """
    def forward(self, x):
        # x: [B, C, T]
        return torch.cat([
            x.mean(dim=-1),                     # average activation
            x.amax(dim=-1),                     # peak activation
            x.std(dim=-1, unbiased=False),      # variability
        ], dim=1)                               # → [B, 3C]


class ImprovedBinaryCNN(nn.Module):
    """
    Improved 1D CNN for binary chirp ERG classification.

    Fixes vs BinaryCNN_NoAge:
      1. InstanceNorm1d at input  — per-sample z-score normalisation,
         removes amplitude baseline differences between preparations.
      2. TemporalStatPool         — replaces AdaptiveAvgPool1d(1); preserves
         mean, peak and variability of learned feature maps.
      3. Smaller channel counts (8→16→32) — fewer parameters relative to
         training set size (~370 trials from ~37 subjects per fold).
      4. Stronger dropout (0.5 vs 0.3) — additional regularisation.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 8,  kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(8),  nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(8,        16, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16,       32, kernel_size=7,  stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.temporal_pool = TemporalStatPool()  # 3 × 32 = 96 features
        self.fc = nn.Sequential(
            nn.Linear(96, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.input_norm(x)
        return self.fc(self.temporal_pool(self.conv(x)))


class ImprovedBinaryCNN_AgeMetadata(nn.Module):
    """
    Age-metadata variant of ImprovedBinaryCNN.
    The age scalar (normalised to [0, 1]) is concatenated with the
    96-dimensional TemporalStatPool feature vector before the FC head,
    giving 97 inputs to the classifier.
    Used only for the age-effect ablation study (script 03).
    """
    def __init__(self):
        super().__init__()
        self.backbone = ImprovedBinaryCNN()
        # Replace the backbone's FC with one that accepts 96 + 1 features
        self.fc = nn.Sequential(
            nn.Linear(97, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 2)
        )

    def forward(self, x, age):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.backbone.input_norm(x)
        feat = self.backbone.temporal_pool(self.backbone.conv(x))   # [B, 96]
        if age.dim() == 1: age = age.unsqueeze(1)                   # [B, 1]
        return self.fc(torch.cat([feat, age], dim=1))


# ── Attention-Pooling Architecture ────────────────────────────────────────────

import torch.nn.functional as F

class TemporalAttentionPool(nn.Module):
    """
    Replaces global pooling with a learned scalar attention score over T'
    temporal positions of the last conv feature maps.

    The softmax attention weights [B, T'] are directly interpretable as
    temporal importance — no post-hoc Grad-CAM or Integrated Gradients needed.

    Parameters: only C weights + 1 bias (same order as AdaptiveAvgPool1d).
    """
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Linear(channels, 1, bias=True)

    def forward(self, x):
        # x: [B, C, T']
        xt      = x.permute(0, 2, 1)                   # [B, T', C]
        weights = F.softmax(self.score(xt), dim=1)     # [B, T', 1]
        pooled  = (xt * weights).sum(dim=1)            # [B, C]
        return pooled, weights.squeeze(-1)             # [B, C], [B, T']


class AttentionBinaryCNN(nn.Module):
    """
    1D CNN for chirp ERG binary classification with Temporal Attention Pooling.

    Identical conv backbone to ImprovedBinaryCNN (InstanceNorm → 3× Conv-BN-ReLU
    with MaxPool) but replaces TemporalStatPool with TemporalAttentionPool.

    Key advantage: the attention weights per subject ARE the temporal heatmap.
    No post-hoc interpretability methods are required.

    Architecture:
      InstanceNorm1d → Conv(1→8,k=15,s=2) → BN → ReLU → MaxPool(4)
                     → Conv(8→16,k=11,s=2) → BN → ReLU → MaxPool(4)
                     → Conv(16→32,k=7,s=2) → BN → ReLU
                     → TemporalAttentionPool → [B, 32]
                     → Linear(32,16) → ReLU → Dropout(0.5) → Linear(16,2)

    Parameters (approx):
      Conv layers: ~same as ImprovedBinaryCNN (~3 K)
      Attention:   33  (Linear(32,1) with bias)
      FC head:     16×32+16 + 16×2+2 = 562  (vs 3170 in ImprovedBinaryCNN)
      Total:       ~4 K  vs ~7 K — smaller, better suited to N~46

    forward(x, return_attn=False):
      x            : [B, T] or [B, 1, T]
      return_attn  : if True, returns (logits [B,2], attn [B, T'])
                     where T' = floor(T / 128) approximately
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 8,  kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(8),  nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(8,        16, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16,       32, kernel_size=7,  stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.attn_pool = TemporalAttentionPool(channels=32)
        self.fc = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(16, 2)
        )

    def forward(self, x, return_attn=False):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.input_norm(x)
        feat = self.conv(x)                          # [B, 32, T']
        pooled, attn = self.attn_pool(feat)          # [B, 32], [B, T']
        out = self.fc(pooled)
        if return_attn:
            return out, attn
        return out


# ── Set canonical model class (must appear after ImprovedBinaryCNN is defined) ─
DEFAULT_MODEL_CLASS = ImprovedBinaryCNN
