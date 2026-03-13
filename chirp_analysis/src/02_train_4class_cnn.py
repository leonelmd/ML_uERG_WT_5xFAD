"""
02_train_4class_cnn.py
======================
4-class genotype×age classification of the Chirp stimulus:
  0: WT-Young  |  1: WT-Adult  |  2: 5xFAD-Young  |  3: 5xFAD-Adult

Trains on the FULL chirp signal and on individual chirp segments (flash,
frequency sweep, amplitude sweep) to characterise which parts of the
stimulus are most informative.

MODEL: FourClass1DCNN — same architecture as src_4class_robust/model.py
  Conv(1→16, k=25, s=4) → BN → ReLU → Dropout1d(0.1) → MaxPool(4)
  Conv(16→32, k=15, s=2) → BN → ReLU → Dropout1d(0.1) → MaxPool(4)
  Conv(32→64, k=9, s=1)  → BN → ReLU → AdaptiveAvgPool(1)
  FC: 64 → 32 → 16 → 4

PROTOCOL:
  * 5-fold StratifiedGroupKFold (strictly subject-disjoint)
  * Data augmentation: Gaussian noise (σ=0.02), time shift (±100), amplitude ×(0.8–1.2)
  * AdamW lr=1e-3, weight_decay=0.01
  * ReduceLROnPlateau scheduler (patience=10, factor=0.5)
  * CrossEntropyLoss with label_smoothing=0.1
  * Early stopping (patience=40 epochs)
  * **Post-convergence averaging**: avg probs over 5 epochs starting at best val-loss epoch
  * Evaluation: subject-level majority vote across trials

Usage (from chirp_analysis/ folder):
    python src/02_train_4class_cnn.py

NOTE: Edit DATA_DIR / META_CSV below to point to your raw H5 data.
"""

import os, sys, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

from dataset import ERGChirpDataset

# ── PATHS ──────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_DIR   = os.path.join(ROOT, 'data', 'cache')
FIG_DIR     = os.path.join(ROOT, 'results', 'figures')
TAB_DIR     = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 16
NUM_EPOCHS   = 200  # matches original src_4class_robust/train.py
PATIENCE     = 40
N_FOLDS      = 5
RANDOM_STATE = 42
POST_AVG_K   = 5    # number of epochs to average post-convergence (same as original)

CLASS_NAMES = ['WT-Young', 'WT-Adult', '5xFAD-Young', '5xFAD-Adult']
SEGMENTS    = ['full', 'flash', 'frequency', 'amplitude', 'amplitude_norm']


# ── Model — identical to src_4class_robust/model.py ───────────────────────────
class FourClass1DCNN(nn.Module):
    """
    1D CNN optimised for 4-class ERG classification.
    Uses GlobalAvgPool (AdaptiveAvgPool1d(1)) after the 3rd conv block so
    the same architecture works for any segment length.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 16, kernel_size=25, stride=4, padding=12),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout1d(0.1),
            nn.MaxPool1d(kernel_size=4, stride=4),
            # Block 2
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.1),
            nn.MaxPool1d(kernel_size=4, stride=4),
            # Block 3 — GlobalAvgPool makes length invariant
            nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.squeeze(-1)     # (B, 64)
        return self.fc(x)


# ── Augmentation helpers ───────────────────────────────────────────────────────
def augment(sig: torch.Tensor) -> torch.Tensor:
    sig = sig + torch.randn_like(sig) * 0.02
    shift = np.random.randint(-100, 100)
    sig = torch.roll(sig, shifts=shift, dims=-1)
    sig = sig * np.random.uniform(0.8, 1.2)
    return sig


# ── train one epoch ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, preds_all, labels_all = 0.0, [], []
    for signals, labels, _ in loader:
        signals = augment(signals).to(device)
        labels  = labels if isinstance(labels, torch.Tensor) \
                  else torch.tensor(labels)
        labels  = labels.to(device)
        optimizer.zero_grad()
        out  = model(signals)
        loss = criterion(out, labels)
        loss.backward(); optimizer.step()
        loss_sum += loss.item() * signals.size(0)
        preds_all.extend(out.argmax(1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return loss_sum / len(loader.dataset), accuracy_score(labels_all, preds_all)


# ── validate (subject-level majority vote) ─────────────────────────────────────
def validate(model, loader, criterion, device):
    """
    Returns: (loss, subj_acc, sl_labels, sl_preds, sl_probs_array)
    sl_probs_array shape: (n_subjects, 4) — mean prob across trials per subject
    """
    model.eval()
    loss_sum = 0.0
    votes = {}          # orig_sub → {label, preds:[], probs:[]}
    with torch.no_grad():
        for signals, labels, subjs in loader:
            signals = signals.to(device)
            labels  = labels if isinstance(labels, torch.Tensor) \
                      else torch.tensor(labels)
            labels_dev = labels.to(device)
            out = model(signals)
            loss_sum += criterion(out, labels_dev).item() * signals.size(0)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(1).cpu().tolist()
            for pred, lbl, s, prob in zip(preds, labels.tolist(), subjs, probs):
                orig = s.split('_trial_')[0]
                votes.setdefault(orig, {'label': lbl, 'preds': [], 'probs': []})
                votes[orig]['preds'].append(pred)
                votes[orig]['probs'].append(prob)

    sl_labels = [v['label'] for v in votes.values()]
    sl_preds  = [np.bincount(v['preds']).argmax() for v in votes.values()]
    sl_probs  = np.array([np.mean(v['probs'], axis=0) for v in votes.values()])
    acc       = accuracy_score(sl_labels, sl_preds)
    return loss_sum / len(loader.dataset), acc, sl_labels, sl_preds, sl_probs


# ── Run one segment ────────────────────────────────────────────────────────────
def run_segment(segment: str):
    print(f"\n{'='*60}\nSegment: {segment.upper()}\n{'='*60}")
    dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment=segment,
                              cache_dir=CACHE_DIR)

    orig_subjects = dataset.orig_subjects
    labels_4class = dataset.labels
    unique_subs   = list(dict.fromkeys(orig_subjects))  # order-preserving unique
    sub_labels    = [labels_4class[orig_subjects.index(s)] for s in unique_subs]

    sgkf   = StratifiedGroupKFold(n_splits=N_FOLDS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_labels, all_preds, all_probs = [], [], []
    fold_accs = []

    for fold, (tr_sub_idx, vl_sub_idx) in enumerate(
            sgkf.split(unique_subs, sub_labels, groups=unique_subs), 1):

        train_subs = set(unique_subs[i] for i in tr_sub_idx)
        val_subs   = set(unique_subs[i] for i in vl_sub_idx)
        tr_idx = [i for i, s in enumerate(orig_subjects) if s in train_subs]
        vl_idx = [i for i, s in enumerate(orig_subjects) if s in val_subs]

        tr_loader = DataLoader(Subset(dataset, tr_idx),
                                batch_size=BATCH_SIZE, shuffle=True)
        vl_loader = DataLoader(Subset(dataset, vl_idx),
                                batch_size=BATCH_SIZE, shuffle=False)

        model     = FourClass1DCNN().to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-5)

        best_val_loss = float('inf')
        best_epoch    = 0
        patience_cnt  = 0
        best_models   = {}  # epoch → state_dict

        for epoch in range(NUM_EPOCHS):
            t_loss, _ = train_one_epoch(model, tr_loader, criterion, optimizer, device)
            v_loss, v_acc, _, _, _ = validate(model, vl_loader, criterion, device)
            scheduler.step(v_loss)

            # Store every epoch's weights for post-convergence averaging
            best_models[epoch] = copy.deepcopy(model.state_dict())

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_epoch    = epoch
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  Early stop at epoch {epoch+1} "
                          f"(best epoch {best_epoch+1})")
                    break

        # ── POST-CONVERGENCE AVERAGING (matches original exactly) ─────────────
        # Average probabilities from POST_AVG_K epochs starting at best_epoch
        selected = [best_epoch + i for i in range(POST_AVG_K)
                    if (best_epoch + i) in best_models]
        print(f"  Fold {fold}: converged epoch={best_epoch+1}, "
              f"averaging epochs {[e+1 for e in selected]}")

        # Get reference label ordering from last checkpoint
        model.load_state_dict(best_models[best_epoch])
        _, _, sl_labels_ref, _, _ = validate(model, vl_loader, criterion, device)
        n_sub = len(sl_labels_ref)
        sum_probs = np.zeros((n_sub, 4))

        for e in selected:
            model.load_state_dict(best_models[e])
            _, _, sl_labels_e, _, sl_probs_e = validate(
                model, vl_loader, criterion, device)
            sum_probs += sl_probs_e

        avg_probs   = sum_probs / len(selected)
        final_preds = np.argmax(avg_probs, axis=1)
        acc         = accuracy_score(sl_labels_ref, final_preds)
        fold_accs.append(acc)
        all_labels.extend(sl_labels_ref)
        all_preds.extend(final_preds.tolist())
        all_probs.extend(avg_probs.tolist())
        print(f"  Fold {fold} Acc (post-conv avg) = {acc:.1%}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    acc     = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1      = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs, ddof=1)   # sample std, matches original
    print(f"\n  >>> SEGMENT {segment.upper()}: "
          f"Acc={acc:.2%}  BalAcc={bal_acc:.2%}  F1={f1:.3f}  "
          f"FoldMean={mean_acc:.1%} ± {std_acc:.1%}")

    return {
        'segment':  segment,
        'acc':      acc,
        'bal_acc':  bal_acc,
        'f1':       f1,
        'fold_mean_acc': mean_acc,
        'fold_std_acc':  std_acc,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"4-Class CNN — Chirp Segment Analysis")
    print(f"Segments: {SEGMENTS}")
    print(f"Epochs: {NUM_EPOCHS}  Patience: {PATIENCE}  "
          f"Post-avg: {POST_AVG_K} epochs  Folds: {N_FOLDS}\n")

    summary = []
    for seg in SEGMENTS:
        row = run_segment(seg)
        summary.append(row)

    df_sum = pd.DataFrame(summary)
    out_csv = os.path.join(TAB_DIR, '02_4class_segment_summary.csv')
    df_sum.to_csv(out_csv, index=False)

    print(f"\n{'='*60}")
    print(df_sum[['segment','fold_mean_acc','fold_std_acc','bal_acc','f1']].to_string(index=False))

    # ── Summary bar chart (matches overall_summary_plot.png style) ────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style('whitegrid')
    colors_vir = plt.cm.viridis(np.linspace(0.1, 0.9, len(SEGMENTS)))
    x = np.arange(len(SEGMENTS))
    bars = ax.bar(x, df_sum['fold_mean_acc'], color=colors_vir, width=0.6)
    ax.errorbar(x, df_sum['fold_mean_acc'], yerr=df_sum['fold_std_acc'],
                fmt='none', c='black', capsize=5, elinewidth=1.5)
    for i, row in df_sum.iterrows():
        ax.annotate(f"{row['fold_mean_acc']:.1%} ± {row['fold_std_acc']:.1%}",
                    (x[i], row['fold_mean_acc'] + row['fold_std_acc'] + 0.02),
                    ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color='black')
    ax.axhline(0.25, color='r', ls='--', label='Chance (25%)')
    ax.set_xticks(x); ax.set_xticklabels(df_sum['segment'], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title('Robust 4-Class Performance (Mean ± SD across 5 Folds)',
                 fontsize=14)
    ax.set_ylabel('Subject-Level Accuracy', fontsize=12)
    ax.set_xlabel('Stimulus Segment', fontsize=12)
    ax.legend(loc='upper right')
    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, '02_4class_summary.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Summary figure → {out_fig}")
    print(f"✓ Summary table  → {out_csv}")
