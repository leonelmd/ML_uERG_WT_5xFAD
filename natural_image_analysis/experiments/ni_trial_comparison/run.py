"""
ni_trial_comparison/run.py
==========================
Quantifies the effect of treating NI repetitions as separate trials
(chirp-like) vs. stacking them as parallel channels (current approach).

Approach A — "10-channel" (current):
    One (10, 2500) tensor per subject, all 10 reps seen simultaneously.
    ImprovedNICNN_NoAge(input_ch=10), standard subject-level 5-fold CV.
    → Uses the trained weights from results/models/07_improved_ni_cnn_fold_*.pt
      for fast inference (no retraining needed).

Approach B — "trial-level" (chirp-like):
    One (1, 2500) tensor per trial, 10 entries per subject.
    ImprovedBinaryCNN(input_ch=1), GroupKFold(5) with groups=subject.
    At inference: average the 10 per-trial probabilities → subject-level AUC.
    → Trained from scratch in this script.

Comparison output (saved in experiments/ni_trial_comparison/results/):
    comparison_table.csv  — AUC, Acc, Sens, Spec for both approaches
    comparison_roc.png    — ROC curves + bar chart

Usage:
    cd /Users/leo/retina/machine_learning/natural_image_analysis
    python experiments/ni_trial_comparison/run.py
"""

import os, sys, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
NI_ROOT     = os.path.dirname(os.path.dirname(THIS_DIR))   # natural_image_analysis/
RETINA_ROOT = os.path.dirname(NI_ROOT)                      # retina/machine_learning/..

DATA_DIR    = os.path.join(RETINA_ROOT, '..', 'natural_image_analysis', 'processed_data')
DATA_DIR    = os.path.abspath(DATA_DIR)
META_CSV    = os.path.join(NI_ROOT, 'data', 'metadata.csv')
MOD_DIR     = os.path.join(NI_ROOT, 'results', 'models')
OUT_DIR     = os.path.join(THIS_DIR, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = (torch.device('mps')  if torch.backends.mps.is_available()  else
          torch.device('cuda') if torch.cuda.is_available()           else
          torch.device('cpu'))
print(f'Device: {DEVICE}')
print(f'Data  : {DATA_DIR}')

N_FOLDS  = 5
EPOCHS   = 80
BATCH    = 8    # same as main pipeline (drop_last=True handles small last batch)

# ── Shared architecture pieces ─────────────────────────────────────────────────

class TemporalStatPool(nn.Module):
    def forward(self, x):           # x: [B, C, T]
        return torch.cat([x.mean(-1), x.amax(-1),
                          x.std(-1, unbiased=False)], dim=1)  # [B, 3C]


class ImprovedCNN(nn.Module):
    """
    Shared 1D-CNN backbone for both approaches.
    in_channels=1  → Approach B (one trial at a time)
    in_channels=10 → Approach A (all reps stacked)
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 8,  15, stride=2, padding=7), nn.BatchNorm1d(8),  nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(8,          16, 11, stride=2, padding=5), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16,         32,  7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.pool = TemporalStatPool()          # 3 × 32 = 96
        self.fc   = nn.Sequential(
            nn.Linear(96, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.fc(self.pool(self.conv(self.norm(x))))


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_subject_reps(h5_path):
    """Load all 10 repetitions from one subject's H5 file.

    Averages across all electrodes present per event.
    Returns (10, 2500) float32 array, or None if file missing/unreadable.
    """
    if not os.path.exists(h5_path):
        return None
    reps = []
    try:
        with h5py.File(h5_path, 'r') as f:
            for ev in range(1, 11):
                sigs = []
                for e in range(252):
                    key = f'electrode_{e}/event_{ev}/normalized/data'
                    if key in f:
                        sigs.append(f[key][:2500])
                reps.append(np.mean(sigs, axis=0) if sigs else np.zeros(2500))
    except Exception as exc:
        print(f'  WARNING: {os.path.basename(h5_path)}: {exc}')
        return None
    return np.array(reps, dtype=np.float32)  # (10, 2500)


def load_metadata():
    df = pd.read_csv(META_CSV)
    df['Subject'] = df['Subject'].str.strip()
    return df


# ── Approach A dataset  (10-channel, current) ──────────────────────────────────

class NIDataset_10ch(Dataset):
    """One (10, 2500) sample per subject."""
    def __init__(self):
        df = load_metadata()
        self.signals, self.labels, self.subjects = [], [], []
        missing = []
        for _, row in df.iterrows():
            sub   = row['Subject']
            label = int(row['Label'])
            h5    = os.path.join(DATA_DIR, f'{sub}_natural_images_processed.h5')
            reps  = _load_subject_reps(h5)
            if reps is None:
                missing.append(sub)
                continue
            self.signals.append(torch.tensor(reps))          # (10, 2500)
            self.labels.append(label)
            self.subjects.append(sub)
        if missing:
            print(f'  [10ch] Skipped {len(missing)} subjects (no H5): {missing[:3]}…')
        print(f'  [10ch] {len(self.signals)} subjects loaded')

    def __len__(self):  return len(self.signals)
    def __getitem__(self, i): return self.signals[i], self.labels[i], self.subjects[i]


# ── Approach B dataset  (trial-level, chirp-like) ──────────────────────────────

class NIDataset_trial(Dataset):
    """One (1, 2500) sample per trial; 10 entries per subject."""
    def __init__(self):
        df = load_metadata()
        self.signals, self.labels, self.subjects, self.base_subjects = [], [], [], []
        missing = []
        for _, row in df.iterrows():
            sub   = row['Subject']
            label = int(row['Label'])
            h5    = os.path.join(DATA_DIR, f'{sub}_natural_images_processed.h5')
            reps  = _load_subject_reps(h5)
            if reps is None:
                missing.append(sub)
                continue
            for t in range(10):
                self.signals.append(torch.tensor(reps[t:t+1]))   # (1, 2500)
                self.labels.append(label)
                self.subjects.append(f'{sub}_trial_{t+1}')
                self.base_subjects.append(sub)
        if missing:
            print(f'  [trial] Skipped {len(missing)} subjects (no H5): {missing[:3]}…')
        print(f'  [trial] {len(self.signals)} trials from '
              f'{len(set(self.base_subjects))} subjects')

    def __len__(self):  return len(self.signals)
    def __getitem__(self, i):
        return self.signals[i], self.labels[i], self.subjects[i]


# ── Training helpers ───────────────────────────────────────────────────────────

def make_optimizer(model):
    """Match main pipeline: Adam lr=5e-4, wd=1e-2, CosineAnnealingLR."""
    opt   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return opt, sched


def train_epoch(model, loader, opt, crit):
    """Train one epoch with Gaussian noise augmentation (matches main pipeline)."""
    model.train()
    for X, y, _ in loader:
        X = X.to(DEVICE)
        X = X + torch.randn_like(X) * 0.01   # noise augmentation (same as 07_improved_ni_cnn.py)
        y = torch.tensor(y).to(DEVICE) if not isinstance(y, torch.Tensor) else y.to(DEVICE)
        opt.zero_grad()
        crit(model(X), y).backward()
        opt.step()


@torch.no_grad()
def _proba_and_labels(model, loader):
    """Return (probs, labels) arrays from a loader."""
    model.eval()
    probs, trues = [], []
    for X, y, _ in loader:
        p = torch.softmax(model(X.to(DEVICE)), 1)[:, 1].cpu().numpy()
        probs.extend(p)
        trues.extend(y.numpy() if isinstance(y, torch.Tensor) else y)
    return np.array(probs), np.array(trues)


@torch.no_grad()
def predict_subject_level(model, loader):
    """Average trial probabilities per subject → subject-level predictions."""
    model.eval()
    trial_probs, trial_labels = {}, {}
    for X, y, s_ids in loader:
        p = torch.softmax(model(X.to(DEVICE)), 1)[:, 1].cpu().numpy()
        labels = y.numpy() if isinstance(y, torch.Tensor) else list(y)
        for prob, label, sid in zip(p, labels, s_ids):
            base = sid.rsplit('_trial_', 1)[0]
            trial_probs.setdefault(base, []).append(float(prob))
            trial_labels[base] = int(label)
    subjs  = sorted(trial_probs)
    probs  = np.array([np.mean(trial_probs[s])  for s in subjs])
    labels = np.array([trial_labels[s]           for s in subjs])
    return probs, labels


# ── CV runners ────────────────────────────────────────────────────────────────

def run_cv_10ch(dataset):
    """Standard subject-level CV for 10-channel approach.
    Training recipe matches 07_improved_ni_cnn.py exactly:
      Adam lr=5e-4, wd=1e-2, CosineAnnealingLR, drop_last=True,
      noise augmentation, best-epoch model selection.
    """
    from sklearn.model_selection import StratifiedKFold
    subjects = dataset.subjects
    labels   = np.array(dataset.labels)
    indices  = np.arange(len(dataset))

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=42)
    subj_probs, subj_labels = {}, {}
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(indices, labels)):
        tr_loader = DataLoader(Subset(dataset, tr_idx), BATCH, shuffle=True,
                               drop_last=True, num_workers=0)
        va_loader = DataLoader(Subset(dataset, va_idx), BATCH, shuffle=False, num_workers=0)

        model = ImprovedCNN(in_channels=10).to(DEVICE)
        opt, sched = make_optimizer(model)
        crit  = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_auc, best_state = 0.0, None
        for _ in range(EPOCHS):
            train_epoch(model, tr_loader, opt, crit)
            sched.step()
            probs_v, trues_v = _proba_and_labels(model, va_loader)
            try:    v_auc = roc_auc_score(trues_v, probs_v)
            except: v_auc = 0.5
            if v_auc > best_auc:
                best_auc = v_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        probs, trues = _proba_and_labels(model, va_loader)
        va_subjs = [subjects[i] for i in va_idx]
        for s, p, l in zip(va_subjs, probs, trues):
            subj_probs[s] = p; subj_labels[s] = l

        try:    fa = roc_auc_score(trues, probs)
        except: fa = 0.5
        fold_aucs.append(fa)
        print(f'    fold {fold+1}/{N_FOLDS}: AUC={fa:.3f}  ({len(va_idx)} val subjects)')

    all_s  = sorted(subj_probs)
    all_p  = np.array([subj_probs[s]  for s in all_s])
    all_l  = np.array([subj_labels[s] for s in all_s])
    pooled = roc_auc_score(all_l, all_p)
    return fold_aucs, pooled, all_p, all_l


def run_cv_trial(dataset):
    """Trial-level training, subject-level inference (chirp-like).
    Same training recipe as run_cv_10ch but with GroupKFold to keep
    all 10 trials of each subject in the same fold.
    """
    base_subjects = dataset.base_subjects
    labels        = np.array(dataset.labels)
    indices       = np.arange(len(dataset))

    gkf = GroupKFold(N_FOLDS)
    subj_probs, subj_labels = {}, {}
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(indices, labels, groups=base_subjects)):

        tr_loader = DataLoader(Subset(dataset, tr_idx), BATCH, shuffle=True,
                               drop_last=True, num_workers=0)
        va_loader = DataLoader(Subset(dataset, va_idx), BATCH, shuffle=False, num_workers=0)

        model = ImprovedCNN(in_channels=1).to(DEVICE)
        opt, sched = make_optimizer(model)
        crit  = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_auc, best_state = 0.0, None
        for _ in range(EPOCHS):
            train_epoch(model, tr_loader, opt, crit)
            sched.step()
            probs_v, labels_v = predict_subject_level(model, va_loader)
            try:    v_auc = roc_auc_score(labels_v, probs_v)
            except: v_auc = 0.5
            if v_auc > best_auc:
                best_auc = v_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        probs, labels_subj = predict_subject_level(model, va_loader)
        n_subj = len(probs)

        va_base = sorted(set(base_subjects[i] for i in va_idx))
        for s, p, l in zip(va_base, probs, labels_subj):
            subj_probs[s] = p; subj_labels[s] = l

        try:    fa = roc_auc_score(labels_subj, probs)
        except: fa = 0.5
        fold_aucs.append(fa)
        print(f'    fold {fold+1}/{N_FOLDS}: AUC={fa:.3f}  ({n_subj} val subjects, '
              f'{len(va_idx)} val trials)')

    all_s  = sorted(subj_probs)
    all_p  = np.array([subj_probs[s]  for s in all_s])
    all_l  = np.array([subj_labels[s] for s in all_s])
    pooled = roc_auc_score(all_l, all_p)
    return fold_aucs, pooled, all_p, all_l


# ── Metrics helpers ───────────────────────────────────────────────────────────

def compute_metrics(probs, labels):
    preds = (probs >= 0.5).astype(int)
    return {
        'auc':  round(roc_auc_score(labels, probs), 4),
        'acc':  round(accuracy_score(labels, preds), 4),
        'sens': round(recall_score(labels, preds, pos_label=1, zero_division=0), 4),
        'spec': round(recall_score(labels, preds, pos_label=0, zero_division=0), 4),
    }


def bootstrap_auc(probs, labels, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(labels), len(labels), replace=True)
        if len(np.unique(labels[idx])) < 2: continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return np.std(aucs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '='*70)
    print('NI Trial-Level vs 10-Channel Comparison')
    print('='*70)

    # ── Approach A: 10-channel ────────────────────────────────────────────────
    print('\n[Approach A] Loading 10-channel dataset …')
    ds_10ch = NIDataset_10ch()

    if len(ds_10ch) == 0:
        print('ERROR: No H5 data found. Check DATA_DIR.')
        print(f'  Expected: {DATA_DIR}')
        sys.exit(1)

    print('\n[Approach A] Running 5-fold CV (10-channel, subject-level) …')
    fa_aucs, fa_pooled, fa_probs, fa_labels = run_cv_10ch(ds_10ch)
    fa_metrics = compute_metrics(fa_probs, fa_labels)
    fa_auc_std = bootstrap_auc(fa_probs, fa_labels)
    print(f'  → Pooled AUC={fa_pooled:.4f}  '
          f'fold-mean={np.mean(fa_aucs):.3f} ± {np.std(fa_aucs):.3f}')

    # ── Approach B: trial-level ───────────────────────────────────────────────
    print('\n[Approach B] Loading trial-level dataset …')
    ds_trial = NIDataset_trial()

    print('\n[Approach B] Running 5-fold CV (1-channel trial, subject-level vote) …')
    fb_aucs, fb_pooled, fb_probs, fb_labels = run_cv_trial(ds_trial)
    fb_metrics = compute_metrics(fb_probs, fb_labels)
    fb_auc_std = bootstrap_auc(fb_probs, fb_labels)
    print(f'  → Pooled AUC={fb_pooled:.4f}  '
          f'fold-mean={np.mean(fb_aucs):.3f} ± {np.std(fb_aucs):.3f}')

    # ── Save table ────────────────────────────────────────────────────────────
    rows = []
    for name, pooled, fold_aucs, metrics, auc_std in [
        ('10-channel (current)', fa_pooled, fa_aucs, fa_metrics, fa_auc_std),
        ('Trial-level (chirp-like)', fb_pooled, fb_aucs, fb_metrics, fb_auc_std),
    ]:
        row = {'approach': name,
               'pooled_auc':    pooled,
               'fold_mean_auc': round(np.mean(fold_aucs), 4),
               'fold_std_auc':  round(np.std(fold_aucs), 4),
               'bootstrap_auc_std': round(auc_std, 4),
               **{k: metrics[k] for k in ['acc', 'sens', 'spec']}}
        for i, a in enumerate(fold_aucs):
            row[f'fold{i+1}_auc'] = round(a, 4)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'comparison_table.csv')
    df_out.to_csv(csv_path, index=False)
    print(f'\nResults saved: {csv_path}')

    # ── Print summary ─────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f"{'Metric':<22}  {'10-channel':>12}  {'Trial-level':>12}  {'Δ':>8}")
    print('-'*60)
    for key, label in [('pooled_auc', 'Pooled AUC'),
                       ('fold_mean_auc', 'Fold-mean AUC'),
                       ('acc', 'Accuracy'),
                       ('sens', 'Sensitivity'),
                       ('spec', 'Specificity')]:
        va = rows[0][key]
        vb = rows[1][key]
        d  = vb - va
        print(f'{label:<22}  {va:>12.4f}  {vb:>12.4f}  {d:>+8.4f}')
    print()

    # Fold-level breakdown
    print('Fold-level AUC (10ch | trial):')
    for i in range(N_FOLDS):
        a = rows[0][f'fold{i+1}_auc']
        b = rows[1][f'fold{i+1}_auc']
        print(f'  Fold {i+1}: {a:.3f}  |  {b:.3f}  (Δ={b-a:+.3f})')

    # ── Figure ────────────────────────────────────────────────────────────────
    BG = '#F8FAFC'
    C_A  = '#2563EB'   # blue  — 10-channel
    C_B  = '#E11D48'   # red   — trial-level
    C_HC = '#16A34A'   # green — HC reference

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle('NI CNN: 10-channel (current) vs Trial-level (chirp-like)\n'
                 '5-fold subject-disjoint CV — subject-level AUC',
                 fontsize=15, fontweight='bold', y=1.02)

    # Panel A: ROC curves
    ax = axes[0]
    ax.plot([0,1],[0,1],'--', color='#94A3B8', lw=1.5, label='Chance')
    for probs, labels, color, name, pooled in [
        (fa_probs, fa_labels, C_A, '10-channel', fa_pooled),
        (fb_probs, fb_labels, C_B, 'Trial-level', fb_pooled),
    ]:
        fpr, tpr, _ = roc_curve(labels, probs)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name}  AUC={pooled:.3f}')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('A: ROC Curves (pooled CV)', fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.2); ax.set_facecolor('white')

    # Panel B: Fold-by-fold AUC
    ax = axes[1]
    x = np.arange(1, N_FOLDS + 1)
    ax.plot(x, fa_aucs, 'o-', color=C_A, lw=2, ms=8, label='10-channel')
    ax.plot(x, fb_aucs, 's-', color=C_B, lw=2, ms=8, label='Trial-level')
    ax.axhline(fa_pooled, color=C_A, ls='--', lw=1.2, alpha=0.6)
    ax.axhline(fb_pooled, color=C_B, ls='--', lw=1.2, alpha=0.6)
    ax.axhline(0.5, color='#94A3B8', ls=':', lw=1)
    ax.set_xticks(x); ax.set_xlabel('Fold'); ax.set_ylabel('AUC (subject-level)')
    ax.set_title('B: Per-fold AUC', fontweight='bold')
    ax.legend(fontsize=11); ax.set_ylim(0.3, 1.0)
    ax.grid(True, alpha=0.2); ax.set_facecolor('white')

    # Panel C: Metric comparison bar chart
    ax = axes[2]
    metrics_keys = ['pooled_auc', 'acc', 'sens', 'spec']
    metric_labels = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
    x = np.arange(len(metrics_keys))
    bw = 0.35
    for i, (row, color, name) in enumerate([
        (rows[0], C_A, '10-channel'), (rows[1], C_B, 'Trial-level')
    ]):
        vals = [row[k] for k in metrics_keys]
        ax.bar(x + (i - 0.5) * bw, vals, bw, color=color, alpha=0.85,
               label=name, edgecolor='white')
    # HC reference line
    ax.axhline(0.860, color=C_HC, ls='--', lw=1.5, label='HC ML (0.860)')
    ax.set_xticks(x); ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.set_title('C: Performance Metrics', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'comparison_roc.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'Figure saved: {fig_path}')

    # ── Interpretation note ───────────────────────────────────────────────────
    delta = fb_pooled - fa_pooled
    winner = 'Trial-level' if delta > 0 else '10-channel'
    print(f'\nConclusion:')
    print(f'  Δ AUC (trial - 10ch) = {delta:+.4f}')
    print(f'  → {winner} approach wins by {abs(delta):.4f} AUC points')
    if abs(delta) < 0.02:
        print('  → Difference is small (<0.02); architectural choice likely does not matter.')
    elif abs(delta) >= 0.05:
        print('  → Substantial difference; worth adopting the winning approach in the main pipeline.')
    else:
        print('  → Moderate difference; consider adopting if consistent across folds.')


if __name__ == '__main__':
    main()
