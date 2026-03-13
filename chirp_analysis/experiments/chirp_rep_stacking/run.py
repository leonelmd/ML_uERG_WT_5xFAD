"""
chirp_rep_stacking/run.py
=========================
Quantifies the effect of stacking all 10 chirp repetitions as input channels
(NI-like approach) vs. treating each repetition as a separate trial (current
chirp pipeline).

NOTE ON TERMINOLOGY
  "channel" below always means *repetition channel* (one of the 10 trial
  repeats stacked along the channel axis). It is NOT an electrode channel.
  Electrode averaging (SNR-filtered mean across electrodes) is applied first
  in both approaches, exactly as in the main pipeline.

Approach A — "trial-level" (current chirp pipeline):
    One (1, 2750) tensor per repetition, 10 per subject.
    ImprovedBinaryCNN(in_rep=1), GroupKFold(5) with groups=subject.
    At inference: average 10 per-rep probabilities → subject-level AUC.
    → Directly comparable to results/models/12_improved_amplitude_fold_*.pt
      (pooled AUC = 0.565)

Approach B — "rep-stacked" (NI-like):
    One (10, 2750) tensor per subject, all 10 reps seen simultaneously.
    ImprovedBinaryCNN(in_rep=10), StratifiedKFold(5) at subject level.
    → Does simultaneous access to all reps help exploit cross-rep reliability?

Both use the same training recipe as 12_improved_chirp_cnn.py:
    Adam lr=5e-4, wd=1e-2, CosineAnnealingLR, Gaussian noise aug (σ=0.01),
    drop_last=True, best-epoch model selection, label_smoothing=0.05.

Output (experiments/chirp_rep_stacking/results/):
    rep_stacking_table.csv   — AUC / Acc / Sens / Spec for both approaches
    rep_stacking_roc.png     — ROC curves + bar chart

Usage:
    cd /Users/leo/retina/machine_learning/chirp_analysis
    python experiments/chirp_rep_stacking/run.py
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
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
CHIRP_ROOT   = os.path.dirname(os.path.dirname(THIS_DIR))   # chirp_analysis/
RETINA_ROOT  = os.path.dirname(CHIRP_ROOT)                  # machine_learning/..

DATA_DIR     = os.path.abspath(os.path.join(RETINA_ROOT, '..', 'chirp_analysis', 'processed_data'))
META_CSV     = os.path.join(CHIRP_ROOT, 'data', 'metadata.csv')
OUT_DIR      = os.path.join(THIS_DIR, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = (torch.device('mps')  if torch.backends.mps.is_available()  else
          torch.device('cuda') if torch.cuda.is_available()           else
          torch.device('cpu'))
print(f'Device : {DEVICE}')
print(f'Data   : {DATA_DIR}')

# ── Experiment settings ────────────────────────────────────────────────────────
N_FOLDS      = 5
EPOCHS       = 80
BATCH        = 8          # same as main pipeline
SEGMENT_SLICE = (6000, 8750)   # amplitude segment → 2750 samples
SNR_THRESHOLD = 7.0            # same as ERGChirpDataset default

# ── Architecture ───────────────────────────────────────────────────────────────

class TemporalStatPool(nn.Module):
    def forward(self, x):          # x: (B, C, T)
        return torch.cat([x.mean(-1), x.amax(-1),
                          x.std(-1, unbiased=False)], dim=1)  # (B, 3C)


class ImprovedBinaryCNN(nn.Module):
    """
    1D-CNN matching 12_improved_chirp_cnn.py / ImprovedBinaryCNN in models.py.
    in_rep = 1  → Approach A (one repetition at a time)
    in_rep = 10 → Approach B (all 10 repetitions stacked as input channels)
    """
    def __init__(self, in_rep: int = 1):
        super().__init__()
        self.norm = nn.InstanceNorm1d(in_rep, affine=True)
        self.conv = nn.Sequential(
            nn.Conv1d(in_rep,  8, 15, stride=2, padding=7), nn.BatchNorm1d(8),  nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(8,      16, 11, stride=2, padding=5), nn.BatchNorm1d(16), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(16,     32,  7, stride=2, padding=3), nn.BatchNorm1d(32), nn.GELU(),
        )
        self.pool = TemporalStatPool()     # 3 × 32 = 96
        self.fc   = nn.Sequential(
            nn.Linear(96, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.fc(self.pool(self.conv(self.norm(x))))


# ── Data loading ───────────────────────────────────────────────────────────────

def _get_good_electrodes(data_dir: str, subject: str) -> list:
    snr_path = os.path.join(data_dir, 'data', f'{subject}_SNR.h5')
    if not os.path.exists(snr_path):
        return list(range(251))
    try:
        with h5py.File(snr_path, 'r') as sf:
            return [i for i in range(251)
                    if f'electrode_{i}/SNR' in sf
                    and sf[f'electrode_{i}/SNR'][()] >= SNR_THRESHOLD]
    except Exception:
        return list(range(251))


def _load_subject_reps(data_dir: str, subject: str) -> np.ndarray | None:
    """Load 10 amplitude-segment reps for one subject, electrode-averaged.

    Returns (10, 2750) float32 array, or None if file missing/unreadable.
    """
    h5_path = os.path.join(data_dir, f'{subject}_chirp_processed.h5')
    if not os.path.exists(h5_path):
        return None
    start, end = SEGMENT_SLICE
    good_elec  = _get_good_electrodes(data_dir, subject)
    reps = []
    try:
        with h5py.File(h5_path, 'r') as f:
            for trial in range(1, 11):
                sigs = []
                for e in good_elec:
                    key = f'electrode_{e}/event_{trial}/normalized/data'
                    if key in f:
                        sigs.append(f[key][start:end])
                reps.append(np.mean(sigs, axis=0) if sigs else np.zeros(end - start))
    except Exception as exc:
        print(f'  WARNING {subject}: {exc}')
        return None
    if len(reps) < 10:
        return None
    return np.array(reps, dtype=np.float32)   # (10, 2750)


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(META_CSV)
    df['Subject'] = df['Subject'].str.strip()
    # Binary label: WT=0, 5xFAD=1
    df['bin_label'] = df['Group'].str.contains('5xFAD').astype(int)
    return df


# ── Dataset classes ────────────────────────────────────────────────────────────

class ChirpDataset_TrialLevel(Dataset):
    """One (1, 2750) sample per repetition; 10 per subject (Approach A)."""

    def __init__(self):
        df = load_metadata()
        self.signals, self.labels, self.subjects, self.base_subjects = [], [], [], []
        missing = []
        for _, row in df.iterrows():
            sub   = row['Subject']
            label = int(row['bin_label'])
            reps  = _load_subject_reps(DATA_DIR, sub)
            if reps is None:
                missing.append(sub)
                continue
            for t in range(10):
                self.signals.append(torch.tensor(reps[t:t+1]))  # (1, 2750)
                self.labels.append(label)
                self.subjects.append(f'{sub}_rep_{t+1}')
                self.base_subjects.append(sub)
        if missing:
            print(f'  [trial] Skipped {len(missing)} (no H5): {missing[:4]}')
        print(f'  [trial] {len(self.signals)} reps from '
              f'{len(set(self.base_subjects))} subjects')

    def __len__(self):  return len(self.signals)
    def __getitem__(self, i): return self.signals[i], self.labels[i], self.subjects[i]


class ChirpDataset_RepStacked(Dataset):
    """One (10, 2750) sample per subject (Approach B — NI-like)."""

    def __init__(self):
        df = load_metadata()
        self.signals, self.labels, self.subjects = [], [], []
        missing = []
        for _, row in df.iterrows():
            sub   = row['Subject']
            label = int(row['bin_label'])
            reps  = _load_subject_reps(DATA_DIR, sub)
            if reps is None:
                missing.append(sub)
                continue
            self.signals.append(torch.tensor(reps))   # (10, 2750)
            self.labels.append(label)
            self.subjects.append(sub)
        if missing:
            print(f'  [stacked] Skipped {len(missing)} (no H5): {missing[:4]}')
        print(f'  [stacked] {len(self.signals)} subjects loaded')

    def __len__(self):  return len(self.signals)
    def __getitem__(self, i): return self.signals[i], self.labels[i], self.subjects[i]


# ── Training helpers ───────────────────────────────────────────────────────────

def make_optimizer(model):
    opt   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return opt, sched


def train_epoch(model, loader, opt, crit):
    model.train()
    for X, y, _ in loader:
        X = X.to(DEVICE) + torch.randn_like(X.to(DEVICE)) * 0.01   # noise aug
        y = (y if isinstance(y, torch.Tensor) else torch.tensor(y)).to(DEVICE)
        opt.zero_grad()
        crit(model(X), y).backward()
        opt.step()


@torch.no_grad()
def _proba_and_labels(model, loader):
    model.eval()
    probs, trues = [], []
    for X, y, _ in loader:
        p = torch.softmax(model(X.to(DEVICE)), 1)[:, 1].cpu().numpy()
        probs.extend(p)
        trues.extend(y.numpy() if isinstance(y, torch.Tensor) else y)
    return np.array(probs), np.array(trues)


@torch.no_grad()
def predict_subject_level(model, loader):
    """Average per-rep probabilities per subject → subject-level preds."""
    model.eval()
    rep_probs, rep_labels = {}, {}
    for X, y, s_ids in loader:
        p = torch.softmax(model(X.to(DEVICE)), 1)[:, 1].cpu().numpy()
        labels = y.numpy() if isinstance(y, torch.Tensor) else list(y)
        for prob, label, sid in zip(p, labels, s_ids):
            base = sid.rsplit('_rep_', 1)[0]
            rep_probs.setdefault(base, []).append(float(prob))
            rep_labels[base] = int(label)
    subjs  = sorted(rep_probs)
    probs  = np.array([np.mean(rep_probs[s])  for s in subjs])
    labels = np.array([rep_labels[s]          for s in subjs])
    return probs, labels


# ── CV runners ─────────────────────────────────────────────────────────────────

def run_cv_trial(dataset):
    """Approach A: trial-level training, subject-level vote at inference."""
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
        va_loader = DataLoader(Subset(dataset, va_idx), BATCH, shuffle=False,
                               num_workers=0)

        model = ImprovedBinaryCNN(in_rep=1).to(DEVICE)
        opt, sched = make_optimizer(model)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_auc, best_state = 0.0, None
        for _ in range(EPOCHS):
            train_epoch(model, tr_loader, opt, crit)
            sched.step()
            p_v, l_v = predict_subject_level(model, va_loader)
            try:    v_auc = roc_auc_score(l_v, p_v)
            except: v_auc = 0.5
            if v_auc > best_auc:
                best_auc = v_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        probs, labels_subj = predict_subject_level(model, va_loader)

        va_base = sorted(set(base_subjects[i] for i in va_idx))
        for s, p, l in zip(va_base, probs, labels_subj):
            subj_probs[s] = p; subj_labels[s] = l

        try:    fa = roc_auc_score(labels_subj, probs)
        except: fa = 0.5
        fold_aucs.append(fa)
        n_subj = len(probs)
        print(f'    fold {fold+1}/{N_FOLDS}: AUC={fa:.3f}  '
              f'({n_subj} val subjects, {len(va_idx)} val reps)')

    all_s  = sorted(subj_probs)
    all_p  = np.array([subj_probs[s]  for s in all_s])
    all_l  = np.array([subj_labels[s] for s in all_s])
    return fold_aucs, roc_auc_score(all_l, all_p), all_p, all_l


def run_cv_stacked(dataset):
    """Approach B: rep-stacked, StratifiedKFold at subject level."""
    subjects = dataset.subjects
    labels   = np.array(dataset.labels)
    indices  = np.arange(len(dataset))

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=42)
    subj_probs, subj_labels = {}, {}
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(indices, labels)):
        tr_loader = DataLoader(Subset(dataset, tr_idx), BATCH, shuffle=True,
                               drop_last=True, num_workers=0)
        va_loader = DataLoader(Subset(dataset, va_idx), BATCH, shuffle=False,
                               num_workers=0)

        model = ImprovedBinaryCNN(in_rep=10).to(DEVICE)
        opt, sched = make_optimizer(model)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)

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
        print(f'    fold {fold+1}/{N_FOLDS}: AUC={fa:.3f}  '
              f'({len(va_idx)} val subjects)')

    all_s  = sorted(subj_probs)
    all_p  = np.array([subj_probs[s]  for s in all_s])
    all_l  = np.array([subj_labels[s] for s in all_s])
    return fold_aucs, roc_auc_score(all_l, all_p), all_p, all_l


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(probs, labels):
    preds = (probs >= 0.5).astype(int)
    return {
        'auc':  round(roc_auc_score(labels, probs), 4),
        'acc':  round(accuracy_score(labels, preds), 4),
        'sens': round(recall_score(labels, preds, pos_label=1, zero_division=0), 4),
        'spec': round(recall_score(labels, preds, pos_label=0, zero_division=0), 4),
    }


def bootstrap_auc_std(probs, labels, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(labels), len(labels), replace=True)
        if len(np.unique(labels[idx])) < 2: continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return np.std(aucs)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '='*70)
    print('Chirp: Trial-level vs Rep-stacked (NI-like) comparison')
    print('Segment: amplitude (6000–8750, 2750 samples)')
    print('='*70)

    # ── Approach A: trial-level ───────────────────────────────────────────────
    print('\n[Approach A] Loading trial-level dataset …')
    ds_trial = ChirpDataset_TrialLevel()
    if len(ds_trial) == 0:
        print('ERROR: No H5 data found. Check DATA_DIR:', DATA_DIR)
        sys.exit(1)

    print('\n[Approach A] Running 5-fold CV (trial-level, subject-level vote) …')
    fa_aucs, fa_pooled, fa_probs, fa_labels = run_cv_trial(ds_trial)
    fa_metrics = compute_metrics(fa_probs, fa_labels)
    fa_std     = bootstrap_auc_std(fa_probs, fa_labels)
    print(f'  → Pooled AUC={fa_pooled:.4f}  '
          f'fold-mean={np.mean(fa_aucs):.3f} ± {np.std(fa_aucs):.3f}')

    # ── Approach B: rep-stacked ───────────────────────────────────────────────
    print('\n[Approach B] Loading rep-stacked dataset …')
    ds_stacked = ChirpDataset_RepStacked()

    print('\n[Approach B] Running 5-fold CV (10-rep stacked, subject-level) …')
    fb_aucs, fb_pooled, fb_probs, fb_labels = run_cv_stacked(ds_stacked)
    fb_metrics = compute_metrics(fb_probs, fb_labels)
    fb_std     = bootstrap_auc_std(fb_probs, fb_labels)
    print(f'  → Pooled AUC={fb_pooled:.4f}  '
          f'fold-mean={np.mean(fb_aucs):.3f} ± {np.std(fb_aucs):.3f}')

    # ── Save table ────────────────────────────────────────────────────────────
    rows = []
    for name, pooled, fold_aucs, metrics, auc_std in [
        ('Trial-level (current)',  fa_pooled, fa_aucs, fa_metrics, fa_std),
        ('Rep-stacked (NI-like)',  fb_pooled, fb_aucs, fb_metrics, fb_std),
    ]:
        row = {'approach': name, 'pooled_auc': pooled,
               'fold_mean_auc': round(np.mean(fold_aucs), 4),
               'fold_std_auc':  round(np.std(fold_aucs), 4),
               'bootstrap_auc_std': round(auc_std, 4),
               **{k: metrics[k] for k in ['acc', 'sens', 'spec']}}
        for i, a in enumerate(fold_aucs):
            row[f'fold{i+1}_auc'] = round(a, 4)
        rows.append(row)

    csv_path = os.path.join(OUT_DIR, 'rep_stacking_table.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'\nResults saved: {csv_path}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f"{'Metric':<22}  {'Trial-level':>12}  {'Rep-stacked':>12}  {'Δ':>8}")
    print('-'*60)
    for key, label in [('pooled_auc',    'Pooled AUC'),
                       ('fold_mean_auc', 'Fold-mean AUC'),
                       ('acc',           'Accuracy'),
                       ('sens',          'Sensitivity'),
                       ('spec',          'Specificity')]:
        va = rows[0][key]; vb = rows[1][key]
        print(f'{label:<22}  {va:>12.4f}  {vb:>12.4f}  {vb-va:>+8.4f}')
    print()

    print('Fold-level AUC (trial | stacked):')
    for i in range(N_FOLDS):
        a = rows[0][f'fold{i+1}_auc']; b = rows[1][f'fold{i+1}_auc']
        print(f'  Fold {i+1}: {a:.3f}  |  {b:.3f}  (Δ={b-a:+.3f})')

    # References
    print()
    print('Reference AUCs (from main pipeline):')
    print('  Improved chirp CNN (trial-level, amplitude): 0.601')
    print('  Chirp HC+Gain-Tracking (best overall):       0.810')
    print('  Chirp multichannel (flash, K=2, spatial):    0.789')

    # ── Figure ────────────────────────────────────────────────────────────────
    BG  = '#F8FAFC'
    C_A = '#2563EB'   # blue  — trial-level
    C_B = '#E11D48'   # red   — rep-stacked
    C_R = '#16A34A'   # green — reference lines

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle(
        'Chirp CNN: Trial-level (current) vs Rep-stacked (NI-like)\n'
        'Amplitude segment · 5-fold subject-disjoint CV · subject-level AUC',
        fontsize=14, fontweight='bold', y=1.02)

    # Panel A: ROC
    ax = axes[0]
    ax.plot([0,1],[0,1], '--', color='#94A3B8', lw=1.5, label='Chance')
    for probs, labels, color, name, pooled in [
        (fa_probs, fa_labels, C_A, 'Trial-level', fa_pooled),
        (fb_probs, fb_labels, C_B, 'Rep-stacked', fb_pooled),
    ]:
        fpr, tpr, _ = roc_curve(labels, probs)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name}  AUC={pooled:.3f}')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('A: ROC Curves (pooled CV)', fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.2); ax.set_facecolor('white')

    # Panel B: Fold-by-fold AUC
    ax = axes[1]
    x = np.arange(1, N_FOLDS + 1)
    ax.plot(x, fa_aucs, 'o-', color=C_A, lw=2, ms=8, label='Trial-level')
    ax.plot(x, fb_aucs, 's-', color=C_B, lw=2, ms=8, label='Rep-stacked')
    ax.axhline(fa_pooled, color=C_A, ls='--', lw=1.2, alpha=0.5)
    ax.axhline(fb_pooled, color=C_B, ls='--', lw=1.2, alpha=0.5)
    ax.axhline(0.601, color='#6B7280', ls=':', lw=1.2,
               label='Main pipeline (0.601)')
    ax.axhline(0.5, color='#94A3B8', ls=':', lw=1)
    ax.set_xticks(x); ax.set_xlabel('Fold'); ax.set_ylabel('AUC (subject-level)')
    ax.set_title('B: Per-fold AUC', fontweight='bold')
    ax.legend(fontsize=10); ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2); ax.set_facecolor('white')

    # Panel C: Metric bar chart
    ax = axes[2]
    keys   = ['pooled_auc', 'acc', 'sens', 'spec']
    klabels = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
    x = np.arange(len(keys)); bw = 0.35
    for i, (row, color, name) in enumerate([
        (rows[0], C_A, 'Trial-level'), (rows[1], C_B, 'Rep-stacked')
    ]):
        vals = [row[k] for k in keys]
        ax.bar(x + (i - 0.5)*bw, vals, bw, color=color, alpha=0.85,
               label=name, edgecolor='white')
    ax.axhline(0.810, color=C_R, ls='--', lw=1.5, label='HC+GT (0.810)')
    ax.axhline(0.601, color='#6B7280', ls=':', lw=1.2, label='CNN baseline (0.601)')
    ax.set_xticks(x); ax.set_xticklabels(klabels)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.set_title('C: Performance Metrics', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'rep_stacking_roc.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'Figure saved: {fig_path}')

    delta = fb_pooled - fa_pooled
    winner = 'Rep-stacked' if delta > 0 else 'Trial-level'
    print(f'\nConclusion: Δ AUC (stacked − trial) = {delta:+.4f}  → {winner} wins')
    if abs(delta) < 0.02:
        print('  Difference is small (<0.02): stacking reps provides no meaningful gain.')
    elif delta > 0:
        print('  Rep-stacking improves AUC — worth adopting in the main chirp pipeline.')
    else:
        print('  Trial-level approach is better for chirp.')


if __name__ == '__main__':
    main()
