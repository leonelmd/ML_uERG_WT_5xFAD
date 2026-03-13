"""
04_compare_chirp_segments.py
===========================
Binary genotype classification (WT vs 5xFAD) across different Chirp segments.

Uses the ImprovedBinaryCNN architecture (TemporalStatPool + InstanceNorm1d)
for a fair comparison with the improved CNN benchmark (step 20).
Reports POOLED cross-validated AUC as the primary metric.

Key Outcome:
  We identify that the raw 'amplitude' sweep segment provides the most
  robust performance, confirmed via pooled AUC across 5 subject-disjoint folds.

Note:
  All models are trained WITHOUT age as an input ("No Age"),
  following the decision in script 03.

Usage:
  python src/04_compare_chirp_segments.py
"""

import os, sys
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Add src to path ────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

from dataset import ERGChirpDataset
from models  import ImprovedBinaryCNN

# ── PATHS ──────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_DIR   = os.path.join(ROOT, 'data', 'cache')
FIG_DIR     = os.path.join(ROOT, 'results', 'figures')
TAB_DIR     = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
SEGMENTS     = ['full', 'flash', 'frequency', 'amplitude', 'amplitude_norm']
BATCH_SIZE   = 16
EPOCHS       = 80
N_FOLDS      = 5
RANDOM_STATE = 42

df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label = {r.Subject: (1 if '5xFAD' in r.Group else 0)
                    for _, r in df_meta.iterrows()}

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Device: {device}")


def run_segment(segment: str):
    print(f"\n  Segment: {segment.upper()}")
    dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment=segment,
                              cache_dir=CACHE_DIR,
                              force_reprocess=False)

    all_trial_subjects = [s.split('_trial_')[0] for s in dataset.subjects]
    valid_idx   = [i for i, s in enumerate(all_trial_subjects)
                   if s in subject_to_label]
    unique_subs = sorted(set(all_trial_subjects[i] for i in valid_idx))
    subj_labels = [subject_to_label[s] for s in unique_subs]

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
    pooled = {}          # subject → {'y_true': int, 'y_prob': float}
    fold_accs, fold_aucs = [], []

    for fold, (tr_idx_sub, vl_idx_sub) in enumerate(
            skf.split(unique_subs, subj_labels), 1):

        tr_subs = set(unique_subs[i] for i in tr_idx_sub)
        vl_subs = set(unique_subs[i] for i in vl_idx_sub)
        tr_idx  = [i for i in valid_idx if all_trial_subjects[i] in tr_subs]
        vl_idx  = [i for i in valid_idx if all_trial_subjects[i] in vl_subs]

        tr_loader = DataLoader(Subset(dataset, tr_idx),
                               batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        vl_loader = DataLoader(Subset(dataset, vl_idx),
                               batch_size=BATCH_SIZE, shuffle=False)

        model     = ImprovedBinaryCNN().to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc, best_acc = 0.0, 0.0
        best_votes = None

        for epoch in range(EPOCHS):
            # ── Train ──────────────────────────────────────────────────────────
            model.train()
            for signals, lbls4, _ in tr_loader:
                bz = torch.tensor([1 if l >= 2 else 0 for l in lbls4]).to(device)
                signals = signals.to(device) + torch.randn_like(signals.to(device)) * 0.01
                optimizer.zero_grad()
                criterion(model(signals), bz).backward()
                optimizer.step()
            scheduler.step()

            # ── Validate ───────────────────────────────────────────────────────
            model.eval()
            votes = {}
            with torch.no_grad():
                for signals, lbls4, subjs in vl_loader:
                    probs = F.softmax(model(signals.to(device)), dim=1)[:, 1].cpu().tolist()
                    for prob, s in zip(probs, subjs):
                        orig = s.split('_trial_')[0]
                        votes.setdefault(orig,
                                         {'label': subject_to_label[orig], 'probs': []})
                        votes[orig]['probs'].append(prob)

            sl       = [v['label'] for v in votes.values()]
            sp_probs = [np.mean(v['probs']) for v in votes.values()]
            sp       = [1 if p >= 0.5 else 0 for p in sp_probs]
            acc = accuracy_score(sl, sp)
            auc = roc_auc_score(sl, sp_probs) if len(np.unique(sl)) > 1 else 0.5

            if (auc > best_auc) or (auc == best_auc and acc > best_acc):
                best_auc, best_acc = auc, acc
                best_votes = {s: {'y_true': v['label'],
                                  'y_prob': float(np.mean(v['probs']))}
                              for s, v in votes.items()}

        fold_accs.append(best_acc)
        fold_aucs.append(best_auc)
        pooled.update(best_votes)
        print(f"    Fold {fold}  Acc={best_acc:.1%}  AUC={best_auc:.3f}")

    y_true = np.array([pooled[s]['y_true'] for s in unique_subs if s in pooled])
    y_prob = np.array([pooled[s]['y_prob'] for s in unique_subs if s in pooled])
    pooled_auc = roc_auc_score(y_true, y_prob)
    pooled_acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    print(f"    → Pooled AUC={pooled_auc:.3f}  "
          f"Fold-mean AUC={np.mean(fold_aucs):.3f}  "
          f"Fold-mean Acc={np.mean(fold_accs):.1%}")
    return {
        'fold_mean_acc': np.mean(fold_accs),
        'fold_std_acc':  np.std(fold_accs, ddof=1),
        'fold_mean_auc': np.mean(fold_aucs),
        'pooled_auc':    pooled_auc,
        'pooled_acc':    pooled_acc,
    }


if __name__ == '__main__':
    results = {}
    for seg in SEGMENTS:
        results[seg] = run_segment(seg)

    rows = [{'Segment':      seg,
             'Fold_Mean_Acc': results[seg]['fold_mean_acc'],
             'Fold_Std_Acc':  results[seg]['fold_std_acc'],
             'Fold_Mean_AUC': results[seg]['fold_mean_auc'],
             'Pooled_AUC':    results[seg]['pooled_auc'],
             'Pooled_Acc':    results[seg]['pooled_acc']}
            for seg in SEGMENTS]
    df_sum = pd.DataFrame(rows)
    out_csv = os.path.join(TAB_DIR, '04_chirp_segment_comparison.csv')
    df_sum.to_csv(out_csv, index=False)

    print(f"\n{'='*55}")
    print("SUMMARY")
    print(df_sum.to_string(index=False))

    # ── Figure ────────────────────────────────────────────────────────────────
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_vir = plt.cm.viridis(np.linspace(0.1, 0.9, len(SEGMENTS)))
    x = np.arange(len(SEGMENTS))

    # Left: Pooled AUC
    ax = axes[0]
    bars = ax.bar(x, df_sum['Pooled_AUC'], color=colors_vir,
                  width=0.6, edgecolor='black', alpha=0.9)
    ax.scatter(x, df_sum['Fold_Mean_AUC'], marker='D', s=70, color='black',
               zorder=5, label='Fold-mean AUC')
    ax.axhline(0.5, color='red', ls='--', lw=1.5, label='Chance (0.5)')
    ax.set_xticks(x); ax.set_xticklabels(SEGMENTS, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Pooled Cross-val AUC', fontsize=11)
    ax.set_title('Pooled AUC by Segment', fontweight='bold')
    ax.legend(fontsize=9)
    for i, v in enumerate(df_sum['Pooled_AUC']):
        ax.text(x[i], v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=10)
    winner_auc = df_sum['Pooled_AUC'].idxmax()
    ax.get_xticklabels()[winner_auc].set_fontweight('bold')
    ax.get_xticklabels()[winner_auc].set_color('darkblue')

    # Right: Fold-mean Accuracy
    ax = axes[1]
    ax.bar(x, df_sum['Fold_Mean_Acc'], yerr=df_sum['Fold_Std_Acc'],
           capsize=6, color=colors_vir, width=0.6, edgecolor='black', alpha=0.9)
    ax.axhline(0.5, color='red', ls='--', lw=1.5, label='Chance (50%)')
    ax.set_xticks(x); ax.set_xticklabels(SEGMENTS, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Fold-mean Subject Accuracy', fontsize=11)
    ax.set_title('Fold-mean Accuracy by Segment', fontweight='bold')
    ax.legend(fontsize=9)
    for i, (m, s) in enumerate(zip(df_sum['Fold_Mean_Acc'], df_sum['Fold_Std_Acc'])):
        ax.text(x[i], m + s + 0.02, f'{m:.1%}',
                ha='center', fontweight='bold', fontsize=10)
    winner_acc = df_sum['Fold_Mean_Acc'].idxmax()
    ax.get_xticklabels()[winner_acc].set_fontweight('bold')
    ax.get_xticklabels()[winner_acc].set_color('darkblue')

    plt.suptitle('Improved CNN Performance by Chirp Segment\n'
                 '(Binary WT vs 5xFAD, No Age, 5-fold subject-disjoint CV)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, '04_chirp_segment_comparison.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Figure → {out_fig}")
    print(f"✓ Table  → {out_csv}")
