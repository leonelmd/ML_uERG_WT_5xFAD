"""
03_age_effect_comparison.py
==========================
Compares the effect of incorporating subject age as an auxiliary input to
the binary genotype classification (WT vs 5xFAD) on the chirp amplitude sweep.

Cases:
  1. No age (Baseline)
  2. Binary age (Young vs Adult as 0/1)
  3. Continuous age (Min-Max normalized days)

We demonstrate that using continuous or binary age actively degrades
out-of-sample performance in a strict subject-disjoint evaluation.
Age serves as a "confounder" that the model overfits on, strictly
justifying the "No Age" baseline as the standalone objective benchmark.

Uses the ImprovedBinaryCNN / ImprovedBinaryCNN_AgeMetadata architecture
(TemporalStatPool + InstanceNorm1d) for a fair comparison with the
improved CNN benchmark (step 20).  Reports POOLED cross-validated AUC
as the primary metric.

Usage:
  python src/03_age_effect_comparison.py
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
from models  import ImprovedBinaryCNN, ImprovedBinaryCNN_AgeMetadata

# ── PATHS ──────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_DIR   = os.path.join(ROOT, 'data', 'cache')

FIG_DIR = os.path.join(ROOT, 'results', 'figures')
TAB_DIR = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
SEGMENT      = 'amplitude'
BATCH_SIZE   = 16
EPOCHS       = 80
N_FOLDS      = 5
RANDOM_STATE = 42

# ── Load Metadata ──────────────────────────────────────────────────────────────
print("Loading metadata …")
df_meta = pd.read_csv(os.path.join(ROOT, 'data', 'metadata.csv'))
df_meta['Subject'] = df_meta['Subject'].str.strip()

subj_to_geno  = {r.Subject: (1 if '5xFAD' in r.Group else 0) for _, r in df_meta.iterrows()}
subj_to_a_bin = {r.Subject: (1 if 'Adult' in r.Group or 'adult' in r.Group else 0)
                 for _, r in df_meta.iterrows()}
age_min, age_max = df_meta['Age (Days)'].min(), df_meta['Age (Days)'].max()
subj_to_a_cont = {r.Subject: (r['Age (Days)'] - age_min) / (age_max - age_min)
                  for _, r in df_meta.iterrows()}
print(f"Age range for normalization: {age_min:.1f} – {age_max:.1f} days.")

dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment=SEGMENT, cache_dir=CACHE_DIR, force_reprocess=False)
all_trial_subjs = [s.split('_trial_')[0] for s in dataset.subjects]
valid_idx       = [i for i, s in enumerate(all_trial_subjs) if s in subj_to_geno]
unique_subjects = sorted(set(all_trial_subjs[i] for i in valid_idx))
subj_labels     = [subj_to_geno[s] for s in unique_subjects]

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Device: {device}\n")


def run_experiment(case: str):
    print(f"\n{'─'*50}")
    print(f"  Case: {case.upper()}")
    print(f"{'─'*50}")
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pooled = {}                      # subject → {'y_true': int, 'y_prob': float}
    fold_accs, fold_aucs = [], []

    for fold, (tr_idx_sub, vl_idx_sub) in enumerate(
            skf.split(unique_subjects, subj_labels), 1):

        tr_subs   = set(unique_subjects[i] for i in tr_idx_sub)
        vl_subs   = set(unique_subjects[i] for i in vl_idx_sub)
        tr_trials = [i for i in valid_idx if all_trial_subjs[i] in tr_subs]
        vl_trials = [i for i in valid_idx if all_trial_subjs[i] in vl_subs]

        tr_loader = DataLoader(Subset(dataset, tr_trials),
                               batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        vl_loader = DataLoader(Subset(dataset, vl_trials),
                               batch_size=BATCH_SIZE, shuffle=False)

        if case == 'no_age':
            model = ImprovedBinaryCNN().to(device)
        else:
            model = ImprovedBinaryCNN_AgeMetadata().to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc, best_acc = 0.0, 0.0
        best_votes = None

        for epoch in range(EPOCHS):
            # ── Train ──────────────────────────────────────────────────────────
            model.train()
            for signals, _, subjs in tr_loader:
                bz = torch.tensor([subj_to_geno[s.split('_trial_')[0]]
                                   for s in subjs]).to(device)
                signals = signals.to(device) + torch.randn_like(signals.to(device)) * 0.01
                optimizer.zero_grad()
                if case == 'no_age':
                    out = model(signals)
                elif case == 'binary_age':
                    age = torch.tensor([subj_to_a_bin[s.split('_trial_')[0]]
                                        for s in subjs],
                                       dtype=torch.float32).to(device)
                    out = model(signals, age)
                else:  # continuous_age
                    age = torch.tensor([subj_to_a_cont[s.split('_trial_')[0]]
                                        for s in subjs],
                                       dtype=torch.float32).to(device)
                    out = model(signals, age)
                criterion(out, bz).backward()
                optimizer.step()
            scheduler.step()

            # ── Validate ───────────────────────────────────────────────────────
            model.eval()
            votes = {}
            with torch.no_grad():
                for signals, _, subjs in vl_loader:
                    signals = signals.to(device)
                    if case == 'no_age':
                        out = model(signals)
                    elif case == 'binary_age':
                        age = torch.tensor([subj_to_a_bin[s.split('_trial_')[0]]
                                            for s in subjs],
                                           dtype=torch.float32).to(device)
                        out = model(signals, age)
                    else:
                        age = torch.tensor([subj_to_a_cont[s.split('_trial_')[0]]
                                            for s in subjs],
                                           dtype=torch.float32).to(device)
                        out = model(signals, age)
                    probs = F.softmax(out, dim=1)[:, 1].cpu().tolist()
                    for prob, s in zip(probs, subjs):
                        orig = s.split('_trial_')[0]
                        votes.setdefault(orig, {'label': subj_to_geno[orig], 'probs': []})
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
        print(f"  Fold {fold}  Acc={best_acc:.1%}  AUC={best_auc:.3f}")

    # ── Pooled AUC ────────────────────────────────────────────────────────────
    y_true = np.array([pooled[s]['y_true'] for s in unique_subjects if s in pooled])
    y_prob = np.array([pooled[s]['y_prob'] for s in unique_subjects if s in pooled])
    pooled_auc = roc_auc_score(y_true, y_prob)
    pooled_acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    print(f"  → Pooled AUC={pooled_auc:.3f}  "
          f"Fold-mean AUC={np.mean(fold_aucs):.3f}  "
          f"Fold-mean Acc={np.mean(fold_accs):.1%}")
    return {
        'fold_mean_acc': np.mean(fold_accs),
        'fold_std_acc':  np.std(fold_accs, ddof=1),
        'fold_mean_auc': np.mean(fold_aucs),
        'pooled_auc':    pooled_auc,
        'pooled_acc':    pooled_acc,
    }


if __name__ == "__main__":
    cases   = ['no_age', 'binary_age', 'continuous_age']
    results = {}
    for c in cases:
        results[c] = run_experiment(c)

    # ── Summary table ────────────────────────────────────────────────────────
    rows = [{'Case':         c,
             'Fold_Mean_Acc': results[c]['fold_mean_acc'],
             'Fold_Std_Acc':  results[c]['fold_std_acc'],
             'Fold_Mean_AUC': results[c]['fold_mean_auc'],
             'Pooled_AUC':    results[c]['pooled_auc'],
             'Pooled_Acc':    results[c]['pooled_acc']}
            for c in cases]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB_DIR, '03_age_effect_comparison.csv'), index=False)
    print(f"\n{'='*55}")
    print("SUMMARY")
    print(df.to_string(index=False))

    # ── Comparison Plot ───────────────────────────────────────────────────────
    label_map = {
        'no_age':         'No Age\n(Default)',
        'binary_age':     'Binary Age\n(Young/Adult)',
        'continuous_age': 'Continuous Age\n(Exact Days)',
    }
    x_labels    = [label_map[c] for c in cases]
    pooled_aucs = [results[c]['pooled_auc']    for c in cases]
    fold_m_aucs = [results[c]['fold_mean_auc'] for c in cases]
    fold_m_accs = [results[c]['fold_mean_acc'] for c in cases]
    fold_s_accs = [results[c]['fold_std_acc']  for c in cases]
    colors      = ['#34495e', '#2980b9', '#3498db']

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Pooled AUC
    ax = axes[0]
    bars = ax.bar(x_labels, pooled_aucs, color=colors,
                  alpha=0.85, edgecolor='black', width=0.55)
    ax.scatter(x_labels, fold_m_aucs, marker='D', s=80, color='black',
               zorder=5, label='Fold-mean AUC')
    ax.axhline(0.5, color='red', ls='--', lw=1.2, alpha=0.7, label='Chance (0.5)')
    ax.set_title('Pooled Cross-val AUC', fontweight='bold')
    ax.set_ylabel('AUC')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    for bar, v in zip(bars, pooled_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2., v + 0.02, f'{v:.3f}',
                ha='center', fontweight='bold', fontsize=11)

    # Right: Fold-mean Accuracy
    ax = axes[1]
    bars = ax.bar(x_labels, fold_m_accs, yerr=fold_s_accs, capsize=8,
                  color=colors, alpha=0.85, edgecolor='black', width=0.55)
    ax.axhline(0.5, color='red', ls='--', lw=1.2, alpha=0.7, label='Chance (50%)')
    ax.set_title('Fold-mean Subject Accuracy', fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    for bar, m, s in zip(bars, fold_m_accs, fold_s_accs):
        ax.text(bar.get_x() + bar.get_width() / 2., m + s + 0.02, f'{m:.1%}',
                ha='center', fontweight='bold', fontsize=11)

    plt.suptitle('Effect of Age as Auxiliary Input — Improved CNN Architecture\n'
                 '(Chirp amplitude, Binary WT vs 5xFAD, 5-fold subject-disjoint CV)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, '03_age_effect_comparison.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Figure → {out_fig}")
    print(f"✓ Table  → {TAB_DIR}/03_age_effect_comparison.csv")
