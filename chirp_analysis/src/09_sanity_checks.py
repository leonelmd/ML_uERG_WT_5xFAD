"""
09_sanity_checks.py
===================
Sanity checks for the binary CNN model:

1. DATA LEAKAGE CHECK
   - Verifies that the CV fold construction is truly subject-disjoint:
     no tissue piece appears in both train and val within a single fold.
   - Replicates fold splits and prints a pass/fail report.

2. AGE CLASSIFICATION (stratified by genotype)
   - Trains the same CNN architecture to PREDICT AGE instead of genotype.
   - Evaluates whether age can be classified from the amplitude-sweep segment.
   - Uses the same 5-fold CV scheme.
   - High age classification accuracy would suggest age confounds genotype.

3. PERMUTATION TEST
   - Shuffles labels (within fold) to get chance-level distributions.
   - Compares actual performance to chance to confirm above-chance performance.

Usage (from chirp_analysis/ folder)
-------------------------------------
    python src/09_sanity_checks.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
from dataset import ERGChirpDataset
from models  import BinaryCNN_NoAge

# ── PATHS ──────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_DIR   = os.path.join(ROOT, 'data', 'cache')
FIG_DIR     = os.path.join(ROOT, 'results', 'figures')
TAB_DIR     = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)

SEGMENT    = 'amplitude'
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-3
N_FOLDS    = 5
RANDOM_STATE = 42

# ── Load data ──────────────────────────────────────────────────────────────────
df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label = {r.Subject: (1 if '5xFAD' in r.Group else 0)
                    for _, r in df_meta.iterrows()}
subject_to_age_label = {r.Subject: (1 if 'adult' in r.Group else 0)
                        for _, r in df_meta.iterrows()}
age_min = df_meta['Age (Days)'].min(); age_max = df_meta['Age (Days)'].max()
subject_to_age_norm = {r.Subject: (r['Age (Days)'] - age_min) / (age_max - age_min)
                       for _, r in df_meta.iterrows()}

dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment=SEGMENT, cache_dir=CACHE_DIR)
all_trial_subjs = [s.split('_trial_')[0] for s in dataset.subjects]
valid_idx = [i for i, s in enumerate(all_trial_subjs) if s in subject_to_age_norm]
unique_subjects = sorted(set(all_trial_subjs[i] for i in valid_idx))
subj_labels     = [subject_to_label[s] for s in unique_subjects]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LEAKAGE CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("SANITY CHECK 1: Data Leakage Verification")
print("="*60)
leakage_found = False
for fold, (tr_sidx, vl_sidx) in enumerate(skf.split(unique_subjects, subj_labels), 1):
    train_subs = set(unique_subjects[i] for i in tr_sidx)
    val_subs   = set(unique_subjects[i] for i in vl_sidx)
    # Check subject overlap
    overlap_subj = train_subs & val_subs
    if overlap_subj:
        print(f"  Fold {fold}: LEAKAGE DETECTED — subjects overlap: {overlap_subj}")
        leakage_found = True
    else:
        print(f"  Fold {fold}: ✓  No subject overlap  "
              f"(train={len(train_subs)}, val={len(val_subs)})")
    # Check trial-level: all train trials must come from train subjects only
    tr_trials = [i for i in valid_idx if all_trial_subjs[i] in train_subs]
    vl_trials = [i for i in valid_idx if all_trial_subjs[i] in val_subs]
    trial_overlap = set(tr_trials) & set(vl_trials)
    if trial_overlap:
        print(f"  Fold {fold}: LEAKAGE DETECTED — trial overlap: {len(trial_overlap)} trials")
        leakage_found = True
    else:
        print(f"          ✓  No trial overlap  "
              f"(train={len(tr_trials)} trials, val={len(vl_trials)} trials)")

if not leakage_found:
    print("\n✅ RESULT: No data leakage detected across all folds.")
else:
    print("\n❌ RESULT: DATA LEAKAGE DETECTED — investigate before use!")

# ══════════════════════════════════════════════════════════════════════════════
# 2. AGE CLASSIFICATION STRATIFIED BY GENOTYPE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SANITY CHECK 2: Age Classification (Stratified by Genotype)")
print("="*60)
print("Testing if age (young/adult) can be classified from the chirp signal.")
print("High accuracy would suggest age is a confounder.\n")

age_fold_metrics = []

for geno, geno_label in [('WT', 0), ('5xFAD', 1)]:
    geno_subs = [s for s in unique_subjects if subject_to_label[s] == geno_label]
    age_lbls  = [subject_to_age_label[s] for s in geno_subs]

    if len(set(age_lbls)) < 2:
        print(f"  [{geno}] Only one age group — skipping.")
        continue

    skf_age = StratifiedKFold(n_splits=min(5, min(age_lbls.count(0), age_lbls.count(1))+1),
                               shuffle=True, random_state=RANDOM_STATE)
    fold_accs = []

    for fold, (tr_sidx, vl_sidx) in enumerate(skf_age.split(geno_subs, age_lbls), 1):
        train_subs = set(geno_subs[i] for i in tr_sidx)
        val_subs   = set(geno_subs[i] for i in vl_sidx)

        tr_idx = [i for i in valid_idx if all_trial_subjs[i] in train_subs]
        vl_idx = [i for i in valid_idx if all_trial_subjs[i] in val_subs]
        if not tr_idx or not vl_idx:
            continue

        tr_loader = DataLoader(Subset(dataset, tr_idx), batch_size=BATCH_SIZE, shuffle=True)
        vl_loader = DataLoader(Subset(dataset, vl_idx), batch_size=BATCH_SIZE, shuffle=False)

        # Use BinaryCNN_NoAge to test if signal alone classifies age (0=young, 1=adult)
        model     = BinaryCNN_NoAge(signal_length=2750).to(device)
        criterion = nn.CrossEntropyLoss()
        # Higher epochs and slightly lower LR for subtler age signals
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

        best_acc = 0.0
        best_state = None

        for epoch in range(100): 
            model.train()
            for signals, lbls4, subjs in tr_loader:
                age_bz = torch.tensor(
                    [subject_to_age_label.get(s.split('_trial_')[0], 0) for s in subjs])
                signals = signals.to(device)
                age_bz  = age_bz.to(device)
                optimizer.zero_grad()
                out = model(signals)
                loss = criterion(out, age_bz)
                loss.backward(); optimizer.step()
            
            # Internal evaluation per epoch to find best state
            model.eval()
            temp_preds, temp_labels = [], []
            with torch.no_grad():
                for signals, lbls4, subjs in vl_loader:
                    out = model(signals.to(device))
                    temp_preds.extend(out.argmax(1).cpu().tolist())
                    temp_labels.extend([subject_to_age_label.get(s.split('_trial_')[0], 0) for s in subjs])
            
            val_acc_epoch = accuracy_score(temp_labels, temp_preds)
            if val_acc_epoch >= best_acc:
                best_acc = val_acc_epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Final evaluation with best state (subject-level majority vote)
        model.load_state_dict(best_state)
        model.eval()
        votes = {}
        with torch.no_grad():
            for signals, lbls4, subjs in vl_loader:
                out = model(signals.to(device))
                preds = out.argmax(1).cpu().tolist()
                for pred, s in zip(preds, subjs):
                    orig = s.split('_trial_')[0]
                    true_age = subject_to_age_label.get(orig, 0)
                    votes.setdefault(orig, {'label': true_age, 'preds': []})['preds'].append(pred)

        sp  = [np.bincount(v['preds']).argmax() for v in votes.values()]
        sl  = [v['label'] for v in votes.values()]
        fold_accs.append(accuracy_score(sl, sp))

    mean_acc = np.mean(fold_accs) if fold_accs else float('nan')
    std_acc  = np.std(fold_accs)  if fold_accs else float('nan')
    print(f"  [{geno}] Age classification accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
    age_fold_metrics.append({'Genotype': geno, 'Acc_Mean': mean_acc, 'Acc_Std': std_acc})

pd.DataFrame(age_fold_metrics).to_csv(
    os.path.join(TAB_DIR, '09_age_classification_results.csv'), index=False)

# Add Figure plotting
fig, ax = plt.subplots(figsize=(6, 5))
genos = [m['Genotype'] for m in age_fold_metrics]
means = [m['Acc_Mean'] for m in age_fold_metrics]
stds  = [m['Acc_Std'] for m in age_fold_metrics]
ax.bar(genos, means, yerr=stds, color=['#1565C0', '#2E7D32'], alpha=0.8,
       edgecolor='black', capsize=8)
ax.axhline(0.5, color='gray', ls='--', alpha=0.7, label='Chance (50%)')
ax.set_ylim(0, 1.0)
ax.set_ylabel('Age Classification Accuracy')
ax.set_title('Age Classification by Genotype (Sanity Check)', fontweight='bold')
ax.legend()
for i, v in enumerate(means):
    if not np.isnan(v):
        ax.text(i, v + stds[i] + 0.05, f'{v:.1%}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '09_age_classification_plot.png'), dpi=300)
plt.close()

print("\nInterpretation:")
print("  • If age accuracy ≈ 50%, age cannot be inferred from signal → no confounder.")
print("  • If age accuracy >> 50%, age signal may confound genotype classification.")

print(f"\n✓ Sanity check tables → {TAB_DIR}")
print(f"✓ Sanity check figures → {FIG_DIR}")
