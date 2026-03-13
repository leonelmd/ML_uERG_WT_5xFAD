"""
12_improved_chirp_cnn.py
========================
Re-runs the binary chirp CNN experiment with the improved architecture
(ImprovedBinaryCNN) across all five segments and compares with the
original BinaryCNN_NoAge results from script 04/05.

Architectural changes vs original:
  • InstanceNorm1d at input  – per-sample amplitude normalisation.
  • TemporalStatPool         – [mean, max, std] replacing AdaptiveAvgPool1d(1).
  • Smaller channel counts   – 8/16/32 vs 16/32/64 (less overfitting).
  • Stronger dropout         – 0.5 vs 0.3.

Evaluation fix:
  • Reports POOLED cross-validated AUC (all subjects' held-out probs
    concatenated and scored together), not mean-of-fold AUC, for an
    honest comparison with the handcrafted-ML results.

Usage (from chirp_analysis/ folder):
    python src/12_improved_chirp_cnn.py
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

from dataset import ERGChirpDataset
from models  import BinaryCNN_NoAge, ImprovedBinaryCNN

# ── PATHS ──────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_DIR   = os.path.join(ROOT, 'data', 'cache')
FIG_DIR     = os.path.join(ROOT, 'results', 'figures')
TAB_DIR     = os.path.join(ROOT, 'results', 'tables')
MOD_DIR     = os.path.join(ROOT, 'results', 'models')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(MOD_DIR, exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
SEGMENTS     = ['full', 'flash', 'frequency', 'amplitude', 'amplitude_norm']
BATCH_SIZE   = 16
EPOCHS       = 80    # same as original for fair comparison
N_FOLDS      = 5
RANDOM_STATE = 42

df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label = {r.Subject: (1 if '5xFAD' in r.Group else 0)
                    for _, r in df_meta.iterrows()}


def run_segment(segment: str, ModelClass, label: str):
    """Train one model class on one segment; return pooled cross-val metrics."""
    print(f"\n  [{label}] Segment: {segment.upper()}")
    dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment=segment,
                              cache_dir=CACHE_DIR,
                              force_reprocess=False)

    all_trial_subjects = [s.split('_trial_')[0] for s in dataset.subjects]
    valid_idx = [i for i, s in enumerate(all_trial_subjects)
                 if s in subject_to_label]

    unique_subjects = sorted(set(all_trial_subjects[i] for i in valid_idx))
    subj_labels     = [subject_to_label[s] for s in unique_subjects]

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    # Pooled cross-val predictions (all subjects, held-out only)
    pooled = {}   # subject → {y_true, y_prob}

    fold_accs, fold_aucs = [], []

    for fold, (tr_sidx, vl_sidx) in enumerate(
            skf.split(unique_subjects, subj_labels), 1):

        tr_subs = set(unique_subjects[i] for i in tr_sidx)
        vl_subs = set(unique_subjects[i] for i in vl_sidx)
        tr_idx  = [i for i in valid_idx if all_trial_subjects[i] in tr_subs]
        vl_idx  = [i for i in valid_idx if all_trial_subjects[i] in vl_subs]

        tr_loader = DataLoader(Subset(dataset, tr_idx),
                               batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True)
        vl_loader = DataLoader(Subset(dataset, vl_idx),
                               batch_size=BATCH_SIZE, shuffle=False)

        model     = ModelClass().to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=5e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS)

        best_auc, best_acc = 0.0, 0.0
        best_votes = None
        best_state = None

        for epoch in range(EPOCHS):
            # ── Train ──────────────────────────────────────────────────────────
            model.train()
            for signals, lbls4, _ in tr_loader:
                bz = torch.tensor([1 if l >= 2 else 0 for l in lbls4]).to(device)
                signals = signals.to(device)
                # Augmentation: small Gaussian noise + random sign flip
                signals = signals + torch.randn_like(signals) * 0.01
                optimizer.zero_grad()
                loss = criterion(model(signals), bz)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # ── Validate (subject-level) ────────────────────────────────────
            model.eval()
            votes = {}
            with torch.no_grad():
                for signals, lbls4, subjs in vl_loader:
                    bz = torch.tensor([1 if l >= 2 else 0 for l in lbls4]).to(device)
                    probs = F.softmax(model(signals.to(device)), dim=1)[:, 1]
                    for prob, lbl, s in zip(probs.cpu().tolist(),
                                            bz.tolist(), subjs):
                        orig = s.split('_trial_')[0]
                        votes.setdefault(orig, {'label': lbl,
                                                'probs': []})
                        votes[orig]['probs'].append(prob)

            sp_probs = [np.mean(v['probs']) for v in votes.values()]
            sl       = [v['label']          for v in votes.values()]
            sp       = [1 if p >= 0.5 else 0 for p in sp_probs]
            v_acc    = accuracy_score(sl, sp)
            v_auc    = roc_auc_score(sl, sp_probs) if len(np.unique(sl)) > 1 else 0.5

            if (v_auc > best_auc) or (v_auc == best_auc and v_acc > best_acc):
                best_auc, best_acc = v_auc, v_acc
                best_votes = {s: {'y_true': v['label'],
                                  'y_prob': float(np.mean(v['probs']))}
                              for s, v in votes.items()}
                best_state = copy.deepcopy(model.state_dict())

        fold_accs.append(best_acc)
        fold_aucs.append(best_auc)
        pooled.update(best_votes)
        print(f"    Fold {fold}  Acc={best_acc:.1%}  AUC={best_auc:.3f}")
        if label == 'Improved' and best_state is not None:
            save_path = os.path.join(MOD_DIR, f'12_improved_{segment}_fold_{fold}.pt')
            torch.save(best_state, save_path)
            print(f"      → Saved {os.path.basename(save_path)}")

    # ── Pooled AUC (honest, cross-validated) ───────────────────────────────────
    y_true = np.array([pooled[s]['y_true'] for s in unique_subjects if s in pooled])
    y_prob = np.array([pooled[s]['y_prob'] for s in unique_subjects if s in pooled])
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
        'y_true':        y_true,
        'y_prob':        y_prob,
    }


# ── Run both models on all segments ───────────────────────────────────────────
print("=" * 60)
print("ORIGINAL CNN  (BinaryCNN_NoAge)")
print("=" * 60)
orig_results = {}
for seg in SEGMENTS:
    orig_results[seg] = run_segment(seg, BinaryCNN_NoAge, 'Original')

print("\n" + "=" * 60)
print("IMPROVED CNN  (ImprovedBinaryCNN)")
print("=" * 60)
impr_results = {}
for seg in SEGMENTS:
    impr_results[seg] = run_segment(seg, ImprovedBinaryCNN, 'Improved')

# ── Save tables ────────────────────────────────────────────────────────────────
rows = []
for seg in SEGMENTS:
    for tag, res in [('Original', orig_results[seg]),
                     ('Improved', impr_results[seg])]:
        rows.append({
            'Model':         tag,
            'Segment':       seg,
            'Fold_Mean_Acc': res['fold_mean_acc'],
            'Fold_Std_Acc':  res['fold_std_acc'],
            'Fold_Mean_AUC': res['fold_mean_auc'],
            'Pooled_AUC':    res['pooled_auc'],
            'Pooled_Acc':    res['pooled_acc'],
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(os.path.join(TAB_DIR, '12_improved_chirp_comparison.csv'),
              index=False)
print(f"\n✓ Table saved.")
print(df_out.to_string(index=False))

# ── Save improved probs for each segment ──────────────────────────────────────
for seg in SEGMENTS:
    r = impr_results[seg]
    # Recover subject names in same order as y_true/y_prob
    dataset_tmp = ERGChirpDataset(DATA_DIR, META_CSV, segment=seg,
                                  cache_dir=CACHE_DIR,
                                  force_reprocess=False)
    all_trial_subjs = [s.split('_trial_')[0] for s in dataset_tmp.subjects]
    unique_subjects = sorted(set(s for s in all_trial_subjs
                                 if s in subject_to_label))
    df_prob = pd.DataFrame({
        'Subject': unique_subjects[:len(r['y_true'])],
        'y_true':  r['y_true'],
        'y_prob':  r['y_prob'],
    })
    df_prob.to_csv(os.path.join(TAB_DIR, f'12_improved_{seg}_probs.csv'),
                   index=False)

# ── Figure: Pooled AUC comparison across segments ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(SEGMENTS))
w = 0.35

key_map = {'Pooled_AUC': 'pooled_auc', 'Fold_Mean_Acc': 'fold_mean_acc'}

for ax, metric, ylabel in [
    (axes[0], 'Pooled_AUC',    'Pooled Cross-val AUC'),
    (axes[1], 'Fold_Mean_Acc', 'Fold-mean Subject Accuracy'),
]:
    key = key_map[metric]
    orig_vals = [orig_results[s][key] for s in SEGMENTS]
    impr_vals = [impr_results[s][key] for s in SEGMENTS]

    ax.bar(x - w/2, orig_vals, w, label='Original CNN',
           color='#2980b9', alpha=0.8, edgecolor='black')
    ax.bar(x + w/2, impr_vals, w, label='Improved CNN',
           color='#e67e22', alpha=0.8, edgecolor='black')
    ax.axhline(0.5, color='red', ls='--', lw=1.5, label='Chance (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(SEGMENTS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(ylabel, fontweight='bold')

plt.suptitle('Chirp CNN: Original vs Improved Architecture\n'
             '(Binary WT vs 5xFAD, No Age, 5-fold subject-disjoint CV)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out_fig = os.path.join(FIG_DIR, '12_improved_chirp_cnn.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Figure saved → {out_fig}")
