"""
07_improved_ni_cnn.py
=====================
Re-runs the natural image CNN binary classification experiment with the
improved architecture (ImprovedNICNN_NoAge) and compares against the
original NaturalImageCNN_NoAge from script 01.

Architectural changes (same rationale as 12_improved_chirp_cnn.py):
  • InstanceNorm1d at input  – per-sample normalisation.
  • TemporalStatPool         – [mean, max, std] replacing AdaptiveAvgPool1d(1).
  • Smaller channel counts   – 8/16/32 vs 16/32/64.
  • Stronger dropout         – 0.5 vs 0.3.

Evaluation:
  • Reports POOLED cross-validated AUC for honest comparison.

Usage (from natural_image_analysis/ folder):
    python src/07_improved_ni_cnn.py
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

from models import (NaturalImageDataset, NaturalImageCNN_NoAge,
                    ImprovedNICNN_NoAge, parse_ni_metadata)

# ── PATHS ──────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'natural_image_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_PATH  = os.path.join(ROOT, 'data', 'cache', 'ni_dataset.pt')

FIG_DIR = os.path.join(ROOT, 'results', 'figures')
TAB_DIR = os.path.join(ROOT, 'results', 'tables')
MOD_DIR = os.path.join(ROOT, 'results', 'models')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(MOD_DIR, exist_ok=True)

# ── Params ─────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 8
EPOCHS       = 80    # same as original for fair comparison
N_FOLDS      = 5
RANDOM_STATE = 42

# ── Load data ──────────────────────────────────────────────────────────────────
print("Parsing metadata …")
metadata = parse_ni_metadata(META_CSV)
print(f"  {len(metadata)} subjects in metadata")

print("Loading dataset …")
dataset = NaturalImageDataset(DATA_DIR, metadata,
                              cache_path=CACHE_PATH)

subjects   = np.array(dataset.all_subjects)
all_labels = np.array([dataset.all_labels[i].item() for i in range(len(dataset))])
print(f"  Dataset: {len(dataset)} subjects  "
      f"WT={sum(all_labels==0)}  5xFAD={sum(all_labels==1)}\n")


def run_model(ModelClass, tag):
    """5-fold CV for one model class; returns pooled AUC and per-fold stats."""
    print(f"\n{'─'*50}")
    print(f"  Model: {tag}")
    print(f"{'─'*50}")

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

    cv_results = {}   # subject → {y_true, y_prob}
    fold_metrics = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(subjects, all_labels), 1):
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

        best_auc, best_acc, best_epoch = 0.0, 0.0, 0
        best_state = None

        for epoch in range(EPOCHS):
            model.train()
            for x, m, y, _ in tr_loader:
                x, y = x.to(device), y.to(device)
                x = x + torch.randn_like(x) * 0.01   # Gaussian noise augmentation
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            v_probs, v_labels = [], []
            with torch.no_grad():
                for x, m, y, _ in vl_loader:
                    probs = F.softmax(model(x.to(device)), dim=1)[:, 1]
                    v_probs.extend(probs.cpu().tolist())
                    v_labels.extend(y.tolist())

            v_preds = [1 if p >= 0.5 else 0 for p in v_probs]
            v_acc   = accuracy_score(v_labels, v_preds)
            v_auc   = (roc_auc_score(v_labels, v_probs)
                       if len(set(v_labels)) > 1 else 0.5)

            if (v_auc > best_auc) or (v_auc == best_auc and v_acc > best_acc):
                best_auc, best_acc, best_epoch = v_auc, v_acc, epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Extract probabilities with best model
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for x, m, y, subjs in vl_loader:
                probs = F.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
                for s, p, yt in zip(subjs, probs, y.numpy()):
                    cv_results[s] = {'y_true': int(yt), 'y_prob': float(p)}

        y_true_f = [cv_results[subjects[i]]['y_true'] for i in vl_idx]
        y_prob_f = [cv_results[subjects[i]]['y_prob'] for i in vl_idx]
        fold_auc = (roc_auc_score(y_true_f, y_prob_f)
                    if len(set(y_true_f)) > 1 else 0.5)
        fold_acc = accuracy_score(y_true_f,
                                  [1 if p >= 0.5 else 0 for p in y_prob_f])
        fold_metrics.append({'acc': fold_acc, 'auc': fold_auc,
                             'epoch': best_epoch})
        
        # Save model for future introspection (CCA etc)
        mod_name = 'improved' if 'Improved' in tag else 'original'
        torch.save(best_state, os.path.join(MOD_DIR, f'07_{mod_name}_ni_cnn_fold_{fold}.pt'))
        
        print(f"  Fold {fold}  Acc={fold_acc:.1%}  AUC={fold_auc:.3f}"
              f"  (best epoch {best_epoch})")

    # ── Pooled AUC ──────────────────────────────────────────────────────────
    y_true_all = np.array([cv_results[s]['y_true'] for s in subjects])
    y_prob_all = np.array([cv_results[s]['y_prob'] for s in subjects])
    pooled_auc = roc_auc_score(y_true_all, y_prob_all)
    pooled_acc = accuracy_score(y_true_all, (y_prob_all >= 0.5).astype(int))
    fold_mean_auc = np.mean([m['auc'] for m in fold_metrics])
    fold_mean_acc = np.mean([m['acc'] for m in fold_metrics])

    print(f"\n  ✓ {tag}")
    print(f"    Pooled AUC  = {pooled_auc:.3f}  "
          f"(fold-mean = {fold_mean_auc:.3f})")
    print(f"    Pooled Acc  = {pooled_acc:.1%}  "
          f"(fold-mean = {fold_mean_acc:.1%})")

    # Save probs
    prob_fname = '07_improved_ni_cnn_probs.csv' if 'Improved' in tag \
                 else '07_original_ni_cnn_probs.csv'
    pd.DataFrame([{'Subject': s, **cv_results[s]} for s in subjects]).to_csv(
        os.path.join(TAB_DIR, prob_fname), index=False)

    return {
        'fold_metrics':  fold_metrics,
        'pooled_auc':    pooled_auc,
        'pooled_acc':    pooled_acc,
        'fold_mean_auc': fold_mean_auc,
        'fold_mean_acc': fold_mean_acc,
        'y_true': y_true_all,
        'y_prob': y_prob_all,
    }


# ── Run both models ────────────────────────────────────────────────────────────
res_orig = run_model(NaturalImageCNN_NoAge,   'Original NI CNN')
res_impr = run_model(ImprovedNICNN_NoAge,     'Improved NI CNN')

# ── Summary table ──────────────────────────────────────────────────────────────
summary = pd.DataFrame([
    {'Model': 'Original NI CNN',
     'Fold_Mean_AUC': res_orig['fold_mean_auc'],
     'Pooled_AUC':    res_orig['pooled_auc'],
     'Fold_Mean_Acc': res_orig['fold_mean_acc'],
     'Pooled_Acc':    res_orig['pooled_acc']},
    {'Model': 'Improved NI CNN',
     'Fold_Mean_AUC': res_impr['fold_mean_auc'],
     'Pooled_AUC':    res_impr['pooled_auc'],
     'Fold_Mean_Acc': res_impr['fold_mean_acc'],
     'Pooled_Acc':    res_impr['pooled_acc']},
])
summary.to_csv(os.path.join(TAB_DIR, '07_ni_cnn_comparison.csv'), index=False)
print(f"\n{'='*55}")
print("SUMMARY")
print(summary.to_string(index=False))

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

labels    = ['Original CNN', 'Improved CNN']
colors    = ['#2980b9', '#e67e22']
fold_aucs = [[m['auc'] for m in res_orig['fold_metrics']],
             [m['auc'] for m in res_impr['fold_metrics']]]
fold_accs = [[m['acc'] for m in res_orig['fold_metrics']],
             [m['acc'] for m in res_impr['fold_metrics']]]

for ax, fold_data, pooled_vals, ylabel in [
    (axes[0], fold_aucs, [res_orig['pooled_auc'], res_impr['pooled_auc']], 'AUC'),
    (axes[1], fold_accs, [res_orig['pooled_acc'], res_impr['pooled_acc']], 'Accuracy'),
]:
    for xi, (vals, col, lbl) in enumerate(zip(fold_data, colors, labels)):
        ax.bar(xi, np.mean(vals), color=col, alpha=0.5,
               edgecolor='black', label=f'{lbl} fold-mean')
        ax.errorbar(xi, np.mean(vals), yerr=np.std(vals, ddof=1),
                    fmt='none', color='black', capsize=6)
        # Pooled value as star
        ax.scatter(xi, pooled_vals[xi], marker='*', s=200, color=col,
                   zorder=5, label=f'{lbl} pooled')

    ax.axhline(0.5, color='red', ls='--', lw=1.2, alpha=0.7, label='Chance')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.1)
    ax.set_title(f'NI CNN {ylabel}: bar=fold-mean, ★=pooled', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Natural Image CNN: Original vs Improved Architecture\n'
             '(Binary WT vs 5xFAD, No Age, 5-fold subject-disjoint CV)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
out_fig = os.path.join(FIG_DIR, '07_improved_ni_cnn.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Figure → {out_fig}")
