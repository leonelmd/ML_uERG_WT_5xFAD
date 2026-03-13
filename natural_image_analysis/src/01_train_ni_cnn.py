"""
01_train_ni_cnn.py
==================
Binary genotype classification (WT vs 5xFAD) using Natural Image MEA responses.

Protocol
--------
* Input             : 10 image repetitions × 2500 time points (SNR-filtered electrodes)
* Metadata          : None (Benchmark: No Age/Sex)
* Cross-validation  : 5-fold stratified, strictly subject-disjoint.
* Per-fold          : Best model by validation accuracy; probabilities extracted.
* Final evaluation  : Subject-level (one subject = one sample per fold).

Outputs
-------
  results/figures/01_ni_cnn_training_curves.png
  results/figures/01_ni_cnn_performance_bar.png
  results/tables/01_ni_cnn_probs.csv          ← per-subject probabilities for plotting
  results/tables/01_ni_cnn_fold_results.csv

Usage (from natural_image_analysis/ folder)
---------------------------------------------
    python src/01_train_ni_cnn.py

NOTE: Edit DATA_DIR and META_CSV below to point to your raw data.
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
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
from models import NaturalImageDataset, NaturalImageCNN_AgeMetadata, ImprovedNICNN_NoAge, parse_ni_metadata

# ── PATHS ──────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'natural_image_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_PATH  = os.path.join(ROOT, 'data', 'cache', 'ni_dataset.pt')

FIG_DIR = os.path.join(ROOT, 'results', 'figures')
TAB_DIR = os.path.join(ROOT, 'results', 'tables')
MOD_DIR = os.path.join(ROOT, 'results', 'models')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(MOD_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT, 'data', 'cache'), exist_ok=True)

# ── Params ─────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 16
EPOCHS       = 80
LR           = 5e-4
N_FOLDS      = 5
RANDOM_STATE = 42

# ── Configuration: metadata ────────────────────────────────────────────────────
USE_AGE      = False     # Set to True to include continuous Age and Sex metadata

# ── Load data ──────────────────────────────────────────────────────────────────
print("Parsing metadata …")
metadata = parse_ni_metadata(META_CSV)
print(f"Metadata: {len(metadata)} subjects")

print("Loading dataset …")
dataset = NaturalImageDataset(DATA_DIR, metadata, cache_path=CACHE_PATH)

subjects  = np.array(dataset.all_subjects)
all_labels = np.array([dataset.all_labels[i].item() for i in range(len(dataset))])
print(f"Dataset: {len(dataset)} subjects  WT={sum(all_labels==0)}  5xFAD={sum(all_labels==1)}\n")

# ── CV ─────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Device: {device}\n")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_metrics = []
cv_results   = {}   # subject → {y_true, y_prob}
all_tr_acc, all_vl_acc = [], []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(subjects, all_labels), 1):
    print(f"Fold {fold}/{N_FOLDS}  (train={len(tr_idx)}, val={len(vl_idx)})")
    tr_loader = DataLoader(Subset(dataset, tr_idx), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    vl_loader = DataLoader(Subset(dataset, vl_idx), batch_size=BATCH_SIZE, shuffle=False)

    if USE_AGE:
        model = NaturalImageCNN_AgeMetadata().to(device)
    else:
        model = ImprovedNICNN_NoAge().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = -1.0; best_auc = -1.0; best_state = None
    tr_hist, vl_hist = [], []

    for epoch in range(EPOCHS):
        model.train()
        t_preds, t_labels = [], []
        for x, m, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            
            # Augmentation
            if model.training:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
                
            optimizer.zero_grad()
            if USE_AGE:
                out = model(x, m.to(device))
            else:
                out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            t_preds.extend(out.argmax(1).cpu().tolist())
            t_labels.extend(y.cpu().tolist())
        
        scheduler.step()
        tr_hist.append(accuracy_score(t_labels, t_preds))

        model.eval()
        v_probs_all, v_labels_all = [], []
        with torch.no_grad():
            for x, m, y, _ in vl_loader:
                x = x.to(device)
                if USE_AGE:
                    out = model(x, m.to(device))
                else:
                    out = model(x)
                v_probs_all.extend(F.softmax(out, dim=1)[:, 1].cpu().tolist())
                v_labels_all.extend(y.tolist())
        
        v_preds = [1 if p >= 0.5 else 0 for p in v_probs_all]
        v_acc = accuracy_score(v_labels_all, v_preds)
        vl_hist.append(v_acc)
        v_auc_epoch = roc_auc_score(v_labels_all, v_probs_all) if len(set(v_labels_all)) > 1 else 0.5

        # Selection: Priority to AUC
        if (v_auc_epoch > best_auc) or (v_auc_epoch == best_auc and v_acc > best_acc):
            best_acc = v_acc
            best_auc = v_auc_epoch
            best_epoch_f = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    all_tr_acc.append(tr_hist)
    all_vl_acc.append(vl_hist)

    # Save best model for fold
    torch.save(best_state, os.path.join(MOD_DIR, f'01_ni_cnn_noage_fold_{fold}.pt'))

    # Extract probabilities with best model
    model.load_state_dict(best_state)
    with torch.no_grad():
        for x, m, y, subjs in vl_loader:
            if USE_AGE:
                out = model(x.to(device), m.to(device))
            else:
                out = model(x.to(device))
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            for s, p, yt in zip(subjs, probs, y.numpy()):
                cv_results[s] = {'y_true': int(yt), 'y_prob': float(p)}

    y_true_fold = [cv_results[subjects[i]]['y_true'] for i in vl_idx]
    y_prob_fold = [cv_results[subjects[i]]['y_prob'] for i in vl_idx]
    y_pred_fold = [(1 if p >= 0.5 else 0) for p in y_prob_fold]
    auc = roc_auc_score(y_true_fold, y_prob_fold) if len(set(y_true_fold)) > 1 else 0.5

    acc  = accuracy_score(y_true_fold, y_pred_fold)
    f1   = f1_score(y_true_fold, y_pred_fold, average='macro', zero_division=0)
    sens = recall_score(y_true_fold, y_pred_fold, pos_label=1, zero_division=0)
    fold_metrics.append({'acc': acc, 'f1': f1, 'sens': sens, 'auc': auc, 'epoch': best_epoch_f})
    print(f"  Fold {fold}  Acc={acc:.1%}  AUC={auc:.3f} (Best Epoch: {best_epoch_f})")

# ── Aggregate ──────────────────────────────────────────────────────────────────
def agg(key): return (np.mean([m[key] for m in fold_metrics]),
                      np.std( [m[key] for m in fold_metrics]))

print(f"\n{'='*50}")
print(f"NI CNN — {N_FOLDS}-Fold CV  (N={len(dataset)})")
for k, lbl in [('acc','Accuracy'), ('auc','AUC'), ('f1','F1'), ('sens','Sensitivity')]:
    m, s = agg(k)
    print(f"  {lbl:<15}: {m:.2%} ± {s:.2%}")

# Save probs CSV
prob_rows = [{'Subject': s, 'y_true': v['y_true'], 'y_prob': v['y_prob']}
             for s, v in cv_results.items()]
probs_df = pd.DataFrame(prob_rows)
probs_df.to_csv(os.path.join(TAB_DIR, '01_ni_cnn_noage_probs.csv'), index=False)
probs_df.to_csv(os.path.join(TAB_DIR, '07_improved_ni_cnn_probs.csv'), index=False)

fold_rows = [{'Fold': i+1, **m} for i, m in enumerate(fold_metrics)]
fold_rows.append({'Fold': 'Mean', **{k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}})
fold_rows.append({'Fold': 'Std',  **{k: np.std ([m[k] for m in fold_metrics]) for k in fold_metrics[0]}})
pd.DataFrame(fold_rows).to_csv(
    os.path.join(TAB_DIR, '01_ni_cnn_noage_fold_results.csv'), index=False)

# ── Figures ────────────────────────────────────────────────────────────────────
ep = np.arange(1, EPOCHS + 1)
m_tr = np.mean(all_tr_acc, axis=0)
m_vl = np.mean(all_vl_acc, axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
# Individual folds
for i in range(N_FOLDS):
    ax.plot(ep, all_tr_acc[i], color='#2980b9', lw=1.0, alpha=0.15)
    ax.plot(ep, all_vl_acc[i], color='#c0392b', lw=1.0, alpha=0.15, ls='--')
    
    best_ep = fold_metrics[i]['epoch']
    ax.scatter(best_ep, all_vl_acc[i][best_ep-1], color='#c0392b', s=25, zorder=5, edgecolors='white', alpha=0.6)

# Mean lines
ax.plot(ep, m_tr, '#2980b9', lw=2.5, label='Train Mean')
ax.plot(ep, m_vl, '#c0392b', lw=2.5, ls='--', label='Val Mean')

ax.axhline(0.5, color='gray', ls=':', alpha=0.6, label='Chance')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.set_title(f'Improved NI CNN — {N_FOLDS}-Fold CV Training Curves', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(1, EPOCHS)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '01_ni_cnn_training_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# Performance bar
fig, ax = plt.subplots(figsize=(9, 6))
names = ['Accuracy', 'AUC', 'F1-Score', 'Sensitivity']
vals  = [agg(k)[0] for k in ['acc','auc','f1','sens']]
errs  = [agg(k)[1] for k in ['acc','auc','f1','sens']]
colors_ = ['#2980b9', '#e67e22', '#8e44ad', '#27ae60']
bars = ax.bar(names, vals, yerr=errs, capsize=10, color=colors_,
              edgecolor='black', linewidth=1.5, alpha=0.85, width=0.5)
ax.axhline(0.5, color='gray', ls='--', lw=1.5, alpha=0.5, label='Chance')
ax.set_ylim(0, 1.15); ax.set_ylabel('Score'); ax.legend()
ax.set_title(f'Improved NI CNN — {N_FOLDS}-Fold CV Performance', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, v, e in zip(bars, vals, errs):
    ax.text(bar.get_x() + bar.get_width()/2., v/2,
            f'{v:.1%}\n±{e:.1%}', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '01_ni_cnn_performance_bar.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Figures → {FIG_DIR}")
print(f"✓ Tables  → {TAB_DIR}")
