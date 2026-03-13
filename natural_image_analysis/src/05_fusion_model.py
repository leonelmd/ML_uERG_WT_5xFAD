"""
05_fusion_model.py
==================
Dual-Input CNN: Fusion of Chirp Amplitude Sweep + Natural Image ERG.

This is the capstone model that leverages complementary information from
both stimuli to classify genotype (WT vs 5xFAD).

Protocol
--------
* Inputs:
    - Chirp amplitude-sweep segment (6000–8750 samples, flash-normalized)
    - NI signal (10 repetitions × 2500 samples)
    - Metadata: age_norm, sex_binary
* Two independent CNN branches (chirp + NI), fused before the classifier head.
* Cross-validation: 5-fold, restricted to subjects with BOTH stimuli.
  Strictly subject-disjoint (no leakage).

Outputs
-------
  results/figures/05_fusion_training_curves.png
  results/figures/05_fusion_performance_bar.png
  results/tables/05_fusion_probs.csv
  results/tables/05_fusion_fold_results.csv

Usage (from natural_image_analysis/ folder)
---------------------------------------------
    python src/05_fusion_model.py

NOTE: Edit DATA_DIR_CHIRP / DATA_DIR_NI / META_CSV to match your setup.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
from models import parse_ni_metadata, NaturalImageDataset, DualInputCNN_NoAge, DualInputCNN_AgeMetadata

CHIRP_QA_DIR = os.path.abspath(os.path.join(ROOT, '..', 'chirp_analysis', 'src'))
sys.path.insert(0, CHIRP_QA_DIR)
from dataset import ERGChirpDataset

# ── PATHS ──────────────────────────────────────────────────────────────────────
RETINA_ROOT   = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR_NI   = os.path.join(RETINA_ROOT, 'natural_image_analysis', 'processed_data')
DATA_DIR_CHIR = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
META_CSV      = os.path.abspath(os.path.join(ROOT, '..', 'chirp_analysis', 'data', 'metadata.csv'))

CACHE_NI   = os.path.join(ROOT, 'data', 'cache', 'ni_dataset.pt')
CACHE_CHIR = os.path.join(ROOT, 'data', 'cache', 'chirp_amplitude.pt')
         
FIG_DIR = os.path.join(ROOT, 'results', 'figures')
TAB_DIR = os.path.join(ROOT, 'results', 'tables')
MOD_DIR = os.path.join(ROOT, 'results', 'models')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(MOD_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT, 'data', 'cache'), exist_ok=True)

# ── Params ─────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 8
EPOCHS       = 100
LR           = 5e-4
N_FOLDS      = 5
RANDOM_STATE = 42
NI_SIGNAL_LEN   = 2500
NI_CHANNELS      = 10
CHIRP_SIGNAL_LEN = 2750

# ── Configuration: metadata ────────────────────────────────────────────────────
USE_AGE      = False     # Set to True to include continuous Age and Sex metadata


# ── Dual-stimulus dataset ──────────────────────────────────────────────────────
class DualStimulusDataset(Dataset):
    """
    Aligns NI and Chirp datasets by subject.
    Returns: (ni_x, chirp_x, meta, label, subject)
    """
    def __init__(self, ni_dataset, chirp_dataset, metadata):
        self.ni_idx    = {s: i for i, s in enumerate(ni_dataset.all_subjects)}
        self.chirp_idx = {}   # subject → list of trial-level chirp indices (we average)
        for i, s in enumerate(chirp_dataset.orig_subjects):
            self.chirp_idx.setdefault(s, []).append(i)

        # Only keep subjects present in both datasets
        common = sorted(set(self.ni_idx) & set(self.chirp_idx) & set(metadata))
        self.subjects = common
        self.ni_ds    = ni_dataset
        self.chirp_ds = chirp_dataset
        self.metadata = metadata
        print(f"  [DualDataset] {len(common)} subjects with both stimuli.")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sub  = self.subjects[idx]
        ni_x = self.ni_ds.all_data[self.ni_idx[sub]]           # (10, 2500)
        # Average chirp trials for this subject
        chirp_trials = self.chirp_ds.signals
        c_idx = self.chirp_idx[sub]
        chirp_x = torch.stack([chirp_trials[i] for i in c_idx]).mean(0)  # (1, 2750)
        meta = self.metadata[sub]
        m    = torch.tensor([meta['age_norm'], meta['sex_bin']], dtype=torch.float32)
        lbl  = torch.tensor(meta['label'], dtype=torch.long)
        return ni_x, chirp_x, m, lbl, sub


# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading metadata …")
metadata  = parse_ni_metadata(META_CSV)
df_meta   = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
# Merge exact age (days) normalization into metadata
age_min, age_max = df_meta['Age (Days)'].min(), df_meta['Age (Days)'].max()
for _, r in df_meta.iterrows():
    if r.Subject in metadata:
        metadata[r.Subject]['age_norm'] = (r['Age (Days)'] - age_min) / (age_max - age_min)

print("Loading NI dataset …")
ni_ds = NaturalImageDataset(DATA_DIR_NI, metadata, cache_path=CACHE_NI)

print("Loading Chirp dataset …")
chirp_ds = ERGChirpDataset(DATA_DIR_CHIR, META_CSV, segment='amplitude', cache_dir=os.path.dirname(CACHE_CHIR))

print("Building dual dataset …")
dual_ds = DualStimulusDataset(ni_ds, chirp_ds, metadata)

subjects   = np.array(dual_ds.subjects)
all_labels = np.array([metadata[s]['label'] for s in subjects])
print(f"Dual dataset: {len(dual_ds)} subjects  WT={sum(all_labels==0)}  5xFAD={sum(all_labels==1)}\n")

# ── 5-fold CV ──────────────────────────────────────────────────────────────────
torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_metrics, cv_results = [], {}
all_tr_acc, all_vl_acc   = [], []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(subjects, all_labels), 1):
    print(f"Fold {fold}/{N_FOLDS}  (train={len(tr_idx)}, val={len(vl_idx)})")
    tr_loader = DataLoader(Subset(dual_ds, tr_idx), batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(Subset(dual_ds, vl_idx), batch_size=BATCH_SIZE, shuffle=False)

    if USE_AGE:
        model = DualInputCNN_AgeMetadata().to(device)
    else:
        model = DualInputCNN_NoAge().to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc = -1.0; best_state = None
    tr_hist, vl_hist = [], []

    for epoch in range(EPOCHS):
        model.train()
        t_preds, t_labels = [], []
        for ni_x, c_x, m, y, _ in tr_loader:
            ni_x,c_x,m,y = ni_x.to(device),c_x.to(device),m.to(device),y.to(device)
            optimizer.zero_grad()
            if USE_AGE:
                out = model(ni_x, c_x, m)
            else:
                out = model(ni_x, c_x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            t_preds.extend(out.argmax(1).cpu().tolist())
            t_labels.extend(y.cpu().tolist())
        tr_hist.append(accuracy_score(t_labels, t_preds))

        model.eval()
        v_preds, v_labels = [], []
        with torch.no_grad():
            for ni_x, c_x, m, y, _ in vl_loader:
                if USE_AGE:
                    out = model(ni_x.to(device), c_x.to(device), m.to(device))
                else:
                    out = model(ni_x.to(device), c_x.to(device))
                v_preds.extend(out.argmax(1).cpu().tolist())
                v_labels.extend(y.tolist())
        v_acc = accuracy_score(v_labels, v_preds)
        vl_hist.append(v_acc)
        
        # Subject-level AUC for selection
        with torch.no_grad():
            v_probs_all, v_labels_all = [], []
            for ni_x, c_x, m, y, _ in vl_loader:
                if USE_AGE:
                    out = model(ni_x.to(device), c_x.to(device), m.to(device))
                else:
                    out = model(ni_x.to(device), c_x.to(device))
                v_probs_all.extend(F.softmax(out, dim=1)[:, 1].cpu().tolist())
                v_labels_all.extend(y.tolist())
            v_auc_epoch = roc_auc_score(v_labels_all, v_probs_all) if len(set(v_labels_all)) > 1 else 0.5

        if (v_acc > best_acc) or (v_acc == best_acc and v_auc_epoch > best_auc):
            best_acc  = v_acc
            best_auc  = v_auc_epoch
            best_epoch_f = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    all_tr_acc.append(tr_hist); all_vl_acc.append(vl_hist)

    # Save best model for fold
    torch.save(best_state, os.path.join(MOD_DIR, f'05_fusion_noage_fold_{fold}.pt'))

    # Extract probs with best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        for ni_x, c_x, m, y, subjs in vl_loader:
            if USE_AGE:
                out = model(ni_x.to(device), c_x.to(device), m.to(device))
            else:
                out = model(ni_x.to(device), c_x.to(device))
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            for s, p, yt in zip(subjs, probs, y.numpy()):
                cv_results[s] = {'y_true': int(yt), 'y_prob': float(p)}

    y_true_f = [cv_results[subjects[i]]['y_true'] for i in vl_idx]
    y_prob_f = [cv_results[subjects[i]]['y_prob'] for i in vl_idx]
    y_pred_f = [(1 if p>=0.5 else 0) for p in y_prob_f]
    acc  = accuracy_score(y_true_f, y_pred_f)
    f1   = f1_score(y_true_f, y_pred_f, average='macro', zero_division=0)
    sens = recall_score(y_true_f, y_pred_f, pos_label=1, zero_division=0)
    auc  = roc_auc_score(y_true_f, y_prob_f) if len(set(y_true_f)) > 1 else 0.5
    fold_metrics.append({'acc': acc, 'f1': f1, 'sens': sens, 'auc': auc, 'epoch': best_epoch_f})
    print(f"  Fold {fold}  Acc={acc:.1%}  AUC={auc:.3f} (Best Epoch: {best_epoch_f})")

# ── Aggregate ──────────────────────────────────────────────────────────────────
def agg(k): return np.mean([m[k] for m in fold_metrics]), np.std([m[k] for m in fold_metrics])

print(f"\n{'='*50}")
print(f"DUAL CNN Fusion — {N_FOLDS}-Fold CV  (N={len(dual_ds)})")
for k,lbl in [('acc','Accuracy'),('auc','AUC'),('f1','F1'),('sens','Sensitivity')]:
    m,s = agg(k)
    print(f"  {lbl:<15}: {m:.2%} ± {s:.2%}")

# Save
prob_rows = [{'Subject':s,'y_true':v['y_true'],'y_prob':v['y_prob']} for s,v in cv_results.items()]
pd.DataFrame(prob_rows).to_csv(os.path.join(TAB_DIR,'05_fusion_noage_probs.csv'),index=False)
fold_rows = [{'Fold':i+1,**m} for i,m in enumerate(fold_metrics)]
fold_rows.append({'Fold':'Mean',**{k:np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}})
fold_rows.append({'Fold':'Std', **{k:np.std ([m[k] for m in fold_metrics]) for k in fold_metrics[0]}})
pd.DataFrame(fold_rows).to_csv(os.path.join(TAB_DIR,'05_fusion_noage_fold_results.csv'),index=False)

# Training curves
ep = np.arange(1, EPOCHS+1)
m_tr=np.mean(all_tr_acc,axis=0)
m_vl=np.mean(all_vl_acc,axis=0)

fig, ax = plt.subplots(figsize=(10,5))

# Individual folds
for i in range(N_FOLDS):
    ax.plot(ep, all_tr_acc[i], color='#2980b9', lw=1.0, alpha=0.15)
    ax.plot(ep, all_vl_acc[i], color='#c0392b', lw=1.0, alpha=0.15, ls='--')
    
    best_ep = fold_metrics[i]['epoch']
    ax.scatter(best_ep, all_vl_acc[i][best_ep-1], color='#c0392b', s=25, zorder=5, edgecolors='white', alpha=0.6)

# Mean lines
ax.plot(ep, m_tr, '#2980b9', lw=2.5, label='Train Mean')
ax.plot(ep, m_vl, '#c0392b', lw=2.5, ls='--', label='Val Mean')

ax.axhline(0.5,color='gray',ls=':',alpha=0.6,label='Chance')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.set_title('Dual-Input CNN Fusion (Chirp + NI) — Training Curves',fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(1,EPOCHS)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'05_fusion_training_curves.png'),dpi=300,bbox_inches='tight')
plt.close()

# Performance bar
fig,ax=plt.subplots(figsize=(9,6))
names=['Accuracy','AUC','F1-Score','Sensitivity']
vals=[agg(k)[0] for k in ['acc','auc','f1','sens']]
errs=[agg(k)[1] for k in ['acc','auc','f1','sens']]
colors_=['#2980b9','#e67e22','#8e44ad','#27ae60']
bars=ax.bar(names,vals,yerr=errs,capsize=10,color=colors_,edgecolor='black',linewidth=1.5,alpha=0.85,width=0.5)
ax.axhline(0.5,color='gray',ls='--',lw=1.5,alpha=0.5,label='Chance')
ax.set_ylim(0,1.15); ax.set_ylabel('Score'); ax.legend()
ax.set_title(f'Dual CNN Fusion (Chirp + NI) — {N_FOLDS}-Fold CV',fontweight='bold')
ax.grid(axis='y',alpha=0.3)
for bar,v,e in zip(bars,vals,errs):
    ax.text(bar.get_x()+bar.get_width()/2.,v/2,f'{v:.1%}\n±{e:.1%}',
            ha='center',va='center',fontsize=12,fontweight='bold',color='white')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'05_fusion_performance_bar.png'),dpi=300,bbox_inches='tight')
plt.close()
print(f"\n✓ Figures → {FIG_DIR}")
print(f"✓ Tables  → {TAB_DIR}")
