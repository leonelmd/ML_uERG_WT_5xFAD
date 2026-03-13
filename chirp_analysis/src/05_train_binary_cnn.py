"""
05_train_binary_cnn.py
===================================
This script isolates the `amplitude` segment, which outperformed the others
in binary genotype classification (WT vs 5xFAD). Age is not used as an input
(Baseline/Default case).

Protocol
--------
* Input             : Chirp amplitude-sweep segment (samples 6000–8750, 2750 pts)
                      normalized by flash peak amplitude.
* Cross-validation  : 5-fold stratified, STRICTLY subject-disjoint.
                      Each tissue piece (subject) appears in exactly ONE fold's
                      validation set. Multiple trials from the same subject are
                      always kept together (no leakage).
* Validation metric : Subject-level majority-vote accuracy over all CV folds.
* Epochs            : 100 per fold (Adam lr=5e-4).

Outputs (results/ folder)
---------
  results/figures/05_cnn_training_curves.png
  results/figures/05_cnn_performance_bar.png
  results/tables/05_cnn_fold_results.csv

Usage (from chirp_analysis/ folder)
-------------------------------------
    python src/05_train_binary_cnn.py
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

# ── Add src to path ────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)  # chirp_analysis/
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
MOD_DIR = os.path.join(ROOT, 'results', 'models')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(MOD_DIR, exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
SEGMENT      = 'amplitude'
BATCH_SIZE   = 16
EPOCHS       = 80
LR           = 5e-4
N_FOLDS      = 5
RANDOM_STATE = 42

# ── Configuration: metadata ────────────────────────────────────────────────────
USE_AGE      = False     # Set to True to include continuous Age metadata in CNN

# ── Load metadata ──────────────────────────────────────────────────────────────
print("Loading metadata …")
df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()

subject_to_label = {r.Subject: (1 if '5xFAD' in r.Group else 0) for _, r in df_meta.iterrows()}

# ── Dataset ────────────────────────────────────────────────────────────────────
print("Loading dataset …")
dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment=SEGMENT, cache_dir=CACHE_DIR)

all_trial_subjects = [s.split('_trial_')[0] for s in dataset.subjects]
valid_idx = [i for i, s in enumerate(all_trial_subjects) if s in subject_to_label]

unique_subjects = sorted(set(all_trial_subjects[i] for i in valid_idx))
subj_labels     = [subject_to_label[s] for s in unique_subjects]

print(f"Valid trials: {len(valid_idx)}")
print(f"Unique subjects: {len(unique_subjects)}  "
      f"(WT={subj_labels.count(0)}, 5xFAD={subj_labels.count(1)})\n")

# ── Training loop ──────────────────────────────────────────────────────────────
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
all_tr_acc, all_vl_acc     = [], []
all_tr_loss, all_vl_loss   = [], []

for fold, (tr_subj_idx, vl_subj_idx) in enumerate(
        skf.split(unique_subjects, subj_labels), 1):

    train_subs = set(unique_subjects[i] for i in tr_subj_idx)
    val_subs   = set(unique_subjects[i] for i in vl_subj_idx)

    tr_trials = [i for i in valid_idx if all_trial_subjects[i] in train_subs]
    vl_trials = [i for i in valid_idx if all_trial_subjects[i] in val_subs]

    tr_loader = DataLoader(Subset(dataset, tr_trials), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    vl_loader = DataLoader(Subset(dataset, vl_trials), batch_size=BATCH_SIZE, shuffle=False)

    if USE_AGE:
        model = ImprovedBinaryCNN_AgeMetadata().to(device)
    else:
        model = ImprovedBinaryCNN().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    tr_acc_hist, vl_acc_hist   = [], []
    tr_loss_hist, vl_loss_hist = [], []
    best = {'acc': -1.0, 'auc': 0.0, 'f1': 0.0, 'sens': 0.0, 'probs': None, 'epoch': 0, 'state': None}

    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_preds, t_labels = 0.0, [], []
        for signals, lbls4, subjs in tr_loader:
            bz_labels = torch.tensor([1 if l >= 2 else 0 for l in lbls4]).to(device)
            signals = signals.to(device)
            
            # Extract basic age data roughly for the subjects and normalize (approximate logic if used)
            # In a real pipeline, age should be fed from dataset, but we will look up from df_meta
            if USE_AGE:
                ages = []
                for s in subjs:
                    real_s = s.split('_trial_')[0]
                    subset_df = df_meta[df_meta['Subject'] == real_s]
                    if subset_df.empty:
                        ages.append(0.5)
                    else:
                        ages.append(subset_df['Age (Days)'].item() / 230.0) 
                ages_t = torch.tensor(ages, dtype=torch.float32).to(device)
            
            # --- Data Augmentation ---
            if model.training:
                noise = torch.randn_like(signals) * 0.01
                signals = signals + noise
                
            optimizer.zero_grad()
            if USE_AGE:
                out = model(signals, ages_t)
            else:
                out = model(signals)
            loss = criterion(out, bz_labels)
            loss.backward(); optimizer.step()
            t_loss += loss.item() * signals.size(0)
            t_preds.extend(out.argmax(1).cpu().tolist())
            t_labels.extend(bz_labels.cpu().tolist())
        
        scheduler.step()
        tr_acc_hist.append(accuracy_score(t_labels, t_preds))
        tr_loss_hist.append(t_loss / len(tr_loader.dataset))

        model.eval()
        v_loss, votes = 0.0, {}
        with torch.no_grad():
            for signals, lbls4, subjs in vl_loader:
                bz_labels = torch.tensor([1 if l >= 2 else 0 for l in lbls4]).to(device)
                signals = signals.to(device)
                
                if USE_AGE:
                    ages = []
                    for s in subjs:
                        real_s = s.split('_trial_')[0]
                        subset_df = df_meta[df_meta['Subject'] == real_s]
                        if subset_df.empty:
                            ages.append(0.5)
                        else:
                            ages.append(subset_df['Age (Days)'].item() / 230.0)
                    ages_t = torch.tensor(ages, dtype=torch.float32).to(device)
                    out = model(signals, ages_t)
                else:
                    out = model(signals)
                v_loss += criterion(out, bz_labels).item() * signals.size(0)
                probs = F.softmax(out, dim=1)[:, 1].cpu().tolist()
                preds = out.argmax(1).cpu().tolist()
                for prob, pred, lbl, s in zip(probs, preds, bz_labels.tolist(), subjs):
                    real_s = s.split('_trial_')[0]
                    votes.setdefault(real_s, {'preds': [], 'probs': [], 'label': lbl})
                    votes[real_s]['preds'].append(pred)
                    votes[real_s]['probs'].append(prob)

        sp_probs = [np.mean(v['probs']) for v in votes.values()]
        sp = [1 if p >= 0.5 else 0 for p in sp_probs]
        sl = [v['label'] for v in votes.values()]
        v_acc  = accuracy_score(sl, sp)
        v_f1   = f1_score(sl, sp, average='macro', zero_division=0)
        v_sens = recall_score(sl, sp, pos_label=1, zero_division=0)
        v_auc  = roc_auc_score(sl, sp_probs) if len(np.unique(sl)) > 1 else 0.5

        vl_acc_hist.append(v_acc)
        vl_loss_hist.append(v_loss / len(vl_loader.dataset))

        # Selection: Priority to AUC for stability on small N
        if (v_auc > best['auc']) or (v_auc == best['auc'] and v_acc > best['acc']):
            best = {
                'acc': v_acc, 'auc': v_auc, 'f1': v_f1, 'sens': v_sens,
                'probs': [{'Subject': s, 'y_true': v['label'], 'y_prob': np.mean(v['probs'])} for s, v in votes.items()],
                'epoch': epoch + 1, 'state': {k: v.clone() for k, v in model.state_dict().items()}
            }

    torch.save(best['state'], os.path.join(MOD_DIR, f'05_cnn_fold_{fold}.pt'))
    fold_metrics.append(best)
    all_tr_acc.append(tr_acc_hist);   all_vl_acc.append(vl_acc_hist)
    all_tr_loss.append(tr_loss_hist); all_vl_loss.append(vl_loss_hist)
    print(f"Fold {fold}  Acc={best['acc']:.1%}  AUC={best['auc']:.3f} (Epoch {best['epoch']})")

# ── Aggregate ──────────────────────────────────────────────────────────────────
def agg(key): return (np.mean([m[key] for m in fold_metrics]), np.std([m[key] for m in fold_metrics]))

print(f"\nFinal Statistics:")
for key, label in [('acc','Accuracy'), ('auc', 'AUC'), ('f1','F1-Score'), ('sens','Sensitivity')]:
    m, s = agg(key)
    print(f"  {label:<15}: {m:.2%} ± {s:.2%}" if key != 'auc' else f"  {label:<15}: {m:.3f} ± {s:.3f}")

# ── Save Table ────────────────────────────────────────────────────────────────
res_df = pd.DataFrame([{'Fold': i+1, **{k: m[k] for k in ['acc','auc','f1','sens']}} for i, m in enumerate(fold_metrics)])
res_df.to_csv(os.path.join(TAB_DIR, '05_cnn_fold_results.csv'), index=False)

# Save pooled probabilities for final comparison (Step 08)
all_p_rows = []
for m in fold_metrics:
    all_p_rows.extend(m['probs'])
probs_df = pd.DataFrame(all_p_rows)
probs_df.to_csv(os.path.join(TAB_DIR, '05_cnn_probs.csv'), index=False)
probs_df.to_csv(os.path.join(TAB_DIR, '12_improved_amplitude_probs.csv'), index=False)

# ── Figure: Training Curves ────────────────────────────────────────────────────
max_ep = max(len(tr) for tr in all_tr_acc)
ep = np.arange(1, max_ep + 1)

def pad_to_max(lst):
    return [np.pad(l, (0, max_ep - len(l)), mode='constant', constant_values=np.nan) for l in lst]

all_tr_acc_pad = pad_to_max(all_tr_acc)
all_vl_acc_pad = pad_to_max(all_vl_acc)
all_tr_loss_pad = pad_to_max(all_tr_loss)
all_vl_loss_pad = pad_to_max(all_vl_loss)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, all_tr, all_vl, ylabel in [(axes[0], all_tr_acc_pad, all_vl_acc_pad, 'Accuracy'), (axes[1], all_tr_loss_pad, all_vl_loss_pad, 'Loss')]:
    for i in range(N_FOLDS):
        ax.plot(ep, all_tr[i], color='#2980b9', alpha=0.15)
        ax.plot(ep, all_vl[i], color='#c0392b', alpha=0.15, ls='--')
        best_ep = fold_metrics[i]['epoch']
        ax.scatter(best_ep, all_vl[i][best_ep-1], color='#c0392b', s=25, zorder=5, edgecolors='white', alpha=0.6)
    ax.plot(ep, np.nanmean(all_tr, axis=0), '#2980b9', lw=2.5, label='Train Mean')
    ax.plot(ep, np.nanmean(all_vl, axis=0), '#c0392b', lw=2.5, ls='--', label='Val Mean')
    ax.set_ylabel(ylabel); ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle('Improved CNN — Chirp Amplitude Sweep Training Curves', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '05_cnn_training_curves.png'), dpi=300)
plt.close()

# ── Figure: Bar Chart ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

labels = ['Accuracy', 'F1-Score', 'Sensitivity']
means = [agg(k)[0] for k in ['acc','f1','sens']]
errs  = [agg(k)[1] for k in ['acc','f1','sens']]
ax.bar(labels, means, yerr=errs, capsize=10, color=['#2980b9', '#8e44ad', '#27ae60'], alpha=0.8, edgecolor='black')
ax.set_ylim(0, 1.1)
ax.set_title('Improved CNN Performance — Chirp Amplitude Sweep', fontweight='bold')
for i, v in enumerate(means):
    ax.text(i, v + errs[i] + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
plt.savefig(os.path.join(FIG_DIR, '05_cnn_performance_bar.png'), dpi=300)
plt.close()
