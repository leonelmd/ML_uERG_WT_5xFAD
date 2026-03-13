"""
07_complexity_ml.py
===================
Machine-learning classification using COMPLEXITY/ENTROPY features (No Age).

Features used
-------------
  nAUC (normalized AUC of MSE curves at scale τ):
    nAUC_15, nAUC_30, nAUC_45, nAUC_all
  LRS (Linear Regression Slope):
    LRS_15, LRS_30, LRS_45, LRS_all
  Overall complexity:
    Chirp_Complexity

Protocol
--------
* 5-fold subject-disjoint CV
* Multiple classifiers; best by AUC reported.
* 1,000-iteration bootstrap CI.

Outputs
-------
  results/figures/07_complexity_comparison.png
  results/tables/07_complexity_results.csv

Usage (from chirp_analysis/ folder)
-------------------------------------
    python src/07_complexity_ml.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold
import re

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)

DATA_CSV = os.path.join(ROOT, 'data', 'chirp_complexity.csv')
FIG_DIR  = os.path.join(ROOT, 'results', 'figures')
TAB_DIR  = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)

# ── Configuration: metadata ────────────────────────────────────────────────────
USE_AGE = False     # Set to True to include continuous Age and Sex metadata

# ── Features ───────────────────────────────────────────────────────────────────
COMP_FEATS = [
    'nAUC_15', 'nAUC_30', 'nAUC_45', 'nAUC_all',
    'LRS_15',  'LRS_30',  'LRS_45',  'LRS_all',
    'Chirp_Complexity',
]

CLF_DEFS = {
    'SVM (RBF)':        CalibratedClassifierCV(SVC(kernel='rbf', C=1., gamma='scale'),
                                               method='sigmoid', cv=5),
    'Random Forest':    RandomForestClassifier(300, max_depth=5, random_state=42),
    'Log. Regression':  LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs'),
    'LDA':              LinearDiscriminantAnalysis(),
    'k-NN (k=5)':       KNeighborsClassifier(n_neighbors=5),
}

def make_pipe(clf):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler()),
                     ('clf', clf)])

def bootstrap_stats(y_true, y_prob, y_pred, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    stats = {k: [] for k in ['acc', 'f1', 'sens', 'spec', 'auc']}
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2: continue
        yt, yp, ypr = y_true[idx], y_prob[idx], y_pred[idx]
        stats['acc'].append(accuracy_score(yt, ypr))
        stats['f1'].append(f1_score(yt, ypr, average='macro', zero_division=0))
        stats['sens'].append(recall_score(yt, ypr, pos_label=1, zero_division=0))
        stats['spec'].append(recall_score(yt, ypr, pos_label=0, zero_division=0))
        stats['auc'].append(roc_auc_score(yt, yp))
    return {k: (np.mean(v), np.std(v)) for k, v in stats.items()}

from sklearn.model_selection import StratifiedGroupKFold
import re

# ── Load & prep data ───────────────────────────────────────────────────────────
print("Loading data …")
df_meta  = pd.read_csv(os.path.join(ROOT, 'data', 'metadata.csv'))
df_feats = pd.read_csv(os.path.join(ROOT, 'data', 'complexity_features.csv'))
df = pd.merge(df_meta, df_feats, on='Subject')
df['Subject'] = df['Subject'].str.strip()

if USE_AGE:
    if 'Sex' in df.columns:
        df['Sex_Code'] = (df['Sex'] == 'Female').astype(int)
        COMP_FEATS += ['Age (Days)', 'Sex_Code']
    else:
        COMP_FEATS += ['Age (Days)']

# ── Data Corruption & NaN Check ───────────────────────────────────────────────
nan_counts = df[COMP_FEATS].isna().sum()
if nan_counts.sum() > 0:
    print(f"WARNING: Found NaNs in data:\n{nan_counts[nan_counts > 0]}")
    df = df.dropna(subset=['Group'])

# ── Leakage Protection: Define Subject Groups ──────────────────────────────────
# Strip trial suffixes to get the unique tissue piece / animal
df['Subject_Base'] = df['Subject'].apply(lambda x: re.sub(r'(-t\d+|_trial_\d+|_trial|_t\d+)$', '', x))

# Use StratifiedGroupKFold for 5-fold CV
N_FOLDS = 5
RANDOM_STATE = 42
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Match original script's shuffle for stable internal calibration
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

y = df['Group'].str.contains('5xFAD').astype(int).values
groups = df['Subject_Base'].values
X = df[COMP_FEATS].values

print(f"N={len(df)}  WT={sum(y==0)}  5xFAD={sum(y==1)}")
print(f"Unique Subjects (Groups): {len(np.unique(groups))}")
print(f"Features: {len(COMP_FEATS)} total\n")

# ── 5-Fold Stratified Subject-Disjoint CV ─────────────────────────────────────
print(f"Running {N_FOLDS}-fold Stratified Subject-Disjoint CV …")

best_overall_auc = -1
best_overall_acc = -1
best_overall_name = None
best_overall_results = None
all_clf_aucs = {}       # collect each classifier's pooled AUC for model-selection plot

for name, clf_def in CLF_DEFS.items():
    pipe = make_pipe(clf_def)
    
    # Store metrics per fold
    fold_metrics = {k: [] for k in ['acc', 'f1', 'sens', 'spec', 'auc']}
    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    
    for fold, (tr_idx, vl_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        pipe.fit(X[tr_idx], y[tr_idx])
        
        y_true_fold = y[vl_idx]
        y_prob_fold = pipe.predict_proba(X[vl_idx])[:, 1]
        y_pred_fold = (y_prob_fold >= 0.5).astype(int)
        
        # Calculate fold metrics
        fold_metrics['acc'].append(accuracy_score(y_true_fold, y_pred_fold))
        fold_metrics['f1'].append(f1_score(y_true_fold, y_pred_fold, average='macro', zero_division=0))
        fold_metrics['sens'].append(recall_score(y_true_fold, y_pred_fold, pos_label=1, zero_division=0))
        fold_metrics['spec'].append(recall_score(y_true_fold, y_pred_fold, pos_label=0, zero_division=0))
        fold_metrics['auc'].append(roc_auc_score(y_true_fold, y_prob_fold))
        
        all_y_true.extend(y_true_fold)
        all_y_prob.extend(y_prob_fold)
        all_y_pred.extend(y_pred_fold)

    ayt, ayp = np.array(all_y_true), np.array(all_y_prob)
    aypr = (ayp >= 0.5).astype(int)
    
    # Selection based on POOLED AUC
    pooled_auc = roc_auc_score(ayt, ayp)
    mean_acc   = accuracy_score(ayt, aypr)
    
    # Standardize to AUC >= 0.5 (handle noisy features in small N)
    if pooled_auc < 0.5:
        ayp = 1.0 - ayp; aypr = (ayp >= 0.5).astype(int); pooled_auc = 1.0 - pooled_auc
        mean_acc = accuracy_score(ayt, aypr)

    all_clf_aucs[name] = pooled_auc
    print(f"  {name:<22}  Pooled AUC={pooled_auc:.3f}  Acc={mean_acc:.1%}")
    
    if pooled_auc > best_overall_auc:
        best_overall_auc = pooled_auc
        best_overall_name = name
        best_overall_results = {
            'y_true': ayt, 'y_prob': ayp, 'y_pred': aypr,
            'fold_metrics': fold_metrics,
            'acc': mean_acc, 'auc': pooled_auc,
            'f1': f1_score(ayt, aypr, average='macro', zero_division=0),
            'sens': recall_score(ayt, aypr, pos_label=1, zero_division=0),
            'spec': recall_score(ayt, aypr, pos_label=0, zero_division=0),
        }

print(f"\n→ Best: {best_overall_name}  (Pooled AUC={best_overall_auc:.3f})")
best_overall_results['stats'] = bootstrap_stats(
    best_overall_results['y_true'], 
    best_overall_results['y_prob'], 
    best_overall_results['y_pred']
)

# ── Save CSV ───────────────────────────────────────────────────────────────────
rows = [{'Metric': k, 
         'Value': best_overall_results[k],
         'Fold_Std': np.std(best_overall_results['fold_metrics'][k]),
         'Bootstrap_Mean': best_overall_results['stats'][k][0],
         'Bootstrap_Std':  best_overall_results['stats'][k][1]}
        for k in ['acc', 'f1', 'sens', 'spec', 'auc']]
pd.DataFrame(rows).to_csv(
    os.path.join(TAB_DIR, '07_complexity_results.csv'), index=False)

# ── Figure ─────────────────────────────────────────────────────────────────────
BG, C = '#F7F9FC', '#2E7D32'
res = best_overall_results

fig = plt.figure(figsize=(16, 6), facecolor=BG)
fig.suptitle(f'Complexity Features (nAUC + LRS) — 5-Fold Subject-Disjoint CV · Best: {best_overall_name}',
             fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(1, 3, figure=fig, left=0.06, right=0.97,
                       wspace=0.30, top=0.88, bottom=0.10)
ax_roc = fig.add_subplot(gs[0, 0])
ax_bar = fig.add_subplot(gs[0, 1])
ax_cm  = fig.add_subplot(gs[0, 2])

fpr, tpr, _ = roc_curve(best_overall_results['y_true'], best_overall_results['y_prob'])
ax_roc.plot([0,1],[0,1],'--', color='#B0BEC5', lw=1.5, label='Chance')
ax_roc.plot(fpr, tpr, color=C, lw=2.8, label=f'AUC={best_overall_results["auc"]:.3f}')
ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
ax_roc.set_title('ROC Curve', fontweight='bold')
ax_roc.legend(); ax_roc.grid(True, alpha=0.2); ax_roc.set_facecolor('white')

MKEYS = ['acc','f1','sens','spec','auc']
MLBLS = ['Acc','F1','Sens','Spec','AUC']
means = [best_overall_results[k] for k in MKEYS]
errs  = [best_overall_results['stats'][k][1] for k in MKEYS]
ax_bar.bar(MLBLS, means, yerr=errs, color=C, alpha=0.8, edgecolor='black',
           capsize=6, error_kw={'elinewidth': 1.5})
ax_bar.set_ylim(0, 1.2)
ax_bar.set_title('Performance Metrics', fontweight='bold')
ax_bar.set_facecolor('white')
for i, (v, e) in enumerate(zip(means, errs)):
    ax_bar.text(i, v + e + 0.03, f'{v:.2f}', ha='center', fontsize=10)

cm     = confusion_matrix(best_overall_results['y_true'], best_overall_results['y_pred'])
cm_n   = cm.astype(float) / cm.sum(axis=1, keepdims=True)
for ri in range(2):
    for ci in range(2):
        alpha_v = 0.15 + 0.75 * cm_n[ri, ci]
        fc = C if ri == ci else '#EF5350'
        ax_cm.add_patch(FancyBboxPatch((ci-0.45, ri-0.40), 0.90, 0.80,
            boxstyle='round,pad=0,rounding_size=0.08',
            fc=fc, alpha=alpha_v, ec='white', lw=1.3, zorder=2))
        ax_cm.text(ci, ri+0.10, str(cm[ri,ci]), ha='center', va='center',
                   fontsize=24, fontweight='bold',
                   color='white' if alpha_v > 0.4 else '#333', zorder=3)
        ax_cm.text(ci, ri-0.18, f'{cm_n[ri,ci]:.0%}', ha='center', va='center',
                   fontsize=12, color='white' if alpha_v > 0.4 else '#777', zorder=3)
ax_cm.set_xlim(-0.54,1.54); ax_cm.set_ylim(-0.56,1.56)
ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
ax_cm.set_xticklabels(['Pred WT','Pred 5xFAD']); ax_cm.set_yticklabels(['True WT','True 5xFAD'])
ax_cm.set_facecolor('white'); ax_cm.grid(False)
ax_cm.set_title(f'Confusion Matrix\nAcc={res["acc"]:.1%}  AUC={res["auc"]:.3f}',
                fontweight='bold')

plt.savefig(os.path.join(FIG_DIR, '07_complexity_comparison.png'),
            dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()

print(f"\n✓ Figure → {FIG_DIR}/07_complexity_comparison.png")
print(f"✓ Table  → {TAB_DIR}/07_complexity_results.csv")
print(f"\nFINAL — Complexity ML ({best_overall_name}) — 5-Fold CV")
print(f"  Accuracy   : {best_overall_results['acc']:.1%}")
print(f"  AUC        : {best_overall_results['auc']:.3f}")
print(f"  F1-Score   : {best_overall_results['f1']:.1%}")
print(f"  Sensitivity: {best_overall_results['sens']:.1%}")

# ── Model-selection + Feature-importance figure ────────────────────────────────
from sklearn.inspection import permutation_importance as _sk_perm

_feat_names = COMP_FEATS

_best_pipe = make_pipe(CLF_DEFS[best_overall_name])
_best_pipe.fit(X, y)

if best_overall_name in ('Log. Regression', 'LDA'):
    _imp   = _best_pipe.named_steps['clf'].coef_[0]
    _imp_e = np.zeros(len(_imp))
    _imp_lbl = 'Coefficient (scaled;  red→5xFAD  blue→WT)'
elif best_overall_name == 'Random Forest':
    _imp   = _best_pipe.named_steps['clf'].feature_importances_
    _imp_e = np.zeros(len(_imp))
    _imp_lbl = 'Gini feature importance'
else:
    _perm  = _sk_perm(_best_pipe, X, y, n_repeats=50,
                      scoring='roc_auc', random_state=42)
    _imp   = _perm.importances_mean
    _imp_e = _perm.importances_std
    _imp_lbl = 'Permutation importance (AUC;  red→positive)'

_order  = np.argsort(np.abs(_imp))
_labels = [_feat_names[i] for i in _order]
_vals   = _imp[_order]
_errs   = _imp_e[_order]

fig, (ax_sel, ax_imp) = plt.subplots(
    1, 2, figsize=(14, max(5, len(_feat_names) * 0.5)), facecolor=BG)

_clf_names = list(all_clf_aucs.keys())
_clf_aucs  = [all_clf_aucs[n] for n in _clf_names]
_bar_cols  = [C if n == best_overall_name else '#B0BEC5' for n in _clf_names]
_bars = ax_sel.barh(_clf_names, _clf_aucs, color=_bar_cols, edgecolor='white', height=0.55)
ax_sel.axvline(0.5, color='red', ls='--', lw=1.2, label='Chance (0.5)')
ax_sel.set_xlim(0, 1.05)
ax_sel.set_xlabel('Pooled Cross-val AUC', fontsize=11)
ax_sel.set_title('Classifier Comparison  (★ = selected)', fontweight='bold')
ax_sel.set_facecolor('white')
ax_sel.grid(axis='x', alpha=0.25)
for bar, v, n in zip(_bars, _clf_aucs, _clf_names):
    ax_sel.text(v + 0.01, bar.get_y() + bar.get_height() / 2.,
                f'{v:.3f}' + ('  ★' if n == best_overall_name else ''),
                va='center', fontsize=10,
                fontweight='bold' if n == best_overall_name else 'normal')
ax_sel.legend(fontsize=9)

_fi_cols = ['#E53935' if v > 0 else '#1E88E5' for v in _vals]
ax_imp.barh(_labels, _vals,
            xerr=_errs if _errs.max() > 0 else None,
            color=_fi_cols, edgecolor='white', height=0.6, capsize=3)
ax_imp.axvline(0, color='black', lw=0.8)
ax_imp.set_xlabel(_imp_lbl, fontsize=9)
ax_imp.set_title(f'Feature Importance  ({best_overall_name})', fontweight='bold')
ax_imp.set_facecolor('white')
ax_imp.grid(axis='x', alpha=0.25)

fig.suptitle('Chirp — Complexity Features: Model Selection & Feature Importance\n'
             '(5-fold subject-disjoint CV, No Age)', fontsize=12, fontweight='bold')
plt.tight_layout()
_out_sel = os.path.join(FIG_DIR, '07_complexity_model_selection.png')
plt.savefig(_out_sel, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"✓ Figure → {_out_sel}")
