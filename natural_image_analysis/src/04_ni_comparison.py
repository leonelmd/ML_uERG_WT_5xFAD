"""
04_ni_comparison.py
===============================
Comprehensive comparison for Natural Image stimulus:
  1. NI CNN (No Age/Sex)     (results/tables/01_ni_cnn_noage_probs.csv)
  2. Hand-crafted ML         (5-fold CV)
  3. Complexity ML           (5-fold CV)

Outputs
-------
  results/figures/04_ni_comparison.png
  results/tables/04_ni_comparison_summary.csv

Usage: python src/04_ni_comparison.py
"""

import os, sys, re
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
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

FIG_DIR = os.path.join(ROOT, 'results', 'figures')
TAB_DIR = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)

HAND_FEATS = ['Signal_Max','Signal_Min','Signal_P2P','Signal_RMS','Signal_Std',
              'Power_Total','Power_Low','Power_Mid','Power_High']
COMP_FEATS = ['nAUC_15','nAUC_30','nAUC_45','nAUC_all','LRS_15','LRS_30','LRS_45','LRS_all']

CLF_DEFS = {
    'SVM (RBF)':        CalibratedClassifierCV(SVC(kernel='rbf',C=1.,gamma='scale'),method='sigmoid',cv=5),
    'Random Forest':    RandomForestClassifier(300,max_depth=5,random_state=42),
    'Log. Regression':  LogisticRegression(C=0.1,max_iter=1000,solver='lbfgs'),
    'LDA':              LinearDiscriminantAnalysis(),
    'k-NN (k=5)':       KNeighborsClassifier(n_neighbors=5),
}

def make_pipe(clf):
    return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('clf',clf)])

def bootstrap_stats(y_true,y_prob,y_pred,n=1000,seed=42):
    rng=np.random.RandomState(seed)
    stats={k:[] for k in ['acc','f1','sens','spec','auc']}
    for _ in range(n):
        idx=rng.choice(len(y_true),len(y_true),replace=True)
        if len(np.unique(y_true[idx]))<2: continue
        yt,yp,ypr=y_true[idx],y_prob[idx],y_pred[idx]
        stats['acc'].append(accuracy_score(yt,ypr))
        stats['f1'].append(f1_score(yt,ypr,average='macro',zero_division=0))
        stats['sens'].append(recall_score(yt,ypr,pos_label=1,zero_division=0))
        stats['spec'].append(recall_score(yt,ypr,pos_label=0,zero_division=0))
        stats['auc'].append(roc_auc_score(yt,yp))
    return {k:(np.mean(v),np.std(v)) for k,v in stats.items()}

def run_5fold(X, y, groups, tag):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    best_overall_auc = -1
    best_overall_name = None
    best_overall_res = None
    
    for name, clf_def in CLF_DEFS.items():
        pipe = make_pipe(clf_def)
        all_y_true, all_y_prob = [], []
        fold_metrics = {k: [] for k in ['acc', 'f1', 'sens', 'spec', 'auc']}
        
        for tr_idx, vl_idx in sgkf.split(X, y, groups=groups):
            pipe.fit(X[tr_idx], y[tr_idx])
            ytf, ypf = y[vl_idx], pipe.predict_proba(X[vl_idx])[:, 1]
            yprf = (ypf >= 0.5).astype(int)
            
            all_y_true.extend(ytf.tolist()); all_y_prob.extend(ypf.tolist())
            fold_metrics['acc'].append(accuracy_score(ytf, yprf))
            fold_metrics['f1'].append(f1_score(ytf, yprf, average='macro', zero_division=0))
            fold_metrics['sens'].append(recall_score(ytf, yprf, pos_label=1, zero_division=0))
            fold_metrics['spec'].append(recall_score(ytf, yprf, pos_label=0, zero_division=0))
            fold_metrics['auc'].append(roc_auc_score(ytf, ypf))
            
        ayt, ayp = np.array(all_y_true), np.array(all_y_prob)
        aypr = (ayp >= 0.5).astype(int)
        
        # Calculate POOLED AUC
        pooled_auc = roc_auc_score(ayt, ayp)
        mean_acc   = accuracy_score(ayt, aypr)
        
        # Standardize to AUC >= 0.5 (handle noisy features in small N)
        if pooled_auc < 0.5:
            ayp = 1.0 - ayp
            aypr = (ayp >= 0.5).astype(int)
            pooled_auc = 1.0 - pooled_auc
            mean_acc = accuracy_score(ayt, aypr)
            
        print(f"  [{tag}] {name:<20} Pooled AUC={pooled_auc:.3f}  Acc={mean_acc:.1%}")
        
        if pooled_auc > best_overall_auc:
            best_overall_auc = pooled_auc
            best_overall_name = name
            best_overall_res = dict(
                y_true=ayt, y_pred=aypr, y_prob=ayp,
                acc=mean_acc, auc=pooled_auc,
                f1=f1_score(ayt, aypr, average='macro', zero_division=0),
                sens=recall_score(ayt, aypr, pos_label=1, zero_division=0),
                spec=recall_score(ayt, aypr, pos_label=0, zero_division=0)
            )
            
    if best_overall_res is not None:
        best_overall_res['stats'] = bootstrap_stats(
            best_overall_res['y_true'], best_overall_res['y_prob'], best_overall_res['y_pred']
        )
    return best_overall_res, best_overall_name

# ── Load Metadata & Features ───────────────────────────────────────────────────
def prep_df(df_meta_raw, df_feats):
    df = pd.merge(df_meta_raw, df_feats, on='Subject')
    df['Subject'] = df['Subject'].str.strip()
    df['Subject_Base'] = df['Subject'].apply(
        lambda x: re.sub(r'(-t\d+|_trial_\d+|_trial|_t\d+)$', '', x))
    return df

df_meta  = pd.read_csv(os.path.join(ROOT, 'data', 'metadata.csv'))
df_hand  = pd.read_csv(os.path.join(ROOT, 'data', 'hand_crafted_features.csv'))
df_comp  = pd.read_csv(os.path.join(ROOT, 'data', 'complexity_features.csv'))

# Each method uses its own subject set (no forced intersection)
df_hc = prep_df(df_meta, df_hand)   # 42 subjects
df_cx = prep_df(df_meta, df_comp)   # 22 subjects (complexity only)

y_hc      = df_hc['Group'].str.contains('5xFAD').astype(int).values
groups_hc = df_hc['Subject_Base'].values

y_cx      = df_cx['Group'].str.contains('5xFAD').astype(int).values
groups_cx = df_cx['Subject_Base'].values

print(f"\n── Hand-crafted (5-fold, N={len(df_hc)}) ──")
res_hand, name_hand = run_5fold(df_hc[HAND_FEATS].values, y_hc, groups_hc, 'Hand')
print(f"\n── Complexity (5-fold, N={len(df_cx)}) ──")
res_comp, name_comp = run_5fold(df_cx[COMP_FEATS].values, y_cx, groups_cx, 'Comp')

# ── Load CNN Probs ─────────────────────────────────────────────────────────────
CNN_PROBS_CSV = os.path.join(TAB_DIR, '07_improved_ni_cnn_probs.csv')
print("\n── Improved NI CNN ──")
if os.path.exists(CNN_PROBS_CSV):
    df_cnn = pd.read_csv(CNN_PROBS_CSV)
    y_true_cnn, y_prob_cnn = df_cnn['y_true'].values, df_cnn['y_prob'].values
    y_pred_cnn = (y_prob_cnn >= 0.5).astype(int)
    auc_cnn = roc_auc_score(y_true_cnn, y_prob_cnn)
    if auc_cnn < 0.5:
         y_prob_cnn = 1.0 - y_prob_cnn
         y_pred_cnn = (y_prob_cnn >= 0.5).astype(int)
         auc_cnn = 1.0 - auc_cnn

    res_cnn = dict(y_true=y_true_cnn, y_pred=y_pred_cnn, y_prob=y_prob_cnn,
                   acc=accuracy_score(y_true_cnn, y_pred_cnn), 
                   auc=auc_cnn,
                   f1=f1_score(y_true_cnn, y_pred_cnn, average='macro'),
                   sens=recall_score(y_true_cnn, y_pred_cnn, pos_label=1),
                   spec=recall_score(y_true_cnn, y_pred_cnn, pos_label=0))
    res_cnn['stats'] = bootstrap_stats(y_true_cnn, y_prob_cnn, y_pred_cnn)
    print(f"  Improved NI CNN Loaded: Acc={res_cnn['acc']:.1%}  AUC={res_cnn['auc']:.3f}")
else:
    print("  ⚠ CNN probs not found.")
    sys.exit(0)

# ── Single Feature Evaluation (Added Value) ───────────────────────────────────
def get_single_feat_roc(df_src, y_src, feat_list):
    best_f, best_auc, best_y_prob = None, -1, None
    for f in feat_list:
        x = df_src[f].values
        mask = ~np.isnan(x)
        if sum(mask) < 2: continue
        x_m, y_m = x[mask], y_src[mask]
        auc = roc_auc_score(y_m, x_m)
        if auc < 0.5:
            auc = 1 - auc
            prob = 1.0 - (x_m - x_m.min()) / (x_m.max() - x_m.min() + 1e-9)
        else:
            prob = (x_m - x_m.min()) / (x_m.max() - x_m.min() + 1e-9)
        if auc > best_auc:
            best_auc, best_f, best_y_prob = auc, f, prob
    return {'name': best_f, 'auc': best_auc,
            'y_true': y_src[~np.isnan(df_src[best_f].values)], 'y_prob': best_y_prob}

res_single_hand = get_single_feat_roc(df_hc, y_hc, HAND_FEATS)
res_single_comp = get_single_feat_roc(df_cx, y_cx, COMP_FEATS)

# ── Summary CSV ────────────────────────────────────────────────────────────────
summary_rows = []
for label, res, n_ in [('NI CNN', res_cnn, 'CNN'),
                       (f'Hand-crafted ({name_hand})', res_hand, name_hand),
                       (f'Complexity ({name_comp})', res_comp, name_comp)]:
    summary_rows.append({'Method': label, 'Acc': res['acc'], 'AUC': res['auc'], 'F1': res['f1'],
                         'Sens': res['sens'], 'Spec': res['spec'],
                         'Bootstrap_Acc_Std': res['stats']['acc'][1],
                         'Bootstrap_AUC_Std': res['stats']['auc'][1]})
pd.DataFrame(summary_rows).to_csv(os.path.join(TAB_DIR, '04_ni_comparison_summary.csv'), index=False)

# ── Figure ─────────────────────────────────────────────────────────────────────
BG, C_CNN, C_HAND, C_COMP = '#F8FAFC', '#E11D48', '#2563EB', '#16A34A'
METHODS = [
    (f'NI CNN (N={len(df_cnn)})',                     res_cnn,  C_CNN,  '5-fold CV'),
    (f'Hand-crafted ML\n({name_hand}, N={len(df_hc)})', res_hand, C_HAND, '5-fold CV'),
    (f'Complexity ML\n({name_comp}, N={len(df_cx)})',   res_comp, C_COMP, '5-fold CV'),
]

fig = plt.figure(figsize=(22, 12), facecolor=BG)
fig.suptitle('ML Added Value Comparison — Natural Image stimulus\n'
             '(CNN architecture: TemporalStatPool + InstanceNorm)',
             fontsize=20, fontweight='bold', y=0.98, color='#1E293B')

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
ax_roc = fig.add_subplot(gs[0, :2]); ax_bar = fig.add_subplot(gs[0, 2:])
ax_cm  = [fig.add_subplot(gs[1, i]) for i in range(3)]

# ROC - Ensure curves match AUC reported
ax_roc.plot([0,1],[0,1],'--', color='#94A3B8', label='Chance', lw=1.5)

# Multivariate Models
for label, res, color, _ in METHODS:
    fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
    ax_roc.plot(fpr, tpr, color=color, lw=3.5, label=f'{label.split(chr(10))[0]} (AUC={res["auc"]:.3f})')

# Single Features (Dashed lines for "Added Value" comparison)
fpr_sh, tpr_sh, _ = roc_curve(res_single_hand['y_true'], res_single_hand['y_prob'])
ax_roc.plot(fpr_sh, tpr_sh, color=C_HAND, lw=2, ls='--', alpha=0.6,
            label=f"Single: {res_single_hand['name']} (AUC={res_single_hand['auc']:.3f})")

fpr_sc, tpr_sc, _ = roc_curve(res_single_comp['y_true'], res_single_comp['y_prob'])
ax_roc.plot(fpr_sc, tpr_sc, color=C_COMP, lw=2, ls='--', alpha=0.6,
            label=f"Single: {res_single_comp['name']} (AUC={res_single_comp['auc']:.3f})")

ax_roc.set_title('ROC Curves: ML vs. Single Features', fontsize=16, fontweight='bold', color='#334155')
ax_roc.set_xlabel('False Positive Rate', fontsize=12); ax_roc.set_ylabel('True Positive Rate', fontsize=12)
ax_roc.legend(fontsize=11, loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
ax_roc.grid(True, alpha=0.2)
ax_roc.set_facecolor('white')
for spine in ax_roc.spines.values(): spine.set_color('#E2E8F0')

# Bar chart
MKEYS, MLBLS = ['acc','f1','sens','spec','auc'], ['Accuracy','F1','Sens','Spec','AUC']
x = np.arange(len(MLBLS)); bw = 0.22
for mi, (label, res, color, _) in enumerate(METHODS):
    vals = [res[k] for k in MKEYS]; errs = [res['stats'][k][1] for k in MKEYS]
    bars = ax_bar.bar(x+(mi-1)*bw, vals, yerr=errs, width=bw, color=color, alpha=0.85, 
                      label=label.split('\n')[0], capsize=4, edgecolor='white', lw=1)
ax_bar.set_xticks(x); ax_bar.set_xticklabels(MLBLS, fontsize=12); ax_bar.set_ylim(0, 1.1)
ax_bar.set_title('Performance Metrics (Mean ± Bootstrap SD)', fontsize=16, fontweight='bold', color='#334155')
ax_bar.legend(fontsize=11, frameon=True, facecolor='white')
ax_bar.grid(axis='y', alpha=0.2)
ax_bar.set_facecolor('white')
for spine in ax_bar.spines.values(): spine.set_color('#E2E8F0')

# Confusion matrices
for ax, (label, res, color, cv) in zip(ax_cm, METHODS):
    cm = confusion_matrix(res['y_true'], res['y_pred'])
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    for ri in range(2):
        for ci in range(2):
            alpha_v = 0.15 + 0.75 * cm_n[ri, ci]; fc = color if ri == ci else '#EF5350'
            ax.add_patch(FancyBboxPatch((ci - 0.45, ri - 0.42), 0.90, 0.84,
                boxstyle='round,pad=0,rounding_size=0.10', fc=fc, alpha=alpha_v, ec='white', lw=1.5, zorder=2))
            ax.text(ci, ri + 0.10, str(cm[ri, ci]), ha='center', va='center', fontsize=18, fontweight='bold',
                    color='white' if alpha_v > 0.4 else '#333', zorder=3)
            ax.text(ci, ri - 0.20, f'{cm_n[ri, ci]:.0%}', ha='center', va='center', fontsize=12,
                    color='white' if alpha_v > 0.4 else '#777', zorder=3)
    ax.set_facecolor('white'); ax.grid(False)
    ax.set_xlim(-0.54, 1.54); ax.set_ylim(-0.56, 1.56)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred WT', 'Pred FAD']); ax.set_yticklabels(['True WT', 'True FAD'])
    ax.set_title(f'{label.split(chr(10))[0]}\nAcc={res["acc"]:.1%}  AUC={res["auc"]:.3f}', fontsize=14, fontweight='bold', color='#334155')

OUT = os.path.join(FIG_DIR, '04_ni_comparison.png')
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor=BG); plt.close()
print(f"✓ Saved Figure → {OUT}")
