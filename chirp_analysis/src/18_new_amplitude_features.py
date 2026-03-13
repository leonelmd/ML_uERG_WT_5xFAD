"""
18_new_amplitude_features.py
============================
Incorporates the two strongest amplitude-segment windows discovered by script 17
into the hand-crafted feature analysis (script 06).

New features (derived from script 17 — smoothed |d| ≥ 0.5, top-2 by peak |d|)
-------------------------------------------------------------------------------
  AmpSeg_W1_mean  : mean amplitude in [8.36–8.53 s]  (peak |d| = 1.33, WT > 5xFAD)
  AmpSeg_W2_mean  : mean amplitude in [9.35–9.60 s]  (peak |d| = 1.20, WT > 5xFAD)

Feature sets compared
---------------------
  HC_orig   : 13 original hand-crafted features from script 06
  New_only  : 2 new amplitude window features
  HC+New    : 15 combined features

Protocol
--------
Identical 5-fold subject-disjoint CV and classifier pool as script 06.
Best classifier selected by pooled AUC within each feature set.

Outputs
-------
  results/figures/18_new_features_comparison.png
  results/tables/18_new_features_results.csv
"""

import os, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             f1_score, recall_score, confusion_matrix)
from sklearn.inspection import permutation_importance as sk_perm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
from dataset import ERGChirpDataset

# ── Paths ──────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_DIR   = os.path.join(ROOT, 'data', 'cache')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
HC_CSV      = os.path.join(ROOT, 'data', 'hand_crafted_features.csv')
FIG_DIR     = os.path.join(ROOT, 'results', 'figures')
TAB_DIR     = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

FS           = 250.0   # Hz — amplitude segment sample rate
RANDOM_STATE = 42
N_FOLDS      = 5

# New feature windows (from script 17, smoothed |d| ≥ 0.5, top-2 by peak |d|)
W1 = (8.36, 8.53)   # 172 ms, WT > 5xFAD, peak |d| = 1.33
W2 = (9.35, 9.60)   # 252 ms, WT > 5xFAD, peak |d| = 1.20

# Original HC features (same list as script 06)
HC_FEATS = [
    'Flash_Peak_Max', 'Flash_Peak_Min', 'Flash_Peak_P2P', 'Flash_RMS',
    'ChirpFreq_RMS', 'ChirpFreq_Std',
    'ChirpAmp_RMS', 'ChirpAmp_Max', 'ChirpAmp_P2P',
    'Power_Total', 'Power_Low', 'Power_Mid', 'Power_High',
]
NEW_FEATS = ['AmpSeg_W1_mean', 'AmpSeg_W2_mean']


# ══════════════════════════════════════════════════════════════════════════════
# 1. Compute new features from raw amplitude segment signals
# ══════════════════════════════════════════════════════════════════════════════
print("Loading amplitude segment …")
dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment='amplitude',
                          cache_dir=CACHE_DIR)

df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label   = {r.Subject: (1 if '5xFAD' in r.Group else 0)
                      for _, r in df_meta.iterrows()}

trial_accum: dict[str, list] = {}
for i in range(len(dataset)):
    sig, _, name = dataset[i]
    base = name.split('_trial_')[0]
    if base in subject_to_label:
        trial_accum.setdefault(base, []).append(sig.float().numpy().flatten())

# Convert to sample indices for the two windows
w1_s, w1_e = int(W1[0] * FS), int(W1[1] * FS)
w2_s, w2_e = int(W2[0] * FS), int(W2[1] * FS)
print(f"  Window 1: [{W1[0]:.2f}–{W1[1]:.2f} s]  samples [{w1_s}:{w1_e}]  "
      f"({w1_e - w1_s} samples = {(w1_e-w1_s)/FS*1000:.0f} ms)")
print(f"  Window 2: [{W2[0]:.2f}–{W2[1]:.2f} s]  samples [{w2_s}:{w2_e}]  "
      f"({w2_e - w2_s} samples = {(w2_e-w2_s)/FS*1000:.0f} ms)")

new_feat_rows = []
for subj, trials in sorted(trial_accum.items()):
    mean_sig = np.mean(trials, axis=0).astype(np.float32)
    new_feat_rows.append({
        'Subject':        subj,
        'AmpSeg_W1_mean': float(mean_sig[w1_s:w1_e].mean()),
        'AmpSeg_W2_mean': float(mean_sig[w2_s:w2_e].mean()),
    })

df_new = pd.DataFrame(new_feat_rows)
print(f"  Computed new features for {len(df_new)} subjects")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Merge with existing HC features and metadata
# ══════════════════════════════════════════════════════════════════════════════
df_hc   = pd.read_csv(HC_CSV)
df_all  = pd.merge(df_meta, df_hc,  on='Subject')
df_all  = pd.merge(df_all,  df_new, on='Subject')
df_all['Subject_Base'] = df_all['Subject'].apply(
    lambda x: re.sub(r'(-t\d+|_trial_\d+|_trial|_t\d+)$', '', x))
df_all = df_all.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

y      = df_all['Group'].str.contains('5xFAD').astype(int).values
groups = df_all['Subject_Base'].values

print(f"  Merged dataset: N={len(df_all)}  WT={sum(y==0)}  5xFAD={sum(y==1)}")

FEAT_SETS = {
    'HC_orig':  HC_FEATS,
    'New_only': NEW_FEATS,
    'HC+New':   HC_FEATS + NEW_FEATS,
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. CV utilities (mirrors script 06)
# ══════════════════════════════════════════════════════════════════════════════
def make_clf_pool():
    return {
        'SVM (RBF)':       CalibratedClassifierCV(
                               SVC(kernel='rbf', C=1., gamma='scale'),
                               method='sigmoid', cv=5),
        'Random Forest':   RandomForestClassifier(300, max_depth=5,
                                                   random_state=RANDOM_STATE),
        'Log. Regression': LogisticRegression(C=0.1, max_iter=1000,
                                               solver='lbfgs'),
        'LDA':             LinearDiscriminantAnalysis(),
        'k-NN (k=5)':      KNeighborsClassifier(n_neighbors=5),
    }

def make_pipe(clf):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler()),
                     ('clf', clf)])

def run_cv(feat_names):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    X = df_all[feat_names].values
    clf_pool = make_clf_pool()

    best_auc, best_name, best_res = -1, None, None
    all_aucs = {}

    for name, clf_def in clf_pool.items():
        pipe = make_pipe(clf_def)
        ayt_list, ayp_list = [], []
        for tr_idx, vl_idx in sgkf.split(X, y, groups=groups):
            pipe.fit(X[tr_idx], y[tr_idx])
            ayt_list.extend(y[vl_idx])
            ayp_list.extend(pipe.predict_proba(X[vl_idx])[:, 1])

        ayt = np.array(ayt_list)
        ayp = np.array(ayp_list)
        pooled_auc = roc_auc_score(ayt, ayp)
        if pooled_auc < 0.5:
            ayp = 1. - ayp; pooled_auc = 1. - pooled_auc
        aypr = (ayp >= 0.5).astype(int)
        all_aucs[name] = pooled_auc

        if pooled_auc > best_auc:
            best_auc  = pooled_auc
            best_name = name
            best_res  = dict(y_true=ayt, y_prob=ayp, y_pred=aypr,
                             auc=pooled_auc,
                             acc=float(accuracy_score(ayt, aypr)),
                             f1=float(f1_score(ayt, aypr, average='macro',
                                               zero_division=0)),
                             sens=float(recall_score(ayt, aypr, pos_label=1,
                                                     zero_division=0)),
                             spec=float(recall_score(ayt, aypr, pos_label=0,
                                                     zero_division=0)))
    best_res['clf_name'] = best_name
    best_res['all_aucs'] = all_aucs
    return best_res


# ══════════════════════════════════════════════════════════════════════════════
# 4. Run all three feature sets
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning {N_FOLDS}-fold CV for each feature set …")
results = {}
for fs_name, feat_list in FEAT_SETS.items():
    print(f"\n  [{fs_name}]  {len(feat_list)} feature(s)")
    res = run_cv(feat_list)
    results[fs_name] = res
    print(f"    Best: {res['clf_name']}  Pooled AUC={res['auc']:.3f}  "
          f"Acc={res['acc']:.1%}  Sens={res['sens']:.1%}  Spec={res['spec']:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Save results table
# ══════════════════════════════════════════════════════════════════════════════
rows = []
for fs_name, res in results.items():
    rows.append({
        'feature_set': fs_name,
        'n_features':  len(FEAT_SETS[fs_name]),
        'best_clf':    res['clf_name'],
        'auc':         round(res['auc'],  3),
        'acc':         round(res['acc'],  3),
        'f1':          round(res['f1'],   3),
        'sens':        round(res['sens'], 3),
        'spec':        round(res['spec'], 3),
    })
df_out = pd.DataFrame(rows)
df_out.to_csv(os.path.join(TAB_DIR, '18_new_features_results.csv'), index=False)
print(f"\n  Saved: 18_new_features_results.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Feature importance for HC+New (re-train on full data)
# ══════════════════════════════════════════════════════════════════════════════
combined_feats = FEAT_SETS['HC+New']
X_combined     = df_all[combined_feats].values
best_combined  = results['HC+New']

clf_pool_fi = make_clf_pool()
best_pipe_fi = make_pipe(clf_pool_fi[best_combined['clf_name']])
best_pipe_fi.fit(X_combined, y)

clf_obj = best_pipe_fi.named_steps['clf']
if best_combined['clf_name'] in ('Log. Regression', 'LDA'):
    imp   = clf_obj.coef_[0]
    imp_e = np.zeros(len(imp))
    imp_lbl = 'Coefficient  (red → 5xFAD,  blue → WT)'
elif best_combined['clf_name'] == 'Random Forest':
    imp   = clf_obj.feature_importances_
    imp_e = np.zeros(len(imp))
    imp_lbl = 'Gini importance'
else:
    perm  = sk_perm(best_pipe_fi, X_combined, y, n_repeats=50,
                    scoring='roc_auc', random_state=RANDOM_STATE)
    imp   = perm.importances_mean
    imp_e = perm.importances_std
    imp_lbl = 'Permutation importance (AUC)'

order  = np.argsort(np.abs(imp))
labels = [combined_feats[i] for i in order]
vals   = imp[order]
errs   = imp_e[order]

# Highlight new features in the importance chart
new_feat_set = set(NEW_FEATS)
bar_colors   = []
for lbl in labels:
    if lbl in new_feat_set:
        bar_colors.append('#e67e22')   # orange = new
    elif vals[labels.index(lbl)] > 0:
        bar_colors.append('#E53935')   # red = original, positive
    else:
        bar_colors.append('#1E88E5')   # blue = original, negative


# ══════════════════════════════════════════════════════════════════════════════
# 7. Figure
# ══════════════════════════════════════════════════════════════════════════════
BG = '#F7F9FC'
C  = '#1565C0'

fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=BG)
fig.suptitle('Amplitude Window Features — Comparison with Original Hand-Crafted Features\n'
             f'5-fold subject-disjoint CV  |  N={len(df_all)}',
             fontsize=13, fontweight='bold')
fig.subplots_adjust(left=0.06, right=0.97, wspace=0.38, top=0.88, bottom=0.08)

# ── Panel A: AUC comparison across feature sets ────────────────────────────
ax = axes[0]
fs_names = list(results.keys())
aucs     = [results[n]['auc']      for n in fs_names]
clfs     = [results[n]['clf_name'] for n in fs_names]
colors   = ['#B0BEC5', '#e67e22', '#27ae60']   # gray / orange / green

bars = ax.barh(fs_names, aucs, color=colors, edgecolor='white', height=0.5)
ax.axvline(0.5, color='red', ls='--', lw=1.2, label='Chance (0.5)')
ax.set_xlim(0, 1.05)
ax.set_xlabel('Pooled Cross-val AUC', fontsize=11)
ax.set_title('AUC by Feature Set\n(best classifier per set)', fontweight='bold')
ax.set_facecolor('white')
ax.grid(axis='x', alpha=0.25)
for bar, v, clf in zip(bars, aucs, clfs):
    ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2.,
            f'{v:.3f}  [{clf}]', va='center', fontsize=9)
ax.legend(fontsize=9)

# Annotate gain of HC+New vs HC_orig
delta = results['HC+New']['auc'] - results['HC_orig']['auc']
sign  = '+' if delta >= 0 else ''
ax.text(0.52, 2.0, f'Δ AUC = {sign}{delta:.3f}  (HC+New vs HC_orig)',
        fontsize=9, color='#27ae60' if delta >= 0 else '#c0392b',
        fontweight='bold', va='center')

# ── Panel B: ROC curves for HC_orig and HC+New ────────────────────────────
ax = axes[1]
ax.plot([0, 1], [0, 1], '--', color='#B0BEC5', lw=1.5, label='Chance')
roc_styles = {'HC_orig': ('#B0BEC5', '--', 2.0),
              'HC+New':  ('#27ae60', '-',  2.8)}
for fs_name, (col, ls, lw) in roc_styles.items():
    res = results[fs_name]
    fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
    ax.plot(fpr, tpr, color=col, ls=ls, lw=lw,
            label=f'{fs_name}  AUC={res["auc"]:.3f}')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curves\n(HC_orig vs HC+New)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
ax.set_facecolor('white')

# ── Panel C: Feature importance for HC+New ────────────────────────────────
ax = axes[2]
ax.barh(labels, vals,
        xerr=errs if errs.max() > 0 else None,
        color=bar_colors, edgecolor='white', height=0.7, capsize=3)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel(imp_lbl, fontsize=9)
ax.set_title(f'Feature Importance — HC+New\n({best_combined["clf_name"]})',
             fontweight='bold')
ax.set_facecolor('white')
ax.grid(axis='x', alpha=0.25)

# Legend for new vs original features
from matplotlib.patches import Patch
legend_els = [Patch(facecolor='#e67e22', label='New (amplitude window)'),
              Patch(facecolor='#E53935', label='Original HC — positive'),
              Patch(facecolor='#1E88E5', label='Original HC — negative')]
ax.legend(handles=legend_els, fontsize=8, loc='lower right')

out = os.path.join(FIG_DIR, '18_new_features_comparison.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Console summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY — New Amplitude Window Features")
print(f"{'='*60}")
print(f"{'Feature set':<12}  {'AUC':>6}  {'Acc':>6}  {'Sens':>6}  {'Spec':>6}  Best clf")
print('-' * 60)
for fs_name, res in results.items():
    marker = '  ★' if fs_name == 'HC+New' else ''
    print(f"{fs_name:<12}  {res['auc']:.3f}  {res['acc']:.1%}  "
          f"{res['sens']:.1%}  {res['spec']:.1%}  {res['clf_name']}{marker}")
print(f"\n  New features (window locations from script 17):")
print(f"    AmpSeg_W1_mean:  mean amplitude in [{W1[0]:.2f}–{W1[1]:.2f} s]  "
      f"(peak |d|=1.33, WT > 5xFAD)")
print(f"    AmpSeg_W2_mean:  mean amplitude in [{W2[0]:.2f}–{W2[1]:.2f} s]  "
      f"(peak |d|=1.20, WT > 5xFAD)")
delta = results['HC+New']['auc'] - results['HC_orig']['auc']
sign  = '+' if delta >= 0 else ''
print(f"\n  AUC gain (HC+New vs HC_orig): {sign}{delta:.3f}")
print(f"\n  Figure → {FIG_DIR}/18_new_features_comparison.png")
print(f"  Table  → {TAB_DIR}/18_new_features_results.csv")
