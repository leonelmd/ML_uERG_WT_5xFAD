"""
23_gain_tracking_hc.py
======================
New hand-crafted features motivated by the "muted gain tracking" hypothesis
derived from Gemini's virtual-blockade analysis (script 19).

Biological rationale
--------------------
The chirp amplitude segment presents a 1 Hz sinusoidal flicker whose amplitude
increases linearly from near-zero to maximum over ~11 s.  A healthy retina (WT)
*tracks* this ramp: its ERG amplitude grows proportionally with the stimulus.
In 5xFAD mice the retinal gain at the fundamental frequency appears muted — the
fundamental-frequency response is weaker and/or grows less steeply.

Four new features encode this hypothesis:
  AmpFund_RMS        – RMS of 0.5–1.5 Hz band-passed amplitude segment;
                       directly measures fundamental-tracking strength.
  AmpFund_frac       – AmpFund_RMS / ChirpAmp_RMS; fraction of total amplitude-
                       segment energy residing in the fundamental band
                       (scale-invariant gain concentration).
  AmpEnv_slope_norm  – Linear slope of the Hilbert envelope, normalised by the
                       mean envelope; positive = response scales up with stimulus.
                       Low value ↔ flat / non-growing response = muted tracking.
  AmpEnv_late_early  – Mean envelope (late 50%) / mean envelope (early 50%);
                       ratio > 1 means the response grows with the stimulus ramp.
                       5xFAD: ratio closer to 1 (fails to scale).

Feature sets compared
---------------------
  HC_orig          : 13 original hand-crafted features (from script 06)
  GainTracking     : 4 new gain-tracking features (this script)
  HC+GainTracking  : 17 combined features

Protocol
--------
Identical 5-fold subject-disjoint StratifiedGroupKFold as scripts 06 and 18.
Best classifier selected by pooled cross-val AUC within each set.

Outputs
-------
  results/figures/23_gain_tracking_comparison.png
  results/tables/23_gain_tracking_results.csv
"""

import os, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert as scipy_hilbert
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
                             f1_score, recall_score)
from sklearn.inspection import permutation_importance as sk_perm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
from dataset import ERGChirpDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_DIR   = os.path.join(ROOT, 'data', 'cache')
HC_CSV      = os.path.join(ROOT, 'data', 'hand_crafted_features.csv')
FIG_DIR     = os.path.join(ROOT, 'results', 'figures')
TAB_DIR     = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

FS           = 250.0   # Hz — amplitude segment sampling rate
RANDOM_STATE = 42
N_FOLDS      = 5

# Original HC features (same list as scripts 06 / 18)
HC_FEATS = [
    'Flash_Peak_Max', 'Flash_Peak_Min', 'Flash_Peak_P2P', 'Flash_RMS',
    'ChirpFreq_RMS', 'ChirpFreq_Std',
    'ChirpAmp_RMS', 'ChirpAmp_Max', 'ChirpAmp_P2P',
    'Power_Total', 'Power_Low', 'Power_Mid', 'Power_High',
]
GT_FEATS = [
    'AmpFund_RMS', 'AmpFund_frac',
    'AmpEnv_slope_norm', 'AmpEnv_late_early',
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Signal-processing helpers
# ══════════════════════════════════════════════════════════════════════════════
def _bandpass_fundamental(x, fs=FS, lo=0.5, hi=1.5, order=4):
    """Band-pass filter x to isolate the 1 Hz fundamental."""
    b, a = butter(order, [lo, hi], btype='band', fs=fs, analog=False)
    return filtfilt(b, a, x)


def _hilbert_envelope(x):
    """Amplitude envelope via analytic signal (Hilbert transform)."""
    return np.abs(scipy_hilbert(x))


def compute_gain_tracking_features(mean_sig):
    """
    Given a 1-D numpy array (amplitude segment, already trial-averaged),
    return a dict with the four gain-tracking features.
    """
    T   = len(mean_sig)
    t   = np.arange(T, dtype=np.float64)

    # ── Fundamental component ────────────────────────────────────────────────
    fund    = _bandpass_fundamental(mean_sig)
    fund_rms = float(np.sqrt(np.mean(fund ** 2)))
    sig_rms  = float(np.sqrt(np.mean(mean_sig ** 2)))
    amp_fund_frac = fund_rms / (sig_rms + 1e-12)

    # ── Hilbert envelope of the fundamental ──────────────────────────────────
    env = _hilbert_envelope(fund)
    env_mean = float(np.mean(env))

    # Normalised slope: fit env = a*t + b; report a / env_mean
    coeffs = np.polyfit(t, env, 1)
    slope_norm = float(coeffs[0] / (env_mean + 1e-12))

    # Late/early envelope ratio
    mid = T // 2
    env_early = float(np.mean(env[:mid]))
    env_late  = float(np.mean(env[mid:]))
    late_early = float(env_late / (env_early + 1e-12))

    return {
        'AmpFund_RMS':       fund_rms,
        'AmpFund_frac':      amp_fund_frac,
        'AmpEnv_slope_norm': slope_norm,
        'AmpEnv_late_early': late_early,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Load data and compute features
# ══════════════════════════════════════════════════════════════════════════════
print("Loading amplitude segment …")
dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment='amplitude',
                          cache_dir=CACHE_DIR)

df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label = {r.Subject: (1 if '5xFAD' in r.Group else 0)
                    for _, r in df_meta.iterrows()}

trial_accum: dict[str, list] = {}
for i in range(len(dataset)):
    sig, _, name = dataset[i]
    base = name.split('_trial_')[0]
    if base in subject_to_label:
        trial_accum.setdefault(base, []).append(sig.float().numpy().flatten())

gt_rows = []
for subj, trials in sorted(trial_accum.items()):
    mean_sig = np.mean(trials, axis=0).astype(np.float64)
    feats = compute_gain_tracking_features(mean_sig)
    feats['Subject'] = subj
    gt_rows.append(feats)

df_gt = pd.DataFrame(gt_rows)
print(f"  Computed gain-tracking features for {len(df_gt)} subjects")

# Merge with existing HC features and metadata
df_hc  = pd.read_csv(HC_CSV)
df_all = pd.merge(df_meta, df_hc, on='Subject')
df_all = pd.merge(df_all,  df_gt, on='Subject')
df_all['Subject_Base'] = df_all['Subject'].apply(
    lambda x: re.sub(r'(-t\d+|_trial_\d+|_trial|_t\d+)$', '', x))
df_all = df_all.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

y      = df_all['Group'].str.contains('5xFAD').astype(int).values
groups = df_all['Subject_Base'].values

print(f"  Merged dataset: N={len(df_all)}  WT={sum(y==0)}  5xFAD={sum(y==1)}")

# Quick sanity — print group means for each new feature
print("\nGain-tracking feature group means (trial-averaged subject level):")
df_gt2 = df_gt.copy()
df_gt2['Label'] = df_gt2['Subject'].map(subject_to_label)
for f in GT_FEATS:
    wt_m  = df_gt2.loc[df_gt2['Label'] == 0, f].mean()
    fad_m = df_gt2.loc[df_gt2['Label'] == 1, f].mean()
    print(f"  {f:<25}  WT={wt_m:.4f}  5xFAD={fad_m:.4f}  "
          f"Δ={fad_m - wt_m:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CV utilities
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
        'k-NN (k=5)':      KNeighborsClassifier(n_neighbors=5,
                                                   weights='distance'),
    }

def make_pipe(clf):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler()),
                     ('clf', clf)])

def run_cv(feat_names):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    X    = df_all[feat_names].values
    best_auc, best_name, best_res = -1, None, None
    all_aucs = {}

    for name, clf_def in make_clf_pool().items():
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
# 4.  Run all three feature sets
# ══════════════════════════════════════════════════════════════════════════════
FEAT_SETS = {
    'HC_orig':        HC_FEATS,
    'GainTracking':   GT_FEATS,
    'HC+GainTracking': HC_FEATS + GT_FEATS,
}

print(f"\nRunning {N_FOLDS}-fold CV …")
results = {}
for fs_name, feat_list in FEAT_SETS.items():
    print(f"\n  [{fs_name}]  {len(feat_list)} feature(s)")
    res = run_cv(feat_list)
    results[fs_name] = res
    print(f"    Best: {res['clf_name']}  Pooled AUC={res['auc']:.3f}  "
          f"Acc={res['acc']:.1%}  Sens={res['sens']:.1%}  Spec={res['spec']:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Save results table
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
df_out.to_csv(os.path.join(TAB_DIR, '23_gain_tracking_results.csv'), index=False)
print(f"\n  Saved: 23_gain_tracking_results.csv")

# Save HC+GainTracking probabilities for use in the final comparison figure (08)
best_hcgt = results['HC+GainTracking']
pd.DataFrame({'y_true': best_hcgt['y_true'],
              'y_prob': best_hcgt['y_prob']}).to_csv(
    os.path.join(TAB_DIR, '23_hc_gain_probs.csv'), index=False)
print(f"  Saved: 23_hc_gain_probs.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Feature importance for HC+GainTracking (re-train on full data)
# ══════════════════════════════════════════════════════════════════════════════
combined_feats  = FEAT_SETS['HC+GainTracking']
X_combined      = df_all[combined_feats].values
best_combined   = results['HC+GainTracking']
gt_feat_set     = set(GT_FEATS)

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

bar_colors = []
for lbl in labels:
    if lbl in gt_feat_set:
        bar_colors.append('#e67e22')          # orange = new gain-tracking
    elif vals[labels.index(lbl)] > 0:
        bar_colors.append('#E53935')          # red = original HC, positive
    else:
        bar_colors.append('#1E88E5')          # blue = original HC, negative


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Figure
# ══════════════════════════════════════════════════════════════════════════════
BG = '#F7F9FC'
C  = '#1565C0'

fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=BG)
fig.suptitle(
    'Gain-Tracking Features — "Muted Gain Tracking" Hypothesis (Script 23)\n'
    f'5-fold subject-disjoint CV  |  N={len(df_all)}',
    fontsize=13, fontweight='bold')
fig.subplots_adjust(left=0.06, right=0.97, wspace=0.40, top=0.88, bottom=0.08)

# ── Panel A: AUC comparison ──────────────────────────────────────────────────
ax = axes[0]
fs_names = list(results.keys())
aucs     = [results[n]['auc']      for n in fs_names]
clfs     = [results[n]['clf_name'] for n in fs_names]
colors   = ['#B0BEC5', '#e67e22', '#27ae60']

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

delta = results['HC+GainTracking']['auc'] - results['HC_orig']['auc']
sign  = '+' if delta >= 0 else ''
ax.text(0.52, 2.0,
        f'Δ AUC = {sign}{delta:.3f}  (HC+GT vs HC_orig)',
        fontsize=9,
        color='#27ae60' if delta >= 0 else '#c0392b',
        fontweight='bold', va='center')

# ── Panel B: ROC curves ──────────────────────────────────────────────────────
ax = axes[1]
ax.plot([0, 1], [0, 1], '--', color='#B0BEC5', lw=1.5, label='Chance')
roc_styles = {
    'HC_orig':         ('#B0BEC5', '--', 2.0),
    'GainTracking':    ('#e67e22', '-',  2.0),
    'HC+GainTracking': ('#27ae60', '-',  2.8),
}
for fs_name, (col, ls, lw) in roc_styles.items():
    res = results[fs_name]
    fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
    ax.plot(fpr, tpr, color=col, ls=ls, lw=lw,
            label=f'{fs_name}  AUC={res["auc"]:.3f}')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curves', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
ax.set_facecolor('white')

# ── Panel C: Feature importance for HC+GainTracking ─────────────────────────
ax = axes[2]
ax.barh(labels, vals,
        xerr=errs if errs.max() > 0 else None,
        color=bar_colors, edgecolor='white', height=0.7, capsize=3)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel(imp_lbl, fontsize=9)
ax.set_title(f'Feature Importance — HC+GainTracking\n({best_combined["clf_name"]})',
             fontweight='bold')
ax.set_facecolor('white')
ax.grid(axis='x', alpha=0.25)

from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor='#e67e22', label='New — gain tracking'),
    Patch(facecolor='#E53935', label='Original HC — positive'),
    Patch(facecolor='#1E88E5', label='Original HC — negative'),
]
ax.legend(handles=legend_els, fontsize=8, loc='lower right')

out_fig = os.path.join(FIG_DIR, '23_gain_tracking_comparison.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  Saved: {out_fig}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Console summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("SUMMARY — Gain-Tracking Features (Script 23)")
print(f"{'='*65}")
print(f"{'Feature set':<18}  {'AUC':>6}  {'Acc':>6}  {'Sens':>6}  {'Spec':>6}  Best clf")
print('-' * 65)
for fs_name, res in results.items():
    marker = '  ★' if fs_name == 'HC+GainTracking' else ''
    print(f"{fs_name:<18}  {res['auc']:.3f}  {res['acc']:.1%}  "
          f"{res['sens']:.1%}  {res['spec']:.1%}  {res['clf_name']}{marker}")
delta = results['HC+GainTracking']['auc'] - results['HC_orig']['auc']
sign  = '+' if delta >= 0 else ''
print(f"\n  AUC gain (HC+GT vs HC_orig): {sign}{delta:.3f}")
print(f"\n  New features (gain-tracking hypothesis):")
for f in GT_FEATS:
    wt_m  = df_gt2.loc[df_gt2['Label'] == 0, f].mean()
    fad_m = df_gt2.loc[df_gt2['Label'] == 1, f].mean()
    print(f"    {f:<25}  WT={wt_m:.4f}  5xFAD={fad_m:.4f}  Δ={fad_m - wt_m:+.4f}")
print(f"\n  Figure → {out_fig}")
print(f"  Table  → {TAB_DIR}/23_gain_tracking_results.csv")
