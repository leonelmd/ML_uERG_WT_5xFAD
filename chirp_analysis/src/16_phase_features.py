"""
16_phase_features.py
====================
Phase-shift features for WT vs 5xFAD classification.

Motivation
----------
Script 15 (Bayesian optimisation) revealed a visible phase shift between the
CNN-preferred WT and 5xFAD stimuli in the amplitude segment.  The existing
hand-crafted feature set (script 06) is purely amplitude/energy-based
(RMS, P2P, power bands) and captures no temporal phase information.

This script:
  1. Quantifies the phase shift between group means and across individual subjects.
  2. Extracts six phase features per subject.
  3. Compares classification performance:
       Phase-only  vs  Existing HC  vs  Phase + HC combined.

Phase features
--------------
  xcorr_lag       Cross-correlation lag with fold-mean WT template (fold-aware,
                  leakage-free). Positive = subject lags behind WT mean.
  xcorr_val       Peak cross-correlation value (shape similarity to WT mean).
  hilbert_slope   Mean instantaneous frequency (slope of unwrapped Hilbert phase).
  hilbert_std     Residual phase variability after trend removal.
  fft_peak_freq   Dominant spectral frequency (Hz).
  fft_peak_phase  Phase angle (radians) at dominant frequency.

All signals are z-scored before feature extraction so results reflect shape,
not amplitude (mirroring the InstanceNorm1d in the CNN).

Outputs
-------
  results/figures/16_a_phase_motivation.png
  results/figures/16_b_feature_distributions.png
  results/figures/16_c_classification.png
  results/tables/16_phase_features.csv
  results/tables/16_classification_results.csv

Usage (from chirp_analysis/ folder)
-------------------------------------
    python src/16_phase_features.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert, correlate
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

FIG_DIR = os.path.join(ROOT, 'results', 'figures')
TAB_DIR = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

N_FOLDS      = 5
RANDOM_STATE = 42
XCORR_MAX_LAG = 200   # ±200 samples = ±800 ms at 250 Hz
FS           = 250.0  # sampling rate (Hz) for amplitude segment

# ── Colours ───────────────────────────────────────────────────────────────────
C_WT  = '#2980b9'
C_FAD = '#c0392b'


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load dataset and aggregate trials to subject-level mean signals
# ══════════════════════════════════════════════════════════════════════════════
print("Loading dataset …")
dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment='amplitude',
                          cache_dir=CACHE_DIR)

df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label   = {r.Subject: (1 if '5xFAD' in r.Group else 0)
                      for _, r in df_meta.iterrows()}

# Accumulate trials per subject
trial_accum: dict[str, list] = {}
for i in range(len(dataset)):
    sig, _, name = dataset[i]
    base = name.split('_trial_')[0]
    if base not in subject_to_label:
        continue
    arr = sig.float().numpy().flatten()
    trial_accum.setdefault(base, []).append(arr)

# Build subject-level list: (subject, mean_signal, binary_label)
subjects_data = []
for subj, trials in sorted(trial_accum.items()):
    mean_sig = np.mean(trials, axis=0).astype(np.float32)
    subjects_data.append({
        'subject': subj,
        'signal':  mean_sig,
        'label':   subject_to_label[subj],   # 0=WT, 1=5xFAD
    })

subjects = [d['subject'] for d in subjects_data]
signals  = np.array([d['signal']  for d in subjects_data], dtype=np.float32)  # [N, T]
labels   = np.array([d['label']   for d in subjects_data], dtype=np.int32)

N, T = signals.shape
t    = np.arange(T)
print(f"  Subjects: {N}  (WT={sum(labels==0)}, 5xFAD={sum(labels==1)})  T={T}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

def z_score(sig: np.ndarray) -> np.ndarray:
    return (sig - sig.mean()) / (sig.std() + 1e-8)


def compute_xcorr(signal: np.ndarray, template: np.ndarray,
                  max_lag: int = XCORR_MAX_LAG):
    """Normalised cross-correlation → (lag_at_peak, peak_value)."""
    s = z_score(signal)
    t = z_score(template)
    cc   = correlate(s, t, mode='full')        # length 2N-1
    cc  /= (len(t) + 1e-8)                     # normalise by template length
    lags = np.arange(-(len(t) - 1), len(s))
    mask = np.abs(lags) <= max_lag
    best = np.argmax(cc[mask])
    return float(lags[mask][best]), float(cc[mask][best])


def compute_hilbert_phase(signal: np.ndarray):
    """Return (slope, residual_std) of unwrapped instantaneous phase."""
    analytic     = hilbert(z_score(signal))
    inst_phase   = np.unwrap(np.angle(analytic))
    tt           = np.arange(len(signal))
    slope, intercept = np.polyfit(tt, inst_phase, 1)
    residual_std = float(np.std(inst_phase - (slope * tt + intercept)))
    return float(slope), residual_std


def compute_fft_features(signal: np.ndarray, fs: float = FS):
    """Return (peak_freq_Hz, phase_at_peak_rad)."""
    n     = len(signal)
    fft   = np.fft.rfft(z_score(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # skip DC
    mags   = np.abs(fft[1:])
    phases = np.angle(fft[1:])
    freqs  = freqs[1:]
    pk     = int(np.argmax(mags))
    return float(freqs[pk]), float(phases[pk])


def compute_envelope_features(signal: np.ndarray):
    """
    Hilbert envelope features on the RAW (non-z-scored) signal.
    Captures how strongly and how consistently the retina follows the
    amplitude modulation — amplitude information the CNN cannot use
    (InstanceNorm), but that reflects retinal gain.

    Returns: (env_mean, env_slope, env_early, env_late, env_ratio, env_cv)
      env_mean  — mean envelope (overall following strength)
      env_slope — linear trend of envelope (+ = building, - = decaying)
      env_early — mean of first quartile (initial response)
      env_late  — mean of last quartile  (sustained response)
      env_ratio — env_late / env_early   (< 1 = adaptation/decay)
      env_cv    — coefficient of variation (variability of following)
    """
    envelope = np.abs(hilbert(signal.astype(np.float64)))
    n        = len(envelope)
    q        = n // 4
    tt       = np.arange(n, dtype=np.float64)

    env_mean  = float(envelope.mean())
    env_slope = float(np.polyfit(tt / n, envelope, 1)[0])   # normalised by length
    env_early = float(envelope[:q].mean())
    env_late  = float(envelope[3*q:].mean())
    env_ratio = float(env_late / (env_early + 1e-8))
    env_cv    = float(envelope.std() / (env_mean + 1e-8))

    return env_mean, env_slope, env_early, env_late, env_ratio, env_cv


# ══════════════════════════════════════════════════════════════════════════════
# 3. Fold-aware cross-correlation + leakage-free feature extraction
# ══════════════════════════════════════════════════════════════════════════════
print("\nExtracting phase features (fold-aware cross-correlation) …")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Storage for fold-aware xcorr features
xcorr_lag_arr = np.full(N, np.nan)
xcorr_val_arr = np.full(N, np.nan)

for fold, (tr_idx, vl_idx) in enumerate(skf.split(signals, labels), 1):
    # Template: mean z-scored WT signal from training subjects
    wt_tr_sigs  = signals[tr_idx][labels[tr_idx] == 0]
    template_wt = np.mean([z_score(s) for s in wt_tr_sigs], axis=0)

    for idx in vl_idx:
        lag, val          = compute_xcorr(signals[idx], template_wt)
        xcorr_lag_arr[idx] = lag
        xcorr_val_arr[idx] = val

    print(f"  Fold {fold}: {len(vl_idx)} val subjects  "
          f"(WT template from {len(wt_tr_sigs)} training WT subjects)")

# Fold-independent features: computed on all subjects upfront
hilbert_slope_arr = np.zeros(N, dtype=np.float64)
hilbert_std_arr   = np.zeros(N, dtype=np.float64)
fft_freq_arr      = np.zeros(N, dtype=np.float64)
fft_phase_arr     = np.zeros(N, dtype=np.float64)
env_mean_arr      = np.zeros(N, dtype=np.float64)
env_slope_arr     = np.zeros(N, dtype=np.float64)
env_early_arr     = np.zeros(N, dtype=np.float64)
env_late_arr      = np.zeros(N, dtype=np.float64)
env_ratio_arr     = np.zeros(N, dtype=np.float64)
env_cv_arr        = np.zeros(N, dtype=np.float64)

for i, sig in enumerate(signals):
    hilbert_slope_arr[i], hilbert_std_arr[i] = compute_hilbert_phase(sig)
    fft_freq_arr[i],      fft_phase_arr[i]   = compute_fft_features(sig)
    (env_mean_arr[i], env_slope_arr[i], env_early_arr[i],
     env_late_arr[i], env_ratio_arr[i], env_cv_arr[i])  = compute_envelope_features(sig)

# Pre-compute per-subject Hilbert envelopes for visualisation
all_envelopes   = np.array([np.abs(hilbert(s.astype(np.float64))) for s in signals])
wt_envelopes    = all_envelopes[labels == 0]
fad_envelopes   = all_envelopes[labels == 1]
mean_env_wt     = wt_envelopes.mean(axis=0)
mean_env_fad    = fad_envelopes.mean(axis=0)
se_env_wt       = wt_envelopes.std(axis=0)  / np.sqrt(len(wt_envelopes))
se_env_fad      = fad_envelopes.std(axis=0) / np.sqrt(len(fad_envelopes))

# Collect into DataFrame
df_phase = pd.DataFrame({
    'subject'       : subjects,
    'label'         : labels,
    'group'         : ['5xFAD' if l == 1 else 'WT' for l in labels],
    'xcorr_lag'     : xcorr_lag_arr,
    'xcorr_val'     : xcorr_val_arr,
    'hilbert_slope' : hilbert_slope_arr,
    'hilbert_std'   : hilbert_std_arr,
    'fft_peak_freq' : fft_freq_arr,
    'fft_peak_phase': fft_phase_arr,
    'env_mean'      : env_mean_arr,
    'env_slope'     : env_slope_arr,
    'env_early'     : env_early_arr,
    'env_late'      : env_late_arr,
    'env_ratio'     : env_ratio_arr,
    'env_cv'        : env_cv_arr,
})

PHASE_FEATS    = ['xcorr_lag', 'xcorr_val',
                  'hilbert_slope', 'hilbert_std',
                  'fft_peak_freq', 'fft_peak_phase']
ENVELOPE_FEATS = ['env_mean', 'env_slope', 'env_early',
                  'env_late', 'env_ratio', 'env_cv']
ALL_FEATS      = PHASE_FEATS + ENVELOPE_FEATS

df_phase.to_csv(os.path.join(TAB_DIR, '16_phase_features.csv'), index=False)
print(f"  Saved: 16_phase_features.csv  ({N} subjects × {len(ALL_FEATS)} features)")

# Quick group stats
print("\n  Phase features:")
for feat in PHASE_FEATS:
    wt_v  = df_phase.loc[df_phase.label==0, feat]
    fad_v = df_phase.loc[df_phase.label==1, feat]
    d = (fad_v.mean()-wt_v.mean()) / (np.sqrt((wt_v.std()**2+fad_v.std()**2)/2)+1e-8)
    print(f"  {feat:>15s}  WT={wt_v.mean():+.4f}±{wt_v.std():.4f}  "
          f"5xFAD={fad_v.mean():+.4f}±{fad_v.std():.4f}  d={d:+.2f}")
print("  Envelope features:")
for feat in ENVELOPE_FEATS:
    wt_v  = df_phase.loc[df_phase.label==0, feat]
    fad_v = df_phase.loc[df_phase.label==1, feat]
    d = (fad_v.mean()-wt_v.mean()) / (np.sqrt((wt_v.std()**2+fad_v.std()**2)/2)+1e-8)
    print(f"  {feat:>15s}  WT={wt_v.mean():+.4f}±{wt_v.std():.4f}  "
          f"5xFAD={fad_v.mean():+.4f}±{fad_v.std():.4f}  d={d:+.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Classification: phase-only, existing-HC, combined
# ══════════════════════════════════════════════════════════════════════════════
print("\nRunning classification …")

# Load existing HC features (if available)
hc_ok = os.path.exists(HC_CSV)
if hc_ok:
    df_hc   = pd.read_csv(HC_CSV)
    df_hc['Subject'] = df_hc['Subject'].str.strip()
    HC_FEATS = [c for c in df_hc.columns if c != 'Subject']
    df_merged = pd.merge(df_phase, df_hc, left_on='subject', right_on='Subject', how='inner')
    print(f"  HC features loaded: {len(HC_FEATS)} features, {len(df_merged)} subjects matched")
else:
    df_merged = df_phase.copy()
    HC_FEATS  = []
    print("  hand_crafted_features.csv not found — HC comparison skipped")

def make_pipe(C=0.1):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(C=C, max_iter=1000, solver='lbfgs')),
    ])

def cv_auc(X: np.ndarray, y: np.ndarray,
           pipe, skf: StratifiedKFold) -> float:
    """Pooled cross-validated AUC."""
    all_y, all_p = [], []
    for tr, vl in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        all_y.extend(y[vl])
        all_p.extend(pipe.predict_proba(X[vl])[:, 1])
    auc = roc_auc_score(all_y, all_p)
    return max(auc, 1 - auc)   # canonical ≥ 0.5

y_all    = df_phase['label'].values
X_phase  = df_phase[PHASE_FEATS].values
X_env    = df_phase[ENVELOPE_FEATS].values
X_both   = df_phase[ALL_FEATS].values

skf_eval = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

auc_phase = cv_auc(X_phase, y_all, make_pipe(), skf_eval)
auc_env   = cv_auc(X_env,   y_all, make_pipe(), skf_eval)
auc_both  = cv_auc(X_both,  y_all, make_pipe(), skf_eval)
print(f"  Phase-only       AUC = {auc_phase:.3f}")
print(f"  Envelope-only    AUC = {auc_env:.3f}")
print(f"  Phase + Envelope AUC = {auc_both:.3f}")

auc_hc, auc_hc_env, auc_hc_all = None, None, None
if hc_ok:
    y_m       = df_merged['label'].values
    X_hc      = df_merged[HC_FEATS].values
    X_hc_env  = df_merged[HC_FEATS + ENVELOPE_FEATS].values
    X_hc_all  = df_merged[HC_FEATS + ALL_FEATS].values

    auc_hc     = cv_auc(X_hc,     y_m, make_pipe(), skf_eval)
    auc_hc_env = cv_auc(X_hc_env, y_m, make_pipe(), skf_eval)
    auc_hc_all = cv_auc(X_hc_all, y_m, make_pipe(), skf_eval)
    print(f"  HC-only          AUC = {auc_hc:.3f}")
    print(f"  HC + Envelope    AUC = {auc_hc_env:.3f}")
    print(f"  HC + All         AUC = {auc_hc_all:.3f}")

# Save results table
rows_clf = [
    {'feature_set': 'Phase only',        'AUC': round(auc_phase, 4), 'n_features': len(PHASE_FEATS)},
    {'feature_set': 'Envelope only',     'AUC': round(auc_env,   4), 'n_features': len(ENVELOPE_FEATS)},
    {'feature_set': 'Phase + Envelope',  'AUC': round(auc_both,  4), 'n_features': len(ALL_FEATS)},
]
if auc_hc is not None:
    rows_clf += [
        {'feature_set': 'HC only',           'AUC': round(auc_hc,     4), 'n_features': len(HC_FEATS)},
        {'feature_set': 'HC + Envelope',     'AUC': round(auc_hc_env, 4), 'n_features': len(HC_FEATS)+len(ENVELOPE_FEATS)},
        {'feature_set': 'HC + All',          'AUC': round(auc_hc_all, 4), 'n_features': len(HC_FEATS)+len(ALL_FEATS)},
    ]
pd.DataFrame(rows_clf).to_csv(os.path.join(TAB_DIR, '16_classification_results.csv'), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Pre-compute group means / SE for signal panels
# ══════════════════════════════════════════════════════════════════════════════
wt_sigs  = signals[labels == 0]
fad_sigs = signals[labels == 1]
mean_wt  = wt_sigs.mean(axis=0)
mean_fad = fad_sigs.mean(axis=0)
se_wt    = wt_sigs.std(axis=0)  / np.sqrt(len(wt_sigs))
se_fad   = fad_sigs.std(axis=0) / np.sqrt(len(fad_sigs))

# Cross-correlation between group means (in z-scored space)
lags_full = np.arange(-(T - 1), T)
cc_means  = correlate(z_score(mean_fad), z_score(mean_wt), mode='full') / T
mask_lag  = np.abs(lags_full) <= XCORR_MAX_LAG
peak_lag_means = lags_full[mask_lag][np.argmax(cc_means[mask_lag])]
peak_val_means = cc_means[mask_lag].max()

print(f"\n  Phase shift between group means:")
print(f"    Peak xcorr lag = {peak_lag_means} samples = {peak_lag_means/FS*1000:.1f} ms")
print(f"    Peak xcorr value = {peak_val_means:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A: Phase-shift motivation
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure A: phase motivation …")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.subplots_adjust(wspace=0.35)

# Panel 0: group mean signals
ax = axes[0]
ax.fill_between(t, mean_wt  - se_wt,  mean_wt  + se_wt,  alpha=0.25, color=C_WT)
ax.fill_between(t, mean_fad - se_fad, mean_fad + se_fad, alpha=0.25, color=C_FAD)
ax.plot(t, mean_wt,  C_WT,  lw=1.8, label=f'Mean WT (n={len(wt_sigs)}, ±SE)')
ax.plot(t, mean_fad, C_FAD, lw=1.8, label=f'Mean 5xFAD (n={len(fad_sigs)}, ±SE)')
ax.set_xlabel('Sample index', fontsize=9)
ax.set_ylabel('Amplitude (norm.)', fontsize=9)
ax.set_title('A.  Group mean ERG signals\n(Chirp amplitude segment)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 1: cross-correlation of group means
ax = axes[1]
cc_plot  = cc_means[mask_lag]
lags_plot= lags_full[mask_lag]
ax.plot(lags_plot / FS * 1000, cc_plot, color='#8e44ad', lw=2)
ax.axvline(0, color='gray', ls=':', lw=1, alpha=0.6)
ax.axvline(peak_lag_means / FS * 1000, color='gold', ls='--', lw=2,
           label=f'Peak lag = {peak_lag_means:+d} smp\n'
                 f'= {peak_lag_means/FS*1000:+.1f} ms')
ax.axhline(peak_val_means, color='gold', ls=':', lw=1, alpha=0.5)
ax.scatter([peak_lag_means / FS * 1000], [peak_val_means],
           s=80, color='gold', zorder=5)
ax.set_xlabel('Lag (ms)', fontsize=9)
ax.set_ylabel('Normalised cross-correlation', fontsize=9)
ax.set_title('B.  Cross-correlation: mean WT vs mean 5xFAD\n'
             '(positive lag = 5xFAD leads WT)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 2: per-subject xcorr_lag, violin + strip
ax = axes[2]
wt_lags  = df_phase.loc[df_phase.label == 0, 'xcorr_lag'].values
fad_lags = df_phase.loc[df_phase.label == 1, 'xcorr_lag'].values

vp_wt  = ax.violinplot([wt_lags],  positions=[0], widths=0.6,
                        showmedians=True, showextrema=False)
vp_fad = ax.violinplot([fad_lags], positions=[1], widths=0.6,
                        showmedians=True, showextrema=False)
for pc in vp_wt['bodies']:  pc.set_facecolor(C_WT);  pc.set_alpha(0.55)
for pc in vp_fad['bodies']: pc.set_facecolor(C_FAD); pc.set_alpha(0.55)
vp_wt['cmedians'].set_color(C_WT);   vp_wt['cmedians'].set_lw(2)
vp_fad['cmedians'].set_color(C_FAD); vp_fad['cmedians'].set_lw(2)

rng = np.random.default_rng(RANDOM_STATE)
ax.scatter(rng.normal(0, 0.06, len(wt_lags)),  wt_lags,  s=20, color=C_WT,
           alpha=0.7, zorder=3)
ax.scatter(rng.normal(1, 0.06, len(fad_lags)), fad_lags, s=20, color=C_FAD,
           alpha=0.7, zorder=3)

ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(['WT', '5xFAD'], fontsize=10)
ax.set_ylabel('Cross-correlation lag (samples)', fontsize=9)
ax.set_title('C.  Per-subject lag with WT template\n'
             '(fold-aware, leakage-free)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Annotate means
for pos, vals, color in [(0, wt_lags, C_WT), (1, fad_lags, C_FAD)]:
    ax.text(pos, vals.max() + 3, f'μ={vals.mean():+.1f}', ha='center',
            fontsize=9, fontweight='bold', color=color)

plt.suptitle('Phase Shift Between WT and 5xFAD — Chirp Amplitude Segment\n'
             'Motivated by the CNN-optimal stimuli from script 15',
             fontweight='bold', fontsize=11)
plt.savefig(os.path.join(FIG_DIR, '16_a_phase_motivation.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 16_a_phase_motivation.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B: Feature distributions (all 6 phase features)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure B: feature distributions …")

FEAT_LABELS = {
    'xcorr_lag'     : 'Cross-corr lag\n(samples)',
    'xcorr_val'     : 'Cross-corr peak\n(normalised)',
    'hilbert_slope' : 'Hilbert phase slope\n(mean inst. freq.)',
    'hilbert_std'   : 'Hilbert phase std\n(residual)',
    'fft_peak_freq' : 'FFT peak freq\n(Hz)',
    'fft_peak_phase': 'FFT peak phase\n(rad)',
}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.subplots_adjust(hspace=0.50, wspace=0.35)

for ax, feat in zip(axes.ravel(), PHASE_FEATS):
    wt_v  = df_phase.loc[df_phase.label == 0, feat].values
    fad_v = df_phase.loc[df_phase.label == 1, feat].values

    vp_w = ax.violinplot([wt_v],  positions=[0], widths=0.5,
                          showmedians=True, showextrema=False)
    vp_f = ax.violinplot([fad_v], positions=[1], widths=0.5,
                          showmedians=True, showextrema=False)
    for pc in vp_w['bodies']:  pc.set_facecolor(C_WT);  pc.set_alpha(0.55)
    for pc in vp_f['bodies']:  pc.set_facecolor(C_FAD); pc.set_alpha(0.55)
    vp_w['cmedians'].set_color(C_WT);  vp_w['cmedians'].set_lw(2)
    vp_f['cmedians'].set_color(C_FAD); vp_f['cmedians'].set_lw(2)

    rng2 = np.random.default_rng(RANDOM_STATE)
    ax.scatter(rng2.normal(0, 0.07, len(wt_v)),  wt_v,  s=15,
               color=C_WT,  alpha=0.7, zorder=3)
    ax.scatter(rng2.normal(1, 0.07, len(fad_v)), fad_v, s=15,
               color=C_FAD, alpha=0.7, zorder=3)

    ax.set_xticks([0, 1]); ax.set_xticklabels(['WT', '5xFAD'], fontsize=9)
    ax.set_ylabel(FEAT_LABELS[feat], fontsize=8)
    ax.set_title(feat, fontweight='bold', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Effect size annotation (Cohen's d)
    d = (fad_v.mean() - wt_v.mean()) / (np.sqrt((wt_v.std()**2 + fad_v.std()**2) / 2) + 1e-8)
    ax.text(0.97, 0.97, f'd={d:+.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='#555',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

plt.suptitle('Phase Feature Distributions — WT vs 5xFAD\n'
             'Chirp Amplitude Segment  (d = Cohen\'s d effect size)',
             fontweight='bold', fontsize=11)
plt.savefig(os.path.join(FIG_DIR, '16_b_feature_distributions.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 16_b_feature_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C: Classification comparison + 2-D scatter of top-2 features
# ══════════════════════════════════════════════════════════════════════════════
print("Figure C: classification …")

n_clf_sets = 1 + (2 if hc_ok else 0)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(wspace=0.35)

# Left: AUC bar chart (horizontal, all conditions)
ax = axes[0]
bar_labels = ['Phase only', 'Envelope only', 'Phase + Env']
bar_aucs   = [auc_phase,    auc_env,          auc_both]
bar_cols   = ['#8e44ad',    '#e67e22',         '#c0392b']
if hc_ok:
    bar_labels += ['HC only', 'HC + Env', 'HC + All']
    bar_aucs   += [auc_hc,    auc_hc_env, auc_hc_all]
    bar_cols   += ['#27ae60', '#2980b9',   '#2c3e50']

y_pos = np.arange(len(bar_labels))
bars  = ax.barh(y_pos, bar_aucs, color=bar_cols, height=0.55,
                edgecolor='black', alpha=0.85)
ax.axvline(0.5, color='gray', ls=':', lw=1.2, label='Chance')
ax.set_xlim(0, 1.05)
ax.set_yticks(y_pos); ax.set_yticklabels(bar_labels, fontsize=9)
ax.set_xlabel('Pooled cross-val AUC (5-fold)', fontsize=10)
ax.set_title('A.  Classification: all feature sets\n'
             '(Logistic Regression)', fontweight='bold')
for bar, v in zip(bars, bar_aucs):
    ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
            f'{v:.3f}', va='center', fontweight='bold', fontsize=9)
ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.3)

# Right: 2-D scatter — xcorr_lag vs hilbert_slope
ax = axes[1]
wt_mask  = labels == 0
fad_mask = labels == 1
ax.scatter(xcorr_lag_arr[wt_mask],  hilbert_slope_arr[wt_mask],
           c=C_WT,  s=50, alpha=0.8, edgecolors='white', lw=0.5,
           label=f'WT (n={wt_mask.sum()})')
ax.scatter(xcorr_lag_arr[fad_mask], hilbert_slope_arr[fad_mask],
           c=C_FAD, s=50, alpha=0.8, edgecolors='white', lw=0.5,
           label=f'5xFAD (n={fad_mask.sum()})')
ax.set_xlabel('xcorr_lag  (samples)', fontsize=10)
ax.set_ylabel('hilbert_slope  (mean inst. freq.)', fontsize=10)
ax.set_title('B.  Feature space: xcorr lag vs Hilbert phase slope\n'
             '(most discriminating pair)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Mark group centroids
for mask, color, label in [(wt_mask, C_WT, 'WT'), (fad_mask, C_FAD, '5xFAD')]:
    cx = xcorr_lag_arr[mask].mean()
    cy = hilbert_slope_arr[mask].mean()
    ax.scatter([cx], [cy], marker='D', s=120, c=color, edgecolors='black',
               lw=1.2, zorder=5)
    ax.annotate(f'{label}\ncentroid', (cx, cy), textcoords='offset points',
                xytext=(8, 4), fontsize=7, color=color)

plt.suptitle('Phase Features — Classification Results\n'
             'Chirp Amplitude Segment  |  WT vs 5xFAD',
             fontweight='bold', fontsize=11)
plt.savefig(os.path.join(FIG_DIR, '16_c_classification.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 16_c_classification.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D: Envelope analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Figure D: envelope analysis …")

SIGMA_ENV = 30   # smooth envelope for visualisation (samples)
env_wt_sm  = gaussian_filter1d(mean_env_wt,  sigma=SIGMA_ENV)
env_fad_sm = gaussian_filter1d(mean_env_fad, sigma=SIGMA_ENV)
se_env_wt_sm  = gaussian_filter1d(se_env_wt,  sigma=SIGMA_ENV)
se_env_fad_sm = gaussian_filter1d(se_env_fad, sigma=SIGMA_ENV)

ENV_FEAT_LABELS = {
    'env_mean' : 'env_mean\n(overall following strength)',
    'env_ratio': 'env_ratio\n(late/early — adaptation)',
    'env_slope': 'env_slope\n(temporal trend)',
    'env_cv'   : 'env_cv\n(variability of following)',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.subplots_adjust(wspace=0.35)

# Panel 0: raw signals with envelope overlaid
ax = axes[0]
# Plot thin raw signals in background
for sig in wt_sigs:
    ax.plot(t, sig, color=C_WT, lw=0.3, alpha=0.15)
for sig in fad_sigs:
    ax.plot(t, sig, color=C_FAD, lw=0.3, alpha=0.15)
# Smoothed mean envelope — the key observation
ax.fill_between(t, env_wt_sm  - se_env_wt_sm,  env_wt_sm  + se_env_wt_sm,
                alpha=0.35, color=C_WT)
ax.fill_between(t, env_fad_sm - se_env_fad_sm, env_fad_sm + se_env_fad_sm,
                alpha=0.35, color=C_FAD)
ax.plot(t, env_wt_sm,  C_WT,  lw=2.5, label=f'WT envelope  (n={len(wt_sigs)}, ±SE)')
ax.plot(t, env_fad_sm, C_FAD, lw=2.5, label=f'5xFAD envelope  (n={len(fad_sigs)}, ±SE)')
ax.set_xlabel('Sample index', fontsize=9)
ax.set_ylabel('Hilbert envelope (raw amplitude)', fontsize=9)
ax.set_title('A.  Mean Hilbert envelope — WT vs 5xFAD\n'
             '(thin lines = individual subjects)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 1: env_mean violin (overall following strength)
ax = axes[1]
feat = 'env_mean'
wt_v  = df_phase.loc[df_phase.label == 0, feat].values
fad_v = df_phase.loc[df_phase.label == 1, feat].values
vp_w = ax.violinplot([wt_v],  positions=[0], widths=0.55, showmedians=True, showextrema=False)
vp_f = ax.violinplot([fad_v], positions=[1], widths=0.55, showmedians=True, showextrema=False)
for pc in vp_w['bodies']:  pc.set_facecolor(C_WT);  pc.set_alpha(0.55)
for pc in vp_f['bodies']:  pc.set_facecolor(C_FAD); pc.set_alpha(0.55)
vp_w['cmedians'].set_color(C_WT);  vp_w['cmedians'].set_lw(2)
vp_f['cmedians'].set_color(C_FAD); vp_f['cmedians'].set_lw(2)
rng3 = np.random.default_rng(RANDOM_STATE)
ax.scatter(rng3.normal(0, 0.07, len(wt_v)),  wt_v,  s=25, color=C_WT,  alpha=0.8, zorder=3)
ax.scatter(rng3.normal(1, 0.07, len(fad_v)), fad_v, s=25, color=C_FAD, alpha=0.8, zorder=3)
ax.set_xticks([0, 1]); ax.set_xticklabels(['WT', '5xFAD'], fontsize=10)
ax.set_ylabel('Mean Hilbert envelope\n(raw amplitude units)', fontsize=9)
d_env = (fad_v.mean()-wt_v.mean())/(np.sqrt((wt_v.std()**2+fad_v.std()**2)/2)+1e-8)
ax.set_title(f'B.  Overall following strength\n'
             f'(env_mean,  Cohen\'s d={d_env:+.2f})', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for pos, vals, color in [(0, wt_v, C_WT), (1, fad_v, C_FAD)]:
    ax.text(pos, vals.max()*1.04, f'μ={vals.mean():.3f}',
            ha='center', fontsize=9, fontweight='bold', color=color)

# Panel 2: env_ratio (adaptation: late/early)
ax = axes[2]
feat = 'env_ratio'
wt_v  = df_phase.loc[df_phase.label == 0, feat].values
fad_v = df_phase.loc[df_phase.label == 1, feat].values
vp_w = ax.violinplot([wt_v],  positions=[0], widths=0.55, showmedians=True, showextrema=False)
vp_f = ax.violinplot([fad_v], positions=[1], widths=0.55, showmedians=True, showextrema=False)
for pc in vp_w['bodies']:  pc.set_facecolor(C_WT);  pc.set_alpha(0.55)
for pc in vp_f['bodies']:  pc.set_facecolor(C_FAD); pc.set_alpha(0.55)
vp_w['cmedians'].set_color(C_WT);  vp_w['cmedians'].set_lw(2)
vp_f['cmedians'].set_color(C_FAD); vp_f['cmedians'].set_lw(2)
rng4 = np.random.default_rng(RANDOM_STATE)
ax.scatter(rng4.normal(0, 0.07, len(wt_v)),  wt_v,  s=25, color=C_WT,  alpha=0.8, zorder=3)
ax.scatter(rng4.normal(1, 0.07, len(fad_v)), fad_v, s=25, color=C_FAD, alpha=0.8, zorder=3)
ax.axhline(1.0, color='gray', ls='--', lw=1.2, alpha=0.7, label='No adaptation (ratio=1)')
ax.set_xticks([0, 1]); ax.set_xticklabels(['WT', '5xFAD'], fontsize=10)
ax.set_ylabel('Envelope ratio  (last 25% / first 25%)', fontsize=9)
d_ratio = (fad_v.mean()-wt_v.mean())/(np.sqrt((wt_v.std()**2+fad_v.std()**2)/2)+1e-8)
ax.set_title(f'C.  Temporal adaptation of following\n'
             f'(env_ratio,  Cohen\'s d={d_ratio:+.2f})', fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
for pos, vals, color in [(0, wt_v, C_WT), (1, fad_v, C_FAD)]:
    ax.text(pos, vals.max()*1.02, f'μ={vals.mean():.2f}',
            ha='center', fontsize=9, fontweight='bold', color=color)

plt.suptitle('Hilbert Envelope Analysis — Amplitude Sweep Following Response\n'
             'Does the retina follow the amplitude modulation equally in WT and 5xFAD?',
             fontweight='bold', fontsize=11)
plt.savefig(os.path.join(FIG_DIR, '16_d_envelope.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 16_d_envelope.png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE FEATURES — SUMMARY")
print(f"{'='*60}")
print(f"  Phase shift between group means: {peak_lag_means:+d} samples"
      f"  ({peak_lag_means/FS*1000:+.1f} ms)")
print(f"  Envelope: WT env_mean={env_mean_arr[labels==0].mean():.3f}  "
      f"5xFAD env_mean={env_mean_arr[labels==1].mean():.3f}")
print(f"\n  Classification AUCs (Logistic Regression, 5-fold pooled):")
print(f"    Phase only       : {auc_phase:.3f}")
print(f"    Envelope only    : {auc_env:.3f}")
print(f"    Phase + Envelope : {auc_both:.3f}")
if auc_hc is not None:
    print(f"    HC only          : {auc_hc:.3f}")
    print(f"    HC + Envelope    : {auc_hc_env:.3f}")
    print(f"    HC + All         : {auc_hc_all:.3f}")
    print(f"    Δ(HC+Env vs HC)  : {auc_hc_env - auc_hc:+.3f}")
print(f"\n  Figures → {FIG_DIR}")
print(f"  Tables  → {TAB_DIR}")
print(f"{'='*60}")
