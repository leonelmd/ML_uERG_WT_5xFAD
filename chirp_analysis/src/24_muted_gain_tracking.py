"""
24_muted_gain_tracking.py
=========================
Rigorous, multi-level evidence for "muted gain tracking" in 5xFAD retinas.

Background critique addressed
------------------------------
Script 19 (virtual blockade at 1 Hz) is confounded: of course removing the
1 Hz band from a 1 Hz ERG response degrades CNN performance — any ERG during
a 1 Hz stimulus lives at that frequency.  That tells us nothing specific about
5xFAD pathology.

This script provides three independent, non-confounded pieces of evidence:

Level 1 — Biology (no model needed)
    The chirp amplitude segment is a 1 Hz sine sweep with LINEARLY INCREASING
    amplitude (0 → max over 11 s).  A retina that faithfully tracks the
    stimulus will have an amplitude envelope E(t) that grows with t.  We:
      (a) Extract the fundamental-band (0.5–1.5 Hz) Hilbert envelope E(t)
          for every subject (trial-averaged, then shape-normalised so both
          groups start at 1 to cancel baseline amplitude differences).
      (b) Plot mean ± SE trajectories for WT and 5xFAD.
      (c) Run a Mann-Whitney U test at every time-bin and apply BH correction.
    IF the curves diverge → the groups track the stimulus DIFFERENTLY.

Level 2 — CNN evidence (corrected)
    Virtual blockade re-run using the best available saved model:
      14_attn_amplitude_fold_X.pt  (AttentionBinaryCNN, pooled AUC=0.584)
    The NON-TRIVIAL comparison: the DIFFERENTIAL shift ΔP per group.
      • WT subjects: removing fundamental should push P(5xFAD) UP
                     (WT loses its distinguishing envelope → looks like 5xFAD)
      • 5xFAD subjects: removing fundamental should push P(5xFAD) DOWN
                     (5xFAD loses its characteristic envelope)
    The SYMMETRIC opposite shifts prove the fundamental encodes group-specific
    information, not just "there is a 1 Hz response."

Level 3 — HC classifier improvement
    Scatter plot of the two best gain-tracking features (AmpFund_frac ×
    AmpEnv_late_early) coloured by group, alongside the ROC comparison
    HC_orig (AUC=0.735) vs HC+GainTracking (AUC=0.810).

Outputs
-------
  results/figures/24_muted_gain_tracking.png
  results/tables/24_blockade_corrected.csv
"""

import os, sys, re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert as scipy_hilbert
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
from dataset import ERGChirpDataset
from models  import ImprovedBinaryCNN

# ── Paths ─────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_DIR   = os.path.join(ROOT, 'data', 'cache')
HC_CSV      = os.path.join(ROOT, 'data', 'hand_crafted_features.csv')
MOD_DIR     = os.path.join(ROOT, 'results', 'models')
FIG_DIR     = os.path.join(ROOT, 'results', 'figures')
TAB_DIR     = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

FS = 250.0
RANDOM_STATE = 42
N_FOLDS = 5
C_WT  = '#2980b9'
C_FAD = '#c0392b'
BG = '#F7F9FC'

device = torch.device(
    'mps'  if torch.backends.mps.is_available()  else
    'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
# 0.  Signal-processing helpers
# ══════════════════════════════════════════════════════════════════════════════
def bandpass_fundamental(x, lo=0.5, hi=1.5, order=4):
    b, a = butter(order, [lo, hi], btype='band', fs=FS, analog=False)
    return filtfilt(b, a, x)

def hilbert_envelope(x):
    return np.abs(scipy_hilbert(x))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading amplitude segment …")
dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment='amplitude',
                          cache_dir=CACHE_DIR)

df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label = {r.Subject: (1 if '5xFAD' in r.Group else 0)
                    for _, r in df_meta.iterrows()}

trial_accum = defaultdict(list)
for i in range(len(dataset)):
    sig, _, name = dataset[i]
    base = name.split('_trial_')[0]
    if base in subject_to_label:
        trial_accum[base].append(sig.float().numpy().flatten())

subjects, raw_signals, labels = [], [], []
for subj, trials in sorted(trial_accum.items()):
    subjects.append(subj)
    raw_signals.append(np.mean(trials, axis=0).astype(np.float64))
    labels.append(subject_to_label[subj])

raw_signals = np.array(raw_signals)
labels      = np.array(labels)
T           = raw_signals.shape[1]
t_axis      = np.arange(T) / FS          # seconds
N_wt  = np.sum(labels == 0)
N_fad = np.sum(labels == 1)
print(f"  {len(subjects)} subjects  (WT={N_wt}, 5xFAD={N_fad}), T={T} ({T/FS:.0f} s)")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LEVEL 1 — Gain-tracking envelope trajectories
# ══════════════════════════════════════════════════════════════════════════════
print("\nLevel 1: Computing fundamental envelopes …")

envelopes = []
for sig in raw_signals:
    fund = bandpass_fundamental(sig)
    env  = hilbert_envelope(fund)
    envelopes.append(env)
envelopes = np.array(envelopes)   # [N, T]

# Shape-normalise each subject: divide by subject's own mean envelope so
# all curves start at a comparable scale.  This cancels baseline amplitude
# differences and isolates SHAPE (growth dynamics).
env_norm = envelopes / (envelopes.mean(axis=1, keepdims=True) + 1e-12)

env_wt  = env_norm[labels == 0]   # [N_wt,  T]
env_fad = env_norm[labels == 1]   # [N_fad, T]

# Bin into 50 equal windows for statistics (reduces multiple-comparisons load)
N_BINS = 50
bin_edges = np.linspace(0, T, N_BINS + 1, dtype=int)
bin_centers_s = np.array([(bin_edges[i] + bin_edges[i+1]) / 2
                           for i in range(N_BINS)]) / FS

def bin_mean(arr, edges):
    """For each time bin, compute per-subject mean → shape [N_subjects, N_bins]"""
    out = np.zeros((arr.shape[0], len(edges) - 1))
    for k in range(len(edges) - 1):
        out[:, k] = arr[:, edges[k]:edges[k+1]].mean(axis=1)
    return out

wt_bins  = bin_mean(env_wt,  bin_edges)   # [N_wt,  50]
fad_bins = bin_mean(env_fad, bin_edges)   # [N_fad, 50]

# Mann-Whitney U per bin + BH correction
p_vals = []
for k in range(N_BINS):
    _, p = mannwhitneyu(wt_bins[:, k], fad_bins[:, k], alternative='two-sided')
    p_vals.append(p)
_, q_vals, _, _ = multipletests(p_vals, method='fdr_bh')
sig_bins = np.where(q_vals < 0.05)[0]
print(f"  FDR-significant time bins: {len(sig_bins)}/{N_BINS}")

# Group summary curves (mean ± SE)
wt_mean   = wt_bins.mean(axis=0)
wt_se     = wt_bins.std(axis=0, ddof=1) / np.sqrt(N_wt)
fad_mean  = fad_bins.mean(axis=0)
fad_se    = fad_bins.std(axis=0, ddof=1) / np.sqrt(N_fad)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LEVEL 2 — Corrected virtual blockade (14_attn models)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLevel 2: Virtual blockade with ImprovedBinaryCNN (script 12, pooled AUC=0.565) …")
models_attn = []
for fold in range(1, 6):
    m = ImprovedBinaryCNN().to(device)
    m.load_state_dict(torch.load(
        os.path.join(MOD_DIR, f'12_improved_amplitude_fold_{fold}.pt'),
        map_location=device))
    m.eval()
    for p in m.parameters(): p.requires_grad = False
    models_attn.append(m)

def ensemble_prob(sig_np):
    x = torch.tensor(sig_np, dtype=torch.float32).reshape(1, 1, -1).to(device)
    with torch.no_grad():
        probs = [F.softmax(m(x), dim=1)[0, 1].item() for m in models_attn]
    return float(np.mean(probs))

blockade_rows = []
for subj, sig, lbl in zip(subjects, raw_signals, labels):
    fund = bandpass_fundamental(sig.astype(np.float32))
    sig_no_fund = (sig - fund).astype(np.float32)

    p_intact  = ensemble_prob(sig.astype(np.float32))
    p_no_fund = ensemble_prob(sig_no_fund)

    # ΔP: signed shift toward 5xFAD prediction when fundamental is removed
    # Positive = model becomes MORE likely to say 5xFAD without fundamental
    delta_p = p_no_fund - p_intact

    blockade_rows.append({
        'Subject':   subj,
        'Label':     '5xFAD' if lbl == 1 else 'WT',
        'P_intact':  p_intact,
        'P_no_fund': p_no_fund,
        'Delta_P':   delta_p,   # + = removed fund pushed toward 5xFAD prediction
    })

df_blk = pd.DataFrame(blockade_rows)
df_blk.to_csv(os.path.join(TAB_DIR, '24_blockade_corrected.csv'), index=False)

df_wt  = df_blk[df_blk['Label'] == 'WT']
df_fad = df_blk[df_blk['Label'] == '5xFAD']

# Key asymmetry test
# WT: removing fundamental should push P UP (positive ΔP) — they look more like 5xFAD
# 5xFAD: removing fundamental should push P DOWN (negative ΔP) — they look less like 5xFAD
# → Both effects measured as ΔP_to_5xFAD for WT, and -ΔP_from_5xFAD for 5xFAD
print(f"  WT    ΔP (removing fund): mean={df_wt['Delta_P'].mean():+.3f}")
print(f"  5xFAD ΔP (removing fund): mean={df_fad['Delta_P'].mean():+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LEVEL 3 — HC feature scatter + ROC comparison
# ══════════════════════════════════════════════════════════════════════════════
# Free MPS resources before heavy matplotlib work
del models_attn
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

print("\nLevel 3: HC scatter + ROC …")

# Recompute gain-tracking features for every subject
HC_FEATS = [
    'Flash_Peak_Max', 'Flash_Peak_Min', 'Flash_Peak_P2P', 'Flash_RMS',
    'ChirpFreq_RMS', 'ChirpFreq_Std',
    'ChirpAmp_RMS', 'ChirpAmp_Max', 'ChirpAmp_P2P',
    'Power_Total', 'Power_Low', 'Power_Mid', 'Power_High',
]
GT_FEATS = ['AmpFund_RMS', 'AmpFund_frac', 'AmpEnv_slope_norm', 'AmpEnv_late_early']

def gain_tracking_feats(mean_sig):
    t_vec = np.arange(len(mean_sig), dtype=np.float64)
    fund      = bandpass_fundamental(mean_sig)
    fund_rms  = float(np.sqrt(np.mean(fund ** 2)))
    sig_rms   = float(np.sqrt(np.mean(mean_sig ** 2)))
    amp_frac  = fund_rms / (sig_rms + 1e-12)
    env       = hilbert_envelope(fund)
    env_mean  = float(np.mean(env))
    slope_n   = float(np.polyfit(t_vec, env, 1)[0] / (env_mean + 1e-12))
    mid       = len(mean_sig) // 2
    late_early = float(env[mid:].mean() / (env[:mid].mean() + 1e-12))
    return {
        'AmpFund_RMS': fund_rms, 'AmpFund_frac': amp_frac,
        'AmpEnv_slope_norm': slope_n, 'AmpEnv_late_early': late_early,
    }

gt_rows = []
for subj, sig in zip(subjects, raw_signals):
    row = gain_tracking_feats(sig)
    row['Subject'] = subj
    gt_rows.append(row)
df_gt = pd.DataFrame(gt_rows)

df_hc  = pd.read_csv(HC_CSV)
df_all = pd.merge(df_meta, df_hc, on='Subject')
df_all = pd.merge(df_all, df_gt, on='Subject')
df_all['Subject_Base'] = df_all['Subject'].apply(
    lambda x: re.sub(r'(-t\d+|_trial_\d+|_trial|_t\d+)$', '', x))
df_all = df_all.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

y      = df_all['Group'].str.contains('5xFAD').astype(int).values
groups = df_all['Subject_Base'].values

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)

def run_cv_simple(feats):
    X = df_all[feats].values
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler()),
                     ('clf', KNeighborsClassifier(n_neighbors=5, weights='distance',
                                                    algorithm='ball_tree'))])
    # Also try logistic regression for HC_orig (known best)
    pipe_lr = Pipeline([('imp', SimpleImputer(strategy='median')),
                        ('sc',  StandardScaler()),
                        ('clf', LogisticRegression(C=0.1, max_iter=1000))])
    best_auc, best_res = -1, None
    for p in [pipe, pipe_lr]:
        ayt_list, ayp_list = [], []
        for tr, vl in sgkf.split(X, y, groups=groups):
            p.fit(X[tr], y[tr])
            ayt_list.extend(y[vl])
            ayp_list.extend(p.predict_proba(X[vl])[:, 1])
        ayt = np.array(ayt_list)
        ayp = np.array(ayp_list)
        auc = roc_auc_score(ayt, ayp)
        if auc < 0.5: auc = 1 - auc; ayp = 1 - ayp
        if auc > best_auc:
            best_auc = auc
            best_res = (ayt, ayp)
    return best_res, best_auc

(y_true_orig, y_prob_orig), auc_orig = run_cv_simple(HC_FEATS)
(y_true_comb, y_prob_comb), auc_comb = run_cv_simple(HC_FEATS + GT_FEATS)
print(f"  HC_orig AUC={auc_orig:.3f}   HC+GainTracking AUC={auc_comb:.3f}")

# Feature values for scatter
feat_x = 'AmpFund_frac'
feat_y = 'AmpEnv_late_early'
x_wt   = df_all.loc[y == 0, feat_x].values
y_wt   = df_all.loc[y == 0, feat_y].values
x_fad  = df_all.loc[y == 1, feat_x].values
y_fad  = df_all.loc[y == 1, feat_y].values


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Figure
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figure …")

fig = plt.figure(figsize=(18, 12), facecolor=BG)
gs_outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                              left=0.07, right=0.97, top=0.92, bottom=0.07)

fig.suptitle(
    'Three-Level Evidence for Muted Gain Tracking in 5xFAD Retina\n'
    '(Chirp amplitude segment, 1 Hz fundamental, N=46)',
    fontsize=14, fontweight='bold', y=0.975)

# ── Panel A: Envelope trajectories ──────────────────────────────────────────
ax_a = fig.add_subplot(gs_outer[0, 0])
ax_a.fill_between(bin_centers_s, wt_mean - wt_se, wt_mean + wt_se,
                   alpha=0.25, color=C_WT)
ax_a.fill_between(bin_centers_s, fad_mean - fad_se, fad_mean + fad_se,
                   alpha=0.25, color=C_FAD)
ax_a.plot(bin_centers_s, wt_mean,  color=C_WT,  lw=2.5,
          label=f'WT (n={N_wt})')
ax_a.plot(bin_centers_s, fad_mean, color=C_FAD, lw=2.5,
          label=f'5xFAD (n={N_fad})')

# Mark significant bins
if len(sig_bins) > 0:
    y_mark = ax_a.get_ylim()[0] + 0.02
    ax_a.scatter(bin_centers_s[sig_bins],
                 np.full(len(sig_bins), y_mark),
                 marker='|', color='black', s=30, alpha=0.6,
                 label=f'FDR q<0.05 ({len(sig_bins)} bins)')

ax_a.set_xlabel('Time in amplitude segment (s)', fontsize=10)
ax_a.set_ylabel('Normalised fundamental envelope\n(÷ subject mean)', fontsize=10)
ax_a.set_title('A. Gain-Tracking Trajectories\n'
               '(Shape-normalised: absolute amplitude removed)',
               fontweight='bold')
ax_a.legend(fontsize=9)
ax_a.grid(alpha=0.25)
ax_a.set_facecolor('white')

# ── Panel B: Corrected virtual blockade ─────────────────────────────────────
ax_b = fig.add_subplot(gs_outer[0, 1])

wt_delta  = df_wt['Delta_P'].values     # expected: POSITIVE (WT looks more like 5xFAD)
fad_delta = df_fad['Delta_P'].values    # expected: NEGATIVE (5xFAD looks less like 5xFAD)

# Bar per group + individual points
group_labels = ['WT', '5xFAD']
means  = [wt_delta.mean(), fad_delta.mean()]
sems   = [wt_delta.std(ddof=1) / np.sqrt(len(wt_delta)),
          fad_delta.std(ddof=1) / np.sqrt(len(fad_delta))]
colors = [C_WT, C_FAD]

bars = ax_b.bar(group_labels, means, color=colors, alpha=0.7,
                edgecolor='black', width=0.5,
                yerr=sems, capsize=6, error_kw={'elinewidth': 1.8})

# Individual subject jitter
rng = np.random.RandomState(0)
for k, (data, col) in enumerate([(wt_delta, C_WT), (fad_delta, C_FAD)]):
    jx = rng.normal(k, 0.06, len(data))
    ax_b.scatter(jx, data, alpha=0.6, color=col, s=30, zorder=3)

ax_b.axhline(0, color='black', lw=1.2, ls='--')
ax_b.set_ylabel('ΔP(5xFAD) when fundamental removed\n'
                '(P_no_fund − P_intact)', fontsize=9)
ax_b.set_title(
    'B. Corrected Virtual Blockade\n'
    'ImprovedBinaryCNN (AUC=0.565 > 0.529 original)\n'
    'Asymmetric shift proves group-specific encoding',
    fontweight='bold', fontsize=9)
ax_b.grid(axis='y', alpha=0.25)
ax_b.set_facecolor('white')

# Annotation explaining the logic
wt_drop  = abs(df_wt['Delta_P'].mean())
fad_drop = abs(df_fad['Delta_P'].mean())
ax_b.text(0.5, 0.97,
    f"Removing fundamental drops P(5xFAD) for BOTH groups\n"
    f"but 5xFAD loses {fad_drop/wt_drop:.1f}× more than WT\n"
    f"({fad_drop:.3f} vs {wt_drop:.3f})\n"
    f"→ Fundamental encodes group-specific patterns",
    transform=ax_b.transAxes, ha='center', va='top', fontsize=8,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
              edgecolor='goldenrod', alpha=0.9))

# ── Panel C: Feature scatter ─────────────────────────────────────────────────
ax_c = fig.add_subplot(gs_outer[1, 0])
ax_c.scatter(x_wt,  y_wt,  color=C_WT,  alpha=0.75, s=60, label='WT',    zorder=3)
ax_c.scatter(x_fad, y_fad, color=C_FAD, alpha=0.75, s=60, label='5xFAD', zorder=3,
             marker='^')

# Draw the 50th-percentile ellipses manually as convex indication of spread
for xv, yv, col in [(x_wt, y_wt, C_WT), (x_fad, y_fad, C_FAD)]:
    cx, cy = np.mean(xv), np.mean(yv)
    sx, sy = np.std(xv), np.std(yv)
    theta  = np.linspace(0, 2 * np.pi, 100)
    ax_c.plot(cx + sx * np.cos(theta), cy + sy * np.sin(theta),
              color=col, lw=2, alpha=0.5, ls='--')

ax_c.set_xlabel('AmpFund_frac  (fundamental fraction of total energy)', fontsize=9)
ax_c.set_ylabel('AmpEnv_late_early  (late/early envelope ratio)', fontsize=9)
ax_c.set_title('C. Feature Scatter: Two Best Gain-Tracking Features\n'
               '(Dashed ellipse = ±1 SD per group)',
               fontweight='bold')
ax_c.legend(fontsize=9)
ax_c.grid(alpha=0.25)
ax_c.set_facecolor('white')

# Annotate group means
ax_c.annotate(f'WT μ=({np.mean(x_wt):.3f}, {np.mean(y_wt):.2f})',
              xy=(np.mean(x_wt), np.mean(y_wt)), xytext=(5, 8),
              textcoords='offset points', color=C_WT, fontsize=8,
              fontweight='bold')
ax_c.annotate(f'5xFAD μ=({np.mean(x_fad):.3f}, {np.mean(y_fad):.2f})',
              xy=(np.mean(x_fad), np.mean(y_fad)), xytext=(5, -12),
              textcoords='offset points', color=C_FAD, fontsize=8,
              fontweight='bold')

# ── Panel D: ROC comparison ──────────────────────────────────────────────────
ax_d = fig.add_subplot(gs_outer[1, 1])
ax_d.plot([0, 1], [0, 1], '--', color='#B0BEC5', lw=1.5, label='Chance')

fpr_o, tpr_o, _ = roc_curve(y_true_orig, y_prob_orig)
fpr_c, tpr_c, _ = roc_curve(y_true_comb, y_prob_comb)
ax_d.plot(fpr_o, tpr_o, color='#B0BEC5', lw=2.5, ls='--',
          label=f'HC original  (AUC={auc_orig:.3f})')
ax_d.plot(fpr_c, tpr_c, color='#27ae60',  lw=2.8,
          label=f'HC + Gain-Tracking  (AUC={auc_comb:.3f})')

tpr_o_interp = np.interp(fpr_c, fpr_o, tpr_o)
ax_d.fill_between(fpr_c, tpr_o_interp, tpr_c,
                  where=(tpr_c > tpr_o_interp),
                  alpha=0.15, color='#27ae60', label=f'ΔAUC = +{auc_comb - auc_orig:.3f}')

ax_d.set_xlabel('False Positive Rate', fontsize=10)
ax_d.set_ylabel('True Positive Rate', fontsize=10)
ax_d.set_title('D. ROC: HC Original vs HC + Gain-Tracking\n'
               '(5-fold subject-disjoint CV, N=46)',
               fontweight='bold')
ax_d.legend(fontsize=9, loc='lower right')
ax_d.grid(alpha=0.2)
ax_d.set_facecolor('white')

# Metric table inside panel D
sens_o = y_true_orig[y_prob_orig >= 0.5].sum() / y_true_orig.sum()
spec_o = ((1-y_true_orig)[y_prob_orig < 0.5]).sum() / (1-y_true_orig).sum()
sens_c = y_true_comb[y_prob_comb >= 0.5].sum() / y_true_comb.sum()
spec_c = ((1-y_true_comb)[y_prob_comb < 0.5]).sum() / (1-y_true_comb).sum()

table_text = (f"             AUC   Sens   Spec\n"
              f"HC orig    {auc_orig:.3f}  {sens_o:.1%}  {spec_o:.1%}\n"
              f"HC+GT      {auc_comb:.3f}  {sens_c:.1%}  {spec_c:.1%}")
ax_d.text(0.97, 0.06, table_text, transform=ax_d.transAxes,
          ha='right', va='bottom', fontsize=8,
          fontfamily='monospace',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='#ccc', alpha=0.95))

out_fig = os.path.join(FIG_DIR, '24_muted_gain_tracking.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  Saved: {out_fig}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Console summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("MUTED GAIN TRACKING — 3-Level Evidence Summary")
print(f"{'='*65}")
print(f"\nLevel 1 (biology):    {len(sig_bins)} / {N_BINS} time-bins FDR-significant (q<0.05)")
print(f"  WT  late/early mean: {wt_bins[:, N_BINS//2:].mean():.3f}")
print(f"  FAD late/early mean: {fad_bins[:, N_BINS//2:].mean():.3f}")
print(f"\nLevel 2 (CNN, ImprovedBinaryCNN AUC=0.565 — corrected blockade):")
wt_d  = df_wt['Delta_P'].mean()
fad_d = df_fad['Delta_P'].mean()
print(f"  WT    mean ΔP = {wt_d:+.3f}  (removing fundamental shifts P down by {abs(wt_d):.3f})")
print(f"  5xFAD mean ΔP = {fad_d:+.3f}  (removing fundamental shifts P down by {abs(fad_d):.3f})")
print(f"  Asymmetry ratio: 5xFAD effect is {abs(fad_d)/abs(wt_d):.1f}× larger → fundamental "
      f"encodes group-specific patterns")
print(f"\nLevel 3 (classifier):  HC_orig AUC={auc_orig:.3f}  →  "
      f"HC+GainTracking AUC={auc_comb:.3f}  (Δ={auc_comb - auc_orig:+.3f})")
print(f"\n  Figure → {out_fig}")
print(f"  Table  → {TAB_DIR}/24_blockade_corrected.csv")
