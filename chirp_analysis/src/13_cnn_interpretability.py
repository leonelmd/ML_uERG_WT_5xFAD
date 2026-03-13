"""
13_cnn_interpretability.py
==========================
Interpretability analysis for the Improved Binary CNN trained on the chirp
amplitude segment (WT vs 5xFAD).

Methods
-------
1. Layer-1 Kernel Visualisation
   Filter waveforms (mean ± SD across 5 folds) + FFT frequency response.
   Reveals what temporal patterns the first convolutional layer learned to detect.

2. Grad-CAM  (Selvaraju et al., 2017)
   Gradients of the 5xFAD logit w.r.t. activations at the last conv layer
   [B, 32, 22].  Upsampled to [2750] via linear interpolation.
   NOTE: Native resolution is 22 points (~125 samples/point).  The upsampled
   map is provided for overlay only; see IG for full-resolution attribution.

3. Integrated Gradients  (Sundararajan et al., 2017)
   Primary attribution method.  Full 2750-point input resolution.
   Satisfies COMPLETENESS: sum(IG) = F(x) - F(baseline).
   Baseline: mean WT signal (not zero — InstanceNorm1d maps any scaled signal
   to the same normalised form, so a zero baseline yields degenerate zero
   gradients for all alpha > 0).
   Attribution meaning: "how much does this subject's deviation from a typical
   WT response drive the model toward predicting 5xFAD?"

4. Statistical Comparison
   Per-sample-point Mann-Whitney U test on IG maps (WT vs 5xFAD subjects).
   Benjamini-Hochberg FDR correction (q < 0.05).
   Identifies which temporal windows significantly differ between genotypes.

All attributions are computed ONLY on fold-held-out subjects (no train
contamination), by replaying the exact StratifiedKFold used in script 05.

Outputs
-------
  results/figures/13_a_kernels.png
  results/figures/13_b_gradcam.png
  results/figures/13_c_integrated_grads.png
  results/tables/13_significant_windows.csv

Usage (from chirp_analysis/ folder)
-------------------------------------
    python src/13_cnn_interpretability.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold

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

FIG_DIR = os.path.join(ROOT, 'results', 'figures')
TAB_DIR = os.path.join(ROOT, 'results', 'tables')
MOD_DIR = os.path.join(ROOT, 'results', 'models')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ── Params (must match 05_train_binary_cnn.py) ────────────────────────────────
N_FOLDS      = 5
RANDOM_STATE = 42
IG_STEPS     = 50          # Riemann steps for IG integration
SMOOTH_SIGMA = 20          # Gaussian smoothing kernel (samples) for visualisation

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# ── Load metadata ─────────────────────────────────────────────────────────────
df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label = {r.Subject: (1 if '5xFAD' in r.Group else 0)
                    for _, r in df_meta.iterrows()}

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset ...")
dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment='amplitude',
                          cache_dir=CACHE_DIR)

all_trial_subjs = [s.split('_trial_')[0] for s in dataset.subjects]
valid_idx       = [i for i, s in enumerate(all_trial_subjs) if s in subject_to_label]
unique_subjects = sorted(set(all_trial_subjs[i] for i in valid_idx))
subj_labels     = [subject_to_label[s] for s in unique_subjects]

sample_sig, _, _ = dataset[valid_idx[0]]
T = int(sample_sig.shape[-1])
t = np.arange(T)
print(f"Signal length: {T} samples  |  Subjects: {len(unique_subjects)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Attribution helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _to_device(sig):
    """Convert signal (numpy or tensor) → [1, 1, T] float32 on device."""
    if isinstance(sig, np.ndarray):
        sig = torch.tensor(sig, dtype=torch.float32)
    return sig.float().reshape(1, 1, -1).to(device)


def grad_cam(model, x, target_class=1):
    """
    1D Grad-CAM from the final ReLU of model.conv (last convolutional layer).
    Native resolution: 22 points.  Returns upsampled [T] float32 array in [0,1].

    x : [1, 1, T] tensor on device.
    """
    model.eval()
    acts, grads = {}, {}

    def fwd_hook(m, inp, out):
        acts['v'] = out                  # [1, 32, 22]

    def bwd_hook(m, g_in, g_out):
        grads['v'] = g_out[0]            # [1, 32, 22]

    # Hook on the final ReLU (index -1 == model.conv[10])
    hf = model.conv[-1].register_forward_hook(fwd_hook)
    hb = model.conv[-1].register_full_backward_hook(bwd_hook)

    x_in = x.clone().detach().requires_grad_(True)
    out  = model(x_in)
    model.zero_grad()
    out[0, target_class].backward()
    hf.remove(); hb.remove()

    A = acts['v'][0]                                     # [32, 22]
    G = grads['v'][0]                                    # [32, 22]
    alpha = G.mean(dim=-1, keepdim=True)                 # [32, 1]
    cam   = F.relu((alpha * A).sum(dim=0))               # [22]
    cam   = cam / (cam.max() + 1e-8)
    cam_np = cam.detach().cpu().float().numpy()

    # Upsample to input resolution
    cam_up = np.interp(np.linspace(0, 21, T), np.arange(22), cam_np)
    return cam_up                                        # [T], in [0,1]


def integrated_gradients(model, x, baseline, target_class=1, n_steps=IG_STEPS):
    """
    Integrated Gradients (Sundararajan 2017).
    Baseline: mean WT signal (passed in as [1,1,T] tensor on device).
    Returns signed attribution array [T] and completeness error.

    Positive values → increasing the signal here increases P(5xFAD).
    Negative values → increasing the signal here increases P(WT).
    """
    model.eval()
    alphas = torch.linspace(0, 1, n_steps + 1, device=device)  # [n+1]

    # Interpolated inputs: [n+1, 1, T]
    interp = (baseline + alphas.view(-1, 1, 1) * (x - baseline)).detach()
    interp.requires_grad_(True)

    out   = model(interp)                            # [n+1, 2]
    score = out[:, target_class].sum()
    score.backward()

    grads = interp.grad                              # [n+1, 1, T]
    # Trapezoidal rule (correct formula)
    # avg ≈ (1/m) * [g0/2 + g1 + ... + g_{m-1} + gm/2]
    avg_grads = (grads[1:-1].sum(dim=0) +
                 (grads[0] + grads[-1]) / 2) / n_steps   # [1, T]
    ig = ((x - baseline) * avg_grads).squeeze()           # [T]
    ig_np = ig.detach().cpu().float().numpy()

    # Completeness check
    with torch.no_grad():
        F_x    = model(x)[0, target_class].item()
        F_base = model(baseline)[0, target_class].item()
    delta = abs(ig_np.sum() - (F_x - F_base))

    return ig_np, delta


# ══════════════════════════════════════════════════════════════════════════════
# Compute mean-WT baseline from ALL WT subjects (used by IG)
# ══════════════════════════════════════════════════════════════════════════════
print("Computing mean-WT baseline for IG ...")
wt_all_signals = []
for idx in valid_idx:
    sig, _, name = dataset[idx]
    subj = name.split('_trial_')[0]
    if subject_to_label[subj] == 0:
        s = sig.float().numpy() if hasattr(sig, 'numpy') else np.array(sig, dtype=np.float32)
        wt_all_signals.append(s.flatten())

mean_wt_signal  = np.mean(wt_all_signals, axis=0)          # [T]
baseline_tensor = _to_device(mean_wt_signal)                # [1, 1, T]


# ══════════════════════════════════════════════════════════════════════════════
# Main loop: collect attributions on held-out subjects (fold-disjoint)
# ══════════════════════════════════════════════════════════════════════════════
print("Computing fold-disjoint attributions ...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
records = []   # one dict per held-out subject

for fold, (tr_idx, vl_idx) in enumerate(
        skf.split(unique_subjects, subj_labels), 1):

    val_subs  = {unique_subjects[i] for i in vl_idx}
    vl_trials = [i for i in valid_idx if all_trial_subjs[i] in val_subs]

    model = ImprovedBinaryCNN().to(device)
    model.load_state_dict(
        torch.load(os.path.join(MOD_DIR, f'12_improved_amplitude_fold_{fold}.pt'),
                   map_location=device, weights_only=True))
    model.eval()

    # Aggregate trials per subject → one mean signal
    subj_data = {}
    for idx in vl_trials:
        sig, _, name = dataset[idx]
        base = name.split('_trial_')[0]
        arr  = sig.float().numpy() if hasattr(sig, 'numpy') else np.array(sig, dtype=np.float32)
        subj_data.setdefault(base, []).append(arr.flatten())

    for subj, sigs in subj_data.items():
        mean_sig = np.mean(sigs, axis=0)                    # [T]
        lbl      = subject_to_label[subj]
        x        = _to_device(mean_sig)                     # [1, 1, T]

        with torch.no_grad():
            logits   = model(x)
            pred     = logits.argmax(1).item()
            prob_fad = F.softmax(logits, dim=1)[0, 1].item()

        cam = grad_cam(model, x)
        ig, delta = integrated_gradients(model, x, baseline_tensor)

        # TemporalStatPool features: the 96 values the FC classifier sees
        # Shape of model.conv output: [1, 32, 22] → pool → [96]
        with torch.no_grad():
            _store = {}
            def _feat_hook(m, inp, out, _s=_store):
                _s['feat'] = out.detach()        # [1, 32, 22]
            h = model.conv[-1].register_forward_hook(_feat_hook)
            model(x)
            h.remove()
            A   = _store['feat'][0]             # [32, 22]
            tsp = torch.cat([
                A.mean(dim=-1),
                A.amax(dim=-1),
                A.std(dim=-1, unbiased=False),
            ], dim=0).cpu().float().numpy()     # [96]

        records.append({
            'subject' : subj,
            'label'   : lbl,
            'pred'    : pred,
            'prob_fad': prob_fad,
            'signal'  : mean_sig,
            'cam'     : cam,
            'ig'      : ig,
            'ig_delta': delta,
            'tsp'     : tsp,                    # TemporalStatPool features [96]
        })

    fold_deltas = [r['ig_delta'] for r in records[-len(subj_data):]]
    print(f"  Fold {fold}: {len(subj_data)} subjects  "
          f"(mean IG completeness error: {np.mean(fold_deltas):.5f})")

print(f"Total subjects: {len(records)}  "
      f"(WT={sum(r['label']==0 for r in records)}, "
      f"5xFAD={sum(r['label']==1 for r in records)})\n")

# Convenience groupings
wt_rec  = [r for r in records if r['label'] == 0]
fad_rec = [r for r in records if r['label'] == 1]

wt_igs  = np.array([r['ig']  for r in wt_rec])   # [N_wt,  T]
fad_igs = np.array([r['ig']  for r in fad_rec])  # [N_fad, T]
wt_sigs = np.array([r['signal'] for r in wt_rec])
fad_sigs= np.array([r['signal'] for r in fad_rec])


# ── Smoothed mean / SE helpers ────────────────────────────────────────────────
def smean(arr, sigma=SMOOTH_SIGMA):
    return gaussian_filter1d(arr.mean(axis=0), sigma=sigma)

def sse(arr, sigma=SMOOTH_SIGMA):
    se = arr.std(axis=0) / np.sqrt(arr.shape[0])
    return gaussian_filter1d(se, sigma=sigma)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A: Layer-1 Kernel Visualisation
# ══════════════════════════════════════════════════════════════════════════════
print("Figure A: Layer-1 kernels ...")

all_kernels = []
for fold in range(1, N_FOLDS + 1):
    m = ImprovedBinaryCNN()
    m.load_state_dict(
        torch.load(os.path.join(MOD_DIR, f'12_improved_amplitude_fold_{fold}.pt'),
                   map_location='cpu', weights_only=True))
    # conv[0] = Conv1d(1, 8, kernel_size=15)
    all_kernels.append(m.conv[0].weight.detach().cpu().numpy()[:, 0, :])  # [8, 15]
all_kernels = np.array(all_kernels)   # [5, 8, 15]

k_mean = all_kernels.mean(axis=0)    # [8, 15]
n_filt = all_kernels.shape[1]        # 8
n_taps = all_kernels.shape[2]        # 15
taps   = np.arange(n_taps)
freqs  = np.fft.rfftfreq(256)        # normalised frequency [0, 0.5]

FOLD_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # one per fold

fig, axes = plt.subplots(2, n_filt, figsize=(18, 6))

for col in range(n_filt):
    ax_filt = axes[0, col]
    ax_fft  = axes[1, col]
    color   = plt.cm.tab10(col)

    # ── Individual fold traces (waveform) ─────────────────────────────────
    for f in range(all_kernels.shape[0]):
        ax_filt.plot(taps, all_kernels[f, col], color=FOLD_COLORS[f],
                     lw=1.2, alpha=0.8, label=f'Fold {f+1}')
    # Mean across folds as thicker black line
    ax_filt.plot(taps, k_mean[col], color='black', lw=2.2,
                 ls='--', alpha=0.85, label='Mean')
    ax_filt.axhline(0, color='#aaaaaa', lw=0.7, ls=':')
    ax_filt.set_title(f'Filter {col+1}', fontsize=9, fontweight='bold')
    ax_filt.set_xlabel('Tap (×4 ms)', fontsize=7)
    if col == 0:
        ax_filt.set_ylabel('Weight', fontsize=8)
        ax_filt.legend(fontsize=5.5, loc='upper right',
                       handlelength=1.2, labelspacing=0.3)
    ax_filt.tick_params(labelsize=6)
    ax_filt.grid(alpha=0.25)

    # ── Individual fold FFT responses ─────────────────────────────────────
    for f in range(all_kernels.shape[0]):
        fft_f = np.abs(np.fft.rfft(all_kernels[f, col], n=256))
        ax_fft.plot(freqs, fft_f, color=FOLD_COLORS[f], lw=1.0, alpha=0.7)
    # Mean FFT
    fft_mean = np.abs(np.fft.rfft(k_mean[col], n=256))
    ax_fft.plot(freqs, fft_mean, color='black', lw=2.0, ls='--', alpha=0.85)
    peak_f = freqs[fft_mean.argmax()]
    ax_fft.axvline(peak_f, color=color, ls=':', lw=1.2, alpha=0.8)
    ax_fft.text(peak_f + 0.01, fft_mean.max() * 0.88, f'{peak_f:.2f}',
                fontsize=6, ha='left', color=color)
    ax_fft.set_xlabel('Norm. freq.', fontsize=7)
    if col == 0:
        ax_fft.set_ylabel('|FFT|', fontsize=8)
    ax_fft.tick_params(labelsize=6)
    ax_fft.grid(alpha=0.25)

plt.suptitle('Layer-1 Convolutional Filters — ImprovedBinaryCNN (Chirp Amplitude)\n'
             'Top: filter weights per fold (coloured) + mean (black dashed)  |  '
             'Bottom: frequency response (FFT magnitude) per fold + mean',
             fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '13_a_kernels.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 13_a_kernels.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B: Grad-CAM
# ══════════════════════════════════════════════════════════════════════════════
print("Figure B: Grad-CAM ...")

wt_cams  = np.array([r['cam'] for r in wt_rec])   # [N_wt,  T]
fad_cams = np.array([r['cam'] for r in fad_rec])  # [N_fad, T]

# Most confident correct predictions for representative traces
correct_wt  = [r for r in wt_rec  if r['pred'] == 0]
correct_fad = [r for r in fad_rec if r['pred'] == 1]
if not correct_wt:  correct_wt  = wt_rec   # fallback
if not correct_fad: correct_fad = fad_rec

best_wt  = min(correct_wt,  key=lambda r: r['prob_fad'])
best_fad = max(correct_fad, key=lambda r: r['prob_fad'])

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.45, wspace=0.35)

# ── Row 0: Group-mean ERG signals ─────────────────────────────────────────────
for ax, sigs, color, lbl, n in [
        (axes[0, 0], wt_sigs,  '#2980b9', 'WT',    len(wt_rec)),
        (axes[0, 1], fad_sigs, '#c0392b', '5xFAD', len(fad_rec))]:
    m_, s_ = smean(sigs), sse(sigs)
    ax.fill_between(t, m_-s_, m_+s_, alpha=0.2, color=color)
    ax.plot(t, m_, color=color, lw=1.5, label=f'{lbl} (n={n})')
    ax.set_ylabel('Amplitude (norm.)', fontsize=9)
    ax.set_title(f'Mean ERG — {lbl}', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── Row 1: Mean Grad-CAM per group ────────────────────────────────────────────
for ax, cams, color, lbl in [
        (axes[1, 0], wt_cams,  '#2980b9', 'WT'),
        (axes[1, 1], fad_cams, '#c0392b', '5xFAD')]:
    m_cam = smean(cams)
    s_cam = sse(cams)
    ax.fill_between(t, m_cam-s_cam, m_cam+s_cam, alpha=0.25, color='darkorange')
    ax.plot(t, m_cam, color='darkorange', lw=2,
            label=f'Grad-CAM ({lbl})')
    ax.set_ylim(0, None)
    ax.set_ylabel('Grad-CAM (norm., toward 5xFAD)', fontsize=9)
    ax.set_title(f'Mean Grad-CAM — {lbl} subjects', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── Row 2: Overlay on representative traces ───────────────────────────────────
for ax, rec, lbl in [
        (axes[2, 0], best_wt,
         f"Best-classified WT  (p_5xFAD={best_wt['prob_fad']:.2f})"),
        (axes[2, 1], best_fad,
         f"Best-classified 5xFAD  (p_5xFAD={best_fad['prob_fad']:.2f})")]:

    sig = rec['signal']
    cam = gaussian_filter1d(rec['cam'], sigma=SMOOTH_SIGMA)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    y_lo = sig.min() - 0.15 * (sig.max() - sig.min())
    y_hi = sig.max() + 0.15 * (sig.max() - sig.min())

    # Heatmap background via imshow
    cam_2d = np.tile(cam[np.newaxis, :], (30, 1))
    ax.imshow(cam_2d, aspect='auto',
              extent=[t[0], t[-1], y_lo, y_hi],
              cmap='YlOrRd', alpha=0.55, vmin=0, vmax=1,
              origin='lower', interpolation='bilinear')

    c = '#2980b9' if rec['label'] == 0 else '#c0392b'
    ax.plot(t, sig, color=c, lw=1.5, zorder=5, label=lbl)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel('Sample index', fontsize=9)
    ax.set_ylabel('Amplitude (norm.)', fontsize=9)
    ax.set_title(f'Grad-CAM Overlay — {lbl}', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle('Grad-CAM Analysis — Chirp Amplitude Sweep  (toward 5xFAD prediction)\n'
             '[Note: native Grad-CAM resolution = 22 points, upsampled for overlay]',
             fontweight='bold', fontsize=11)
plt.savefig(os.path.join(FIG_DIR, '13_b_gradcam.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 13_b_gradcam.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C: Integrated Gradients + Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Figure C: Integrated Gradients + statistics ...")

# ── Pointwise Mann-Whitney U, BH FDR correction ──────────────────────────────
print("  Running pointwise Mann-Whitney tests (N=2750) ...")
pvals = np.array([
    mannwhitneyu(wt_igs[:, ti], fad_igs[:, ti], alternative='two-sided').pvalue
    for ti in range(T)
])
_, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')
sig_mask = pvals_fdr < 0.05

# Identify contiguous significant windows
sig_windows = []
in_win = False
for ti, is_sig in enumerate(sig_mask):
    if is_sig and not in_win:
        w_start = ti; in_win = True
    elif not is_sig and in_win:
        sig_windows.append((w_start, ti - 1)); in_win = False
if in_win:
    sig_windows.append((w_start, T - 1))

print(f"  Significant windows (FDR q<0.05): {len(sig_windows)}")
for s_, e_ in sig_windows:
    diff_mean = fad_igs[:, s_:e_+1].mean() - wt_igs[:, s_:e_+1].mean()
    print(f"    Samples [{s_}:{e_}]  len={e_-s_+1}  "
          f"mean ΔIG(5xFAD-WT)={diff_mean:.4f}")

# ── Save table ────────────────────────────────────────────────────────────────
rows = []
for s_, e_ in sig_windows:
    diff_mean = fad_igs[:, s_:e_+1].mean() - wt_igs[:, s_:e_+1].mean()
    rows.append({'start_sample': s_, 'end_sample': e_,
                 'length_samples': e_-s_+1,
                 'mean_delta_ig_5xFAD_minus_WT': round(diff_mean, 6),
                 'direction': '5xFAD>WT' if diff_mean > 0 else 'WT>5xFAD'})
pd.DataFrame(rows).to_csv(
    os.path.join(TAB_DIR, '13_significant_windows.csv'), index=False)

# ── Smoothed curves for plotting ──────────────────────────────────────────────
wt_sig_m  = smean(wt_sigs)
fad_sig_m = smean(fad_sigs)
wt_sig_e  = sse(wt_sigs)
fad_sig_e = sse(fad_sigs)

wt_ig_m   = smean(wt_igs)
fad_ig_m  = smean(fad_igs)
wt_ig_e   = sse(wt_igs)
fad_ig_e  = sse(fad_igs)

ig_diff = gaussian_filter1d(fad_igs.mean(0) - wt_igs.mean(0), sigma=SMOOTH_SIGMA)
log_p   = gaussian_filter1d(-np.log10(np.clip(pvals_fdr, 1e-10, 1)), sigma=5)
thresh  = -np.log10(0.05)

# ── 4-panel vertical figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig.subplots_adjust(hspace=0.35)

# Panel 1: Mean signals
ax = axes[0]
ax.fill_between(t, wt_sig_m - wt_sig_e, wt_sig_m + wt_sig_e,
                alpha=0.2, color='#2980b9')
ax.fill_between(t, fad_sig_m - fad_sig_e, fad_sig_m + fad_sig_e,
                alpha=0.2, color='#c0392b')
ax.plot(t, wt_sig_m,  '#2980b9', lw=1.8, label=f'WT (n={len(wt_rec)})')
ax.plot(t, fad_sig_m, '#c0392b', lw=1.8, label=f'5xFAD (n={len(fad_rec)})')
ax.set_ylabel('Amplitude (norm.)', fontsize=10)
ax.set_title('A.  Mean ERG Chirp Amplitude Trace', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 2: Mean Integrated Gradients per group
ax = axes[1]
ax.fill_between(t, wt_ig_m - wt_ig_e, wt_ig_m + wt_ig_e, alpha=0.2, color='#2980b9')
ax.fill_between(t, fad_ig_m - fad_ig_e, fad_ig_m + fad_ig_e, alpha=0.2, color='#c0392b')
ax.plot(t, wt_ig_m,  '#2980b9', lw=1.8, label='WT')
ax.plot(t, fad_ig_m, '#c0392b', lw=1.8, label='5xFAD')
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_ylabel('Integrated Gradients', fontsize=10)
ax.set_title('B.  Mean Integrated Gradients toward 5xFAD  '
             '(relative to mean-WT baseline)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 3: IG difference (5xFAD − WT) with significance shading
ax = axes[2]
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.plot(t, ig_diff, color='#8e44ad', lw=1.8, label='ΔIG = 5xFAD − WT')
ax.fill_between(t, 0, ig_diff, where=ig_diff > 0,
                color='#c0392b', alpha=0.25, label='5xFAD > WT')
ax.fill_between(t, 0, ig_diff, where=ig_diff < 0,
                color='#2980b9', alpha=0.25, label='WT > 5xFAD')
for s_, e_ in sig_windows:
    ax.axvspan(s_, e_, ymin=0, ymax=1, color='gold', alpha=0.35, zorder=0,
               label='Significant (FDR q<0.05)' if s_ == sig_windows[0][0] else '')
ax.set_ylabel('ΔIG  (5xFAD − WT)', fontsize=10)
ax.set_title('C.  Difference in Integrated Gradients  '
             '(gold = FDR-corrected significant)', fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
# Deduplicate legend entries
seen = {}
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = h
ax.legend(seen.values(), seen.keys(), fontsize=8); ax.grid(alpha=0.3)

# Panel 4: −log10(FDR p-value)
ax = axes[3]
ax.plot(t, log_p, color='#27ae60', lw=1.5)
ax.fill_between(t, 0, log_p, where=log_p >= thresh,
                color='#27ae60', alpha=0.35, label=f'FDR q<0.05')
ax.fill_between(t, 0, log_p, where=log_p < thresh,
                color='#27ae60', alpha=0.10)
ax.axhline(thresh, color='red', ls='--', lw=1.5,
           label=f'FDR q = 0.05  (−log₁₀ = {thresh:.2f})')
ax.set_xlabel('Sample index (Chirp Amplitude Segment)', fontsize=10)
ax.set_ylabel('−log₁₀(FDR q)', fontsize=10)
ax.set_title('D.  Statistical Significance — Mann-Whitney U,  BH FDR Correction',
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle('Integrated Gradients & Statistical Analysis\n'
             'Chirp Amplitude Sweep — WT vs 5xFAD  |  5-fold held-out subjects',
             fontweight='bold', fontsize=12)
plt.savefig(os.path.join(FIG_DIR, '13_c_integrated_grads.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 13_c_integrated_grads.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D: TemporalStatPool Feature Analysis
# The 96 values fed to the FC classifier (mean/max/std of each of 32 filters).
# Mann-Whitney U test comparing WT vs 5xFAD per feature, BH FDR correction.
# This is the most direct "statistical proof" of which learned features
# discriminate the genotypes, because these are exactly what the model uses.
# ══════════════════════════════════════════════════════════════════════════════
print("Figure D: TemporalStatPool feature analysis ...")

tsp_wt  = np.array([r['tsp'] for r in wt_rec])   # [N_wt,  96]
tsp_fad = np.array([r['tsp'] for r in fad_rec])  # [N_fad, 96]

# Feature names: <stat>_ch<k>
stat_names = (['mean'] * 32 + ['max'] * 32 + ['std'] * 32)
feat_names = [f'{s}_ch{k%32}' for k, s in enumerate(stat_names)]

# Pointwise tests
tsp_pvals = np.array([
    mannwhitneyu(tsp_wt[:, fi], tsp_fad[:, fi], alternative='two-sided').pvalue
    for fi in range(96)
])
_, tsp_pvals_fdr, _, _ = multipletests(tsp_pvals, method='fdr_bh')
tsp_sig = tsp_pvals_fdr < 0.05

# Effect size: rank-biserial correlation r = |U - n1*n2/2| / (n1*n2/2)
n1, n2 = len(wt_rec), len(fad_rec)
tsp_effect = np.array([
    abs(mannwhitneyu(tsp_wt[:, fi], tsp_fad[:, fi]).statistic - n1*n2/2) / (n1*n2/2)
    for fi in range(96)
])

print(f"  TSP features significant (FDR q<0.05): {tsp_sig.sum()} / 96")
top5_idx_p = np.argsort(tsp_pvals)[:5]
print("  Top-5 by uncorrected p-value:")
for fi in top5_idx_p:
    print(f"    {feat_names[fi]:>12s}  p_raw={tsp_pvals[fi]:.4f}  "
          f"q_BH={tsp_pvals_fdr[fi]:.4f}  r={tsp_effect[fi]:.3f}")

# Save full TSP stats table
tsp_rows = []
for fi in range(96):
    tsp_rows.append({
        'feature'     : feat_names[fi],
        'stat_type'   : stat_names[fi],
        'channel'     : fi % 32,
        'WT_mean'     : tsp_wt[:, fi].mean(),
        'FAD_mean'    : tsp_fad[:, fi].mean(),
        'p_raw'       : tsp_pvals[fi],
        'p_BH'        : tsp_pvals_fdr[fi],
        'effect_r'    : tsp_effect[fi],
        'significant' : bool(tsp_sig[fi]),
    })
pd.DataFrame(tsp_rows).to_csv(
    os.path.join(TAB_DIR, '13_tsp_feature_stats.csv'), index=False)

# Build figure
fig = plt.figure(figsize=(16, 12))
fig.subplots_adjust(hspace=0.5, wspace=0.4)

# ── Panel 1: -log10(FDR p) heatmap across 3 stat types × 32 channels ─────────
ax1 = fig.add_subplot(3, 1, 1)
log_p_tsp = -np.log10(np.clip(tsp_pvals_fdr, 1e-10, 1)).reshape(3, 32)
im = ax1.imshow(log_p_tsp, aspect='auto', cmap='hot',
                vmin=0, vmax=max(2, log_p_tsp.max()))
plt.colorbar(im, ax=ax1, label='−log₁₀(FDR q)')
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(['Mean', 'Max', 'Std'], fontsize=9)
ax1.set_xlabel('Filter channel index (0–31)', fontsize=9)
ax1.set_title('A.  −log₁₀(FDR q) for Each TemporalStatPool Feature  '
              '(rows: stat type, cols: filter channel)',
              fontweight='bold')
ax1.axhline(0.5, color='white', lw=0.5, alpha=0.4)
ax1.axhline(1.5, color='white', lw=0.5, alpha=0.4)
# Mark significant features
sig_2d = tsp_sig.reshape(3, 32)
for row in range(3):
    for col in range(32):
        if sig_2d[row, col]:
            ax1.plot(col, row, 'c*', ms=8, zorder=5)
thresh_line = -np.log10(0.05)
ax1.set_title(ax1.get_title() +
              f'\n(★ = FDR q<0.05, dashed line = q=0.05 threshold, '
              f'{tsp_sig.sum()} significant)', fontsize=9)

# ── Panel 2: Effect size heatmap ─────────────────────────────────────────────
ax2 = fig.add_subplot(3, 1, 2)
eff_2d = tsp_effect.reshape(3, 32)
im2 = ax2.imshow(eff_2d, aspect='auto', cmap='viridis', vmin=0, vmax=1)
plt.colorbar(im2, ax=ax2, label='Effect size r')
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['Mean', 'Max', 'Std'], fontsize=9)
ax2.set_xlabel('Filter channel index (0–31)', fontsize=9)
ax2.set_title('B.  Rank-Biserial Correlation Effect Size per Feature',
              fontweight='bold')
ax2.axhline(0.5, color='white', lw=0.5, alpha=0.4)
ax2.axhline(1.5, color='white', lw=0.5, alpha=0.4)

# ── Panel 3: Violin plots for top-6 features by effect size ──────────────────
ax3 = fig.add_subplot(3, 1, 3)
top6_idx = np.argsort(tsp_pvals)[:6]   # lowest uncorrected p-values
positions = np.arange(len(top6_idx))
vp_wt  = ax3.violinplot([tsp_wt[:, fi]  for fi in top6_idx],
                         positions=positions - 0.2,
                         widths=0.35, showmedians=True)
vp_fad = ax3.violinplot([tsp_fad[:, fi] for fi in top6_idx],
                         positions=positions + 0.2,
                         widths=0.35, showmedians=True)
for pc in vp_wt['bodies']:
    pc.set_facecolor('#2980b9'); pc.set_alpha(0.6)
for pc in vp_fad['bodies']:
    pc.set_facecolor('#c0392b'); pc.set_alpha(0.6)
for part in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
    vp_wt[part].set_color('#2980b9')
    vp_fad[part].set_color('#c0392b')

ax3.set_xticks(positions)
ax3.set_xticklabels([feat_names[fi] for fi in top6_idx], fontsize=8, rotation=15)
ax3.set_ylabel('Feature value (post-norm.)', fontsize=9)
ax3.set_title('C.  Top-6 Features by Effect Size  (blue=WT, red=5xFAD)\n'
              f'(p-values BH-corrected, annotated where q<0.1)',
              fontweight='bold')

# Annotate p-values above each pair
for pos, fi in zip(positions, top6_idx):
    p = tsp_pvals_fdr[fi]
    y_top = max(tsp_wt[:, fi].max(), tsp_fad[:, fi].max())
    label = (f'q={p:.3f}' if p >= 0.001 else 'q<0.001') if p < 0.1 else f'q={p:.2f}'
    color = 'darkgreen' if p < 0.05 else ('darkorange' if p < 0.1 else 'gray')
    ax3.text(pos, y_top * 1.05, label,
             ha='center', fontsize=7, fontweight='bold', color=color)

# Legend proxy
from matplotlib.patches import Patch
ax3.legend(handles=[Patch(color='#2980b9', alpha=0.6, label='WT'),
                    Patch(color='#c0392b', alpha=0.6, label='5xFAD')],
           fontsize=9)
ax3.grid(axis='y', alpha=0.3)

plt.suptitle('TemporalStatPool Feature Analysis — ImprovedBinaryCNN\n'
             'The 96 features (mean/max/std × 32 channels) fed to the FC classifier\n'
             'directly quantify which learned filter responses differ between genotypes',
             fontweight='bold', fontsize=11)
plt.savefig(os.path.join(FIG_DIR, '13_d_tsp_features.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 13_d_tsp_features.png")

print(f"\n✓ Interpretability analysis complete.")
print(f"  Figures → {FIG_DIR}")
print(f"  Tables  → {TAB_DIR}")
print(f"\nIG completeness check (mean |delta|): "
      f"{np.mean([r['ig_delta'] for r in records]):.5f}")
print(f"Significant IG time points (FDR q<0.05): {sig_mask.sum()} / {T}")
print(f"Significant TSP features   (FDR q<0.05): {tsp_sig.sum()} / 96")
