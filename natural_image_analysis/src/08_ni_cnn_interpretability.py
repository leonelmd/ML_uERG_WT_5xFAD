"""
08_ni_cnn_interpretability.py
==============================
Interpretability analysis for the Improved NI CNN (ImprovedNICNN_NoAge)
trained on Natural Image MEA responses (WT vs 5xFAD).

Architecture note
-----------------
Input: [B, 10, 2500]  — 10 image repetitions as channels, 2500 time points.
InstanceNorm1d(10) normalises each repetition channel independently.
Last conv layer: [B, 32, 20].  TemporalStatPool → [B, 96] → FC → [B, 2].

Methods
-------
1. Layer-1 Kernel Visualisation
   conv[0].weight is [8, 10, 15].  Two views:
   a) Channel-averaged filter waveforms (mean ± SD across 5 folds × 10 channels).
   b) Per-filter [10 channels × 15 taps] weight heatmap (most discriminative filter).

2. Grad-CAM
   Gradient of 5xFAD logit w.r.t. last ReLU activations [B, 32, 20].
   Upsampled to 2500 points for overlay.

3. Integrated Gradients  (Sundararajan et al., 2017)
   Baseline: mean WT signal [1, 10, 2500].
   IG result: [10, 2500] — averaged over repetition channels → [2500] for stats.
   Completeness guaranteed with trapezoidal Riemann approximation (50 steps).

4. Statistical Comparison
   Pointwise Mann-Whitney U on channel-averaged IG (WT vs 5xFAD), BH FDR (q<0.05).

5. TemporalStatPool Feature Analysis
   96 learned filter statistics (mean/max/std × 32 channels) compared between
   genotypes — the direct inputs to the FC classifier.

Outputs
-------
  results/figures/08_a_ni_kernels.png
  results/figures/08_b_ni_gradcam.png
  results/figures/08_c_ni_integrated_grads.png
  results/figures/08_d_ni_tsp_features.png
  results/tables/08_ni_significant_windows.csv
  results/tables/08_ni_tsp_feature_stats.csv

Usage (from natural_image_analysis/ folder)
---------------------------------------------
    python src/08_ni_cnn_interpretability.py
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

from models import (NaturalImageDataset, ImprovedNICNN_NoAge,
                    parse_ni_metadata)

# ── Paths ─────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'natural_image_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_PATH  = os.path.join(ROOT, 'data', 'cache', 'ni_dataset.pt')

FIG_DIR = os.path.join(ROOT, 'results', 'figures')
TAB_DIR = os.path.join(ROOT, 'results', 'tables')
MOD_DIR = os.path.join(ROOT, 'results', 'models')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ── Params (must match 01_train_ni_cnn.py) ────────────────────────────────────
N_FOLDS      = 5
RANDOM_STATE = 42
IG_STEPS     = 50
SMOOTH_SIGMA = 15   # Gaussian smoothing (samples) for visualisation

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Parsing metadata ...")
metadata = parse_ni_metadata(META_CSV)

print("Loading dataset ...")
dataset = NaturalImageDataset(DATA_DIR, metadata, cache_path=CACHE_PATH)

subjects   = np.array(dataset.all_subjects)                        # [N]
all_labels = np.array([dataset.all_labels[i].item()
                        for i in range(len(dataset))])             # [N]
N = len(dataset)
print(f"Dataset: {N} subjects  WT={sum(all_labels==0)}  5xFAD={sum(all_labels==1)}\n")

# Get input shape from first sample
sample_x, _, _, _ = dataset[0]
C, T = sample_x.shape[-2], sample_x.shape[-1]   # channels=10, time=2500
t    = np.arange(T)
print(f"Input shape: {C} channels × {T} time points\n")


# ══════════════════════════════════════════════════════════════════════════════
# Attribution helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_device(x):
    """Convert to [1, C, T] float32 on device."""
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    return x.float().reshape(1, C, T).to(device)


def grad_cam(model, x, target_class=1):
    """
    1D Grad-CAM from last ReLU of model.conv.
    x: [1, C, T] on device.
    Returns upsampled [T] float32 array in [0, 1].
    """
    model.eval()
    acts, grads = {}, {}

    def fwd_hook(m, inp, out):
        acts['v'] = out

    def bwd_hook(m, g_in, g_out):
        grads['v'] = g_out[0]

    hf = model.conv[-1].register_forward_hook(fwd_hook)
    hb = model.conv[-1].register_full_backward_hook(bwd_hook)

    x_in = x.clone().detach().requires_grad_(True)
    out  = model(x_in)
    model.zero_grad()
    out[0, target_class].backward()
    hf.remove(); hb.remove()

    A     = acts['v'][0]                                   # [32, T']
    G     = grads['v'][0]                                  # [32, T']
    alpha = G.mean(dim=-1, keepdim=True)                   # [32, 1]
    cam   = F.relu((alpha * A).sum(dim=0))                 # [T']
    cam   = cam / (cam.max() + 1e-8)
    cam_np = cam.detach().cpu().float().numpy()

    T_last = cam_np.shape[0]
    cam_up = np.interp(np.linspace(0, T_last - 1, T),
                       np.arange(T_last), cam_np)
    return cam_up                                          # [T], in [0,1]


def integrated_gradients(model, x, baseline, target_class=1, n_steps=IG_STEPS):
    """
    Integrated Gradients (Sundararajan 2017).
    Baseline: mean WT signal [1, C, T].
    Returns signed attribution [C, T] and completeness error.
    """
    model.eval()
    alphas = torch.linspace(0, 1, n_steps + 1, device=device)
    interp = (baseline + alphas.view(-1, 1, 1) * (x - baseline)).detach()
    interp.requires_grad_(True)

    out   = model(interp.view(-1, C, T))                   # [n+1, 2]
    score = out[:, target_class].sum()
    score.backward()

    grads = interp.grad                                    # [n+1, 1, C, T] or [n+1, C, T]
    # Handle potential batch dim
    if grads.dim() == 4:
        grads = grads.squeeze(1)                           # [n+1, C, T]

    # Trapezoidal rule
    avg_grads = (grads[1:-1].sum(dim=0) +
                 (grads[0] + grads[-1]) / 2) / n_steps    # [C, T]
    ig  = ((x.squeeze(0) - baseline.squeeze(0)) * avg_grads).detach().cpu().float().numpy()

    # Completeness check
    with torch.no_grad():
        F_x    = model(x)[0, target_class].item()
        F_base = model(baseline)[0, target_class].item()
    delta = abs(ig.sum() - (F_x - F_base))

    return ig, delta    # [C, T], scalar


# ══════════════════════════════════════════════════════════════════════════════
# Compute mean-WT baseline (shape [1, C, T])
# ══════════════════════════════════════════════════════════════════════════════
print("Computing mean-WT baseline ...")
wt_sigs_raw = []
for i in range(N):
    if all_labels[i] == 0:
        x_i, _, _, _ = dataset[i]
        wt_sigs_raw.append(x_i.float().numpy() if hasattr(x_i, 'numpy')
                           else np.array(x_i, dtype=np.float32))

mean_wt_np      = np.mean(wt_sigs_raw, axis=0)             # [C, T]
baseline_tensor = torch.tensor(mean_wt_np, dtype=torch.float32
                                ).unsqueeze(0).to(device)   # [1, C, T]


# ══════════════════════════════════════════════════════════════════════════════
# Main loop: fold-disjoint attributions
# ══════════════════════════════════════════════════════════════════════════════
print("Computing fold-disjoint attributions ...")
skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
records = []

for fold, (tr_idx, vl_idx) in enumerate(
        skf.split(subjects, all_labels), 1):

    model = ImprovedNICNN_NoAge().to(device)
    model.load_state_dict(
        torch.load(os.path.join(MOD_DIR, f'01_ni_cnn_noage_fold_{fold}.pt'),
                   map_location=device, weights_only=True))
    model.eval()

    for idx in vl_idx:
        x_raw, _, lbl_t, subj = dataset[idx]
        lbl = int(lbl_t.item() if hasattr(lbl_t, 'item') else lbl_t)
        sig = (x_raw.float().numpy() if hasattr(x_raw, 'numpy')
               else np.array(x_raw, dtype=np.float32))    # [C, T]
        x   = _to_device(sig)                              # [1, C, T]

        with torch.no_grad():
            logits   = model(x)
            pred     = logits.argmax(1).item()
            prob_fad = F.softmax(logits, dim=1)[0, 1].item()

        cam = grad_cam(model, x)
        ig, delta = integrated_gradients(model, x, baseline_tensor)

        # TemporalStatPool features [96]
        _store = {}
        def _feat_hook(m, inp, out, _s=_store):
            _s['feat'] = out.detach()
        h = model.conv[-1].register_forward_hook(_feat_hook)
        with torch.no_grad():
            model(x)
        h.remove()
        A_feat = _store['feat'][0]                         # [32, T']
        tsp = torch.cat([
            A_feat.mean(dim=-1),
            A_feat.amax(dim=-1),
            A_feat.std(dim=-1, unbiased=False),
        ], dim=0).cpu().float().numpy()                    # [96]

        records.append({
            'subject' : subj,
            'label'   : lbl,
            'pred'    : pred,
            'prob_fad': prob_fad,
            'signal'  : sig,      # [C, T]
            'cam'     : cam,      # [T]
            'ig'      : ig,       # [C, T]
            'ig_delta': delta,
            'tsp'     : tsp,      # [96]
        })

    fold_deltas = [r['ig_delta'] for r in records[-len(vl_idx):]]
    print(f"  Fold {fold}: {len(vl_idx)} subjects  "
          f"(mean IG error: {np.mean(fold_deltas):.5f})")

print(f"Total: {len(records)} subjects  "
      f"(WT={sum(r['label']==0 for r in records)}, "
      f"5xFAD={sum(r['label']==1 for r in records)})\n")

wt_rec  = [r for r in records if r['label'] == 0]
fad_rec = [r for r in records if r['label'] == 1]

# Channel-averaged IG: [N, T] — mean over C channels
wt_igs  = np.array([r['ig'].mean(axis=0) for r in wt_rec])    # [N_wt,  T]
fad_igs = np.array([r['ig'].mean(axis=0) for r in fad_rec])   # [N_fad, T]

# Channel-averaged signal: [N, T]
wt_sigs_1d  = np.array([r['signal'].mean(axis=0) for r in wt_rec])
fad_sigs_1d = np.array([r['signal'].mean(axis=0) for r in fad_rec])


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
    m = ImprovedNICNN_NoAge()
    m.load_state_dict(
        torch.load(os.path.join(MOD_DIR, f'01_ni_cnn_noage_fold_{fold}.pt'),
                   map_location='cpu', weights_only=True))
    # conv[0] = Conv1d(10, 8, k=15) → weight [8, 10, 15]
    all_kernels.append(m.conv[0].weight.detach().cpu().numpy())
all_kernels = np.array(all_kernels)   # [5, 8, 10, 15]

# Channel-averaged filters: mean over 10 input channels → [5, 8, 15]
k_ch_avg_all = all_kernels.mean(axis=2)        # [5, 8, 15]
k_mean = k_ch_avg_all.mean(axis=0)             # [8, 15]
k_std  = k_ch_avg_all.std(axis=0)              # [8, 15]
n_filt, n_taps = k_mean.shape

# Best filter to show as full [10×15] heatmap: highest std across taps/folds
filter_var = all_kernels.var(axis=(0, 2, 3))   # [8] variance across folds, ch, taps
best_filter = int(filter_var.argmax())
best_kern_heatmap = all_kernels.mean(axis=0)[best_filter]   # [10, 15] mean across folds

fig = plt.figure(figsize=(18, 8))

# Top row: channel-averaged waveforms (2 rows × 4 cols)
for col in range(n_filt):
    ax = fig.add_subplot(3, n_filt, col + 1)
    color = plt.cm.tab10(col)
    taps = np.arange(n_taps)
    ax.fill_between(taps, k_mean[col] - k_std[col],
                    k_mean[col] + k_std[col], alpha=0.25, color=color)
    ax.plot(taps, k_mean[col], color=color, lw=2)
    ax.axhline(0, color='gray', lw=0.7, ls='--')
    ax.set_title(f'F{col+1}', fontsize=9, fontweight='bold')
    if col == 0:
        ax.set_ylabel('Weight\n(ch-avg)', fontsize=7)
    ax.tick_params(labelsize=6); ax.grid(alpha=0.3)

# Middle row: FFT magnitude
for col in range(n_filt):
    ax = fig.add_subplot(3, n_filt, n_filt + col + 1)
    color = plt.cm.tab10(col)
    fft_mag = np.abs(np.fft.rfft(k_mean[col], n=256))
    freqs   = np.fft.rfftfreq(256)
    ax.plot(freqs, fft_mag, color=color, lw=1.5)
    peak_f = freqs[fft_mag.argmax()]
    ax.axvline(peak_f, color=color, ls=':', lw=1, alpha=0.8)
    ax.text(peak_f + 0.01, fft_mag.max() * 0.85, f'{peak_f:.2f}',
            fontsize=6, color=color)
    if col == 0:
        ax.set_ylabel('|FFT|', fontsize=7)
    ax.set_xlabel('Norm. freq.', fontsize=6)
    ax.tick_params(labelsize=6); ax.grid(alpha=0.3)

# Bottom: full [10-channel × 15-tap] heatmap for the most variable filter
ax_heat = fig.add_subplot(3, 1, 3)
im = ax_heat.imshow(best_kern_heatmap, aspect='auto', cmap='RdBu_r',
                    vmin=-best_kern_heatmap.abs().max() if hasattr(best_kern_heatmap, 'abs')
                    else -np.abs(best_kern_heatmap).max(),
                    vmax=np.abs(best_kern_heatmap).max())
plt.colorbar(im, ax=ax_heat, label='Weight')
ax_heat.set_yticks(range(C))
ax_heat.set_yticklabels([f'Rep {i+1}' for i in range(C)], fontsize=7)
ax_heat.set_xlabel('Tap index (0–14)', fontsize=9)
ax_heat.set_title(f'Full Filter Weights — Filter {best_filter+1} '
                  f'(most variable across folds)  |  10 input channels × 15 taps',
                  fontweight='bold')

plt.suptitle('Layer-1 Convolutional Filters — ImprovedNICNN_NoAge\n'
             'Top: channel-averaged waveforms (mean ± SD across 5 folds)  |  '
             'Middle: frequency response  |  Bottom: full weight heatmap',
             fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '08_a_ni_kernels.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 08_a_ni_kernels.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B: Grad-CAM
# ══════════════════════════════════════════════════════════════════════════════
print("Figure B: Grad-CAM ...")

wt_cams  = np.array([r['cam'] for r in wt_rec])
fad_cams = np.array([r['cam'] for r in fad_rec])

correct_wt  = [r for r in wt_rec  if r['pred'] == 0] or wt_rec
correct_fad = [r for r in fad_rec if r['pred'] == 1] or fad_rec
best_wt  = min(correct_wt,  key=lambda r: r['prob_fad'])
best_fad = max(correct_fad, key=lambda r: r['prob_fad'])

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.45, wspace=0.35)

# Row 0: mean ± SE channel-averaged signals
for ax, sigs, color, lbl, n in [
        (axes[0, 0], wt_sigs_1d,  '#2980b9', 'WT',    len(wt_rec)),
        (axes[0, 1], fad_sigs_1d, '#c0392b', '5xFAD', len(fad_rec))]:
    m_, s_ = smean(sigs), sse(sigs)
    ax.fill_between(t, m_-s_, m_+s_, alpha=0.2, color=color)
    ax.plot(t, m_, color=color, lw=1.5, label=f'{lbl} (n={n})')
    ax.set_ylabel('Response (ch-avg, norm.)', fontsize=9)
    ax.set_title(f'Mean NI Response — {lbl}', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Row 1: mean ± SE Grad-CAM
for ax, cams, color, lbl in [
        (axes[1, 0], wt_cams,  '#2980b9', 'WT'),
        (axes[1, 1], fad_cams, '#c0392b', '5xFAD')]:
    m_cam = smean(cams); s_cam = sse(cams)
    ax.fill_between(t, m_cam-s_cam, m_cam+s_cam, alpha=0.25, color='darkorange')
    ax.plot(t, m_cam, color='darkorange', lw=2,
            label=f'Grad-CAM ({lbl})')
    ax.set_ylim(0, None)
    ax.set_ylabel('Grad-CAM (norm., toward 5xFAD)', fontsize=9)
    ax.set_title(f'Mean Grad-CAM — {lbl} subjects', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Row 2: overlay on best representative traces
for ax, rec, lbl in [
        (axes[2, 0], best_wt,
         f"Best WT  (p_5xFAD={best_wt['prob_fad']:.2f})"),
        (axes[2, 1], best_fad,
         f"Best 5xFAD  (p_5xFAD={best_fad['prob_fad']:.2f})")]:
    sig = rec['signal'].mean(axis=0)           # channel-averaged [T]
    cam = gaussian_filter1d(rec['cam'], sigma=SMOOTH_SIGMA)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    y_lo = sig.min() - 0.15 * (sig.max() - sig.min())
    y_hi = sig.max() + 0.15 * (sig.max() - sig.min())
    cam_2d = np.tile(cam[np.newaxis, :], (30, 1))
    ax.imshow(cam_2d, aspect='auto',
              extent=[t[0], t[-1], y_lo, y_hi],
              cmap='YlOrRd', alpha=0.55, vmin=0, vmax=1,
              origin='lower', interpolation='bilinear')
    c = '#2980b9' if rec['label'] == 0 else '#c0392b'
    ax.plot(t, sig, color=c, lw=1.5, zorder=5, label=lbl)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel('Sample index', fontsize=9)
    ax.set_ylabel('Response (ch-avg, norm.)', fontsize=9)
    ax.set_title(f'Grad-CAM Overlay — {lbl}', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle('Grad-CAM Analysis — Natural Image MEA Responses  (toward 5xFAD)\n'
             '[Native Grad-CAM resolution ≈ 20 points, upsampled for overlay]',
             fontweight='bold', fontsize=11)
plt.savefig(os.path.join(FIG_DIR, '08_b_ni_gradcam.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 08_b_ni_gradcam.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C: Integrated Gradients + Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Figure C: Integrated Gradients + statistics ...")

print("  Running pointwise Mann-Whitney tests (N=2500) ...")
pvals = np.array([
    mannwhitneyu(wt_igs[:, ti], fad_igs[:, ti], alternative='two-sided').pvalue
    for ti in range(T)
])
_, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')
sig_mask = pvals_fdr < 0.05

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
    print(f"    Samples [{s_}:{e_}]  len={e_-s_+1}  ΔIG={diff_mean:.4f}")

# Save windows
rows = []
for s_, e_ in sig_windows:
    diff_mean = fad_igs[:, s_:e_+1].mean() - wt_igs[:, s_:e_+1].mean()
    rows.append({'start_sample': s_, 'end_sample': e_,
                 'length_samples': e_-s_+1,
                 'mean_delta_ig_5xFAD_minus_WT': round(diff_mean, 6),
                 'direction': '5xFAD>WT' if diff_mean > 0 else 'WT>5xFAD'})
pd.DataFrame(rows).to_csv(
    os.path.join(TAB_DIR, '08_ni_significant_windows.csv'), index=False)

# Smoothed curves
wt_sig_m  = smean(wt_sigs_1d);  wt_sig_e  = sse(wt_sigs_1d)
fad_sig_m = smean(fad_sigs_1d); fad_sig_e = sse(fad_sigs_1d)
wt_ig_m   = smean(wt_igs);      wt_ig_e   = sse(wt_igs)
fad_ig_m  = smean(fad_igs);     fad_ig_e  = sse(fad_igs)
ig_diff   = gaussian_filter1d(fad_igs.mean(0) - wt_igs.mean(0), sigma=SMOOTH_SIGMA)
log_p     = gaussian_filter1d(-np.log10(np.clip(pvals_fdr, 1e-10, 1)), sigma=5)
thresh    = -np.log10(0.05)

fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig.subplots_adjust(hspace=0.35)

# Panel 1: Mean signals
ax = axes[0]
ax.fill_between(t, wt_sig_m - wt_sig_e, wt_sig_m + wt_sig_e, alpha=0.2, color='#2980b9')
ax.fill_between(t, fad_sig_m - fad_sig_e, fad_sig_m + fad_sig_e, alpha=0.2, color='#c0392b')
ax.plot(t, wt_sig_m,  '#2980b9', lw=1.8, label=f'WT (n={len(wt_rec)})')
ax.plot(t, fad_sig_m, '#c0392b', lw=1.8, label=f'5xFAD (n={len(fad_rec)})')
ax.set_ylabel('Response (ch-avg)', fontsize=10)
ax.set_title('A.  Mean Channel-Averaged NI Response Trace', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 2: Mean IG
ax = axes[1]
ax.fill_between(t, wt_ig_m - wt_ig_e, wt_ig_m + wt_ig_e, alpha=0.2, color='#2980b9')
ax.fill_between(t, fad_ig_m - fad_ig_e, fad_ig_m + fad_ig_e, alpha=0.2, color='#c0392b')
ax.plot(t, wt_ig_m,  '#2980b9', lw=1.8, label='WT')
ax.plot(t, fad_ig_m, '#c0392b', lw=1.8, label='5xFAD')
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_ylabel('Integrated Gradients\n(ch-avg)', fontsize=10)
ax.set_title('B.  Mean Integrated Gradients toward 5xFAD  '
             '(relative to mean-WT baseline; channels averaged)',
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 3: IG difference with significance shading
ax = axes[2]
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.plot(t, ig_diff, color='#8e44ad', lw=1.8, label='ΔIG = 5xFAD − WT')
ax.fill_between(t, 0, ig_diff, where=ig_diff > 0, color='#c0392b', alpha=0.25,
                label='5xFAD > WT')
ax.fill_between(t, 0, ig_diff, where=ig_diff < 0, color='#2980b9', alpha=0.25,
                label='WT > 5xFAD')
for s_, e_ in sig_windows:
    ax.axvspan(s_, e_, color='gold', alpha=0.4, zorder=0,
               label='FDR q<0.05' if s_ == sig_windows[0][0] else '')
if not sig_windows:
    ax.text(T * 0.5, ig_diff.max() * 0.5,
            'No windows survive FDR correction\n(model uses global statistics)',
            ha='center', fontsize=9, color='gray', style='italic')
ax.set_ylabel('ΔIG  (5xFAD − WT)', fontsize=10)
ax.set_title('C.  IG Difference  (gold = FDR q<0.05)', fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    if l not in seen: seen[l] = h
ax.legend(seen.values(), seen.keys(), fontsize=8); ax.grid(alpha=0.3)

# Panel 4: -log10(FDR q)
ax = axes[3]
ax.plot(t, log_p, color='#27ae60', lw=1.5)
ax.fill_between(t, 0, log_p, where=log_p >= thresh, color='#27ae60', alpha=0.4,
                label=f'FDR q<0.05')
ax.fill_between(t, 0, log_p, where=log_p < thresh,  color='#27ae60', alpha=0.10)
ax.axhline(thresh, color='red', ls='--', lw=1.5,
           label=f'FDR q=0.05  (−log₁₀={thresh:.2f})')
ax.set_xlabel('Sample index (NI Response)', fontsize=10)
ax.set_ylabel('−log₁₀(FDR q)', fontsize=10)
ax.set_title('D.  Statistical Significance — Mann-Whitney U + BH FDR',
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle('Integrated Gradients & Statistical Analysis\n'
             'Natural Image MEA Responses — WT vs 5xFAD  |  5-fold held-out subjects',
             fontweight='bold', fontsize=12)
plt.savefig(os.path.join(FIG_DIR, '08_c_ni_integrated_grads.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 08_c_ni_integrated_grads.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D: TemporalStatPool Feature Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Figure D: TemporalStatPool feature analysis ...")

tsp_wt  = np.array([r['tsp'] for r in wt_rec])    # [N_wt,  96]
tsp_fad = np.array([r['tsp'] for r in fad_rec])   # [N_fad, 96]

stat_names = ['mean'] * 32 + ['max'] * 32 + ['std'] * 32
feat_names = [f'{s}_ch{k%32}' for k, s in enumerate(stat_names)]

tsp_pvals = np.array([
    mannwhitneyu(tsp_wt[:, fi], tsp_fad[:, fi], alternative='two-sided').pvalue
    for fi in range(96)
])
_, tsp_pvals_fdr, _, _ = multipletests(tsp_pvals, method='fdr_bh')
tsp_sig = tsp_pvals_fdr < 0.05

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

# Save stats table
tsp_rows = []
for fi in range(96):
    tsp_rows.append({
        'feature'    : feat_names[fi],
        'stat_type'  : stat_names[fi],
        'channel'    : fi % 32,
        'WT_mean'    : tsp_wt[:, fi].mean(),
        'FAD_mean'   : tsp_fad[:, fi].mean(),
        'p_raw'      : tsp_pvals[fi],
        'p_BH'       : tsp_pvals_fdr[fi],
        'effect_r'   : tsp_effect[fi],
        'significant': bool(tsp_sig[fi]),
    })
pd.DataFrame(tsp_rows).to_csv(
    os.path.join(TAB_DIR, '08_ni_tsp_feature_stats.csv'), index=False)

fig = plt.figure(figsize=(16, 12))
fig.subplots_adjust(hspace=0.5, wspace=0.4)

# Panel 1: -log10(FDR q) heatmap
ax1 = fig.add_subplot(3, 1, 1)
log_p_tsp = -np.log10(np.clip(tsp_pvals_fdr, 1e-10, 1)).reshape(3, 32)
im = ax1.imshow(log_p_tsp, aspect='auto', cmap='hot',
                vmin=0, vmax=max(2, log_p_tsp.max()))
plt.colorbar(im, ax=ax1, label='−log₁₀(FDR q)')
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(['Mean', 'Max', 'Std'], fontsize=9)
ax1.set_xlabel('Filter channel index (0–31)', fontsize=9)
sig_2d = tsp_sig.reshape(3, 32)
for row in range(3):
    for col in range(32):
        if sig_2d[row, col]:
            ax1.plot(col, row, 'c*', ms=8, zorder=5)
ax1.set_title(f'A.  −log₁₀(FDR q) per TSP Feature  '
              f'(★ = FDR q<0.05, {tsp_sig.sum()} significant / 96)',
              fontweight='bold')
ax1.axhline(0.5, color='white', lw=0.5, alpha=0.4)
ax1.axhline(1.5, color='white', lw=0.5, alpha=0.4)

# Panel 2: Effect size heatmap
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

# Panel 3: Violin plots — top-6 by raw p-value
ax3 = fig.add_subplot(3, 1, 3)
top6_idx = np.argsort(tsp_pvals)[:6]
positions = np.arange(len(top6_idx))
vp_wt  = ax3.violinplot([tsp_wt[:, fi]  for fi in top6_idx],
                         positions=positions - 0.2, widths=0.35, showmedians=True)
vp_fad = ax3.violinplot([tsp_fad[:, fi] for fi in top6_idx],
                         positions=positions + 0.2, widths=0.35, showmedians=True)
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
ax3.set_title('C.  Top-6 TSP Features by Raw p-Value  (blue=WT, red=5xFAD)',
              fontweight='bold')
for pos, fi in zip(positions, top6_idx):
    p = tsp_pvals_fdr[fi]
    y_top = max(tsp_wt[:, fi].max(), tsp_fad[:, fi].max())
    label = (f'q={p:.3f}' if p >= 0.001 else 'q<0.001') if p < 0.1 else f'q={p:.2f}'
    color = 'darkgreen' if p < 0.05 else ('darkorange' if p < 0.1 else 'gray')
    ax3.text(pos, y_top * 1.05, label,
             ha='center', fontsize=7, fontweight='bold', color=color)

from matplotlib.patches import Patch
ax3.legend(handles=[Patch(color='#2980b9', alpha=0.6, label='WT'),
                    Patch(color='#c0392b', alpha=0.6, label='5xFAD')], fontsize=9)
ax3.grid(axis='y', alpha=0.3)

plt.suptitle('TemporalStatPool Feature Analysis — ImprovedNICNN_NoAge\n'
             'The 96 features (mean/max/std × 32 channels) fed to the FC classifier',
             fontweight='bold', fontsize=11)
plt.savefig(os.path.join(FIG_DIR, '08_d_ni_tsp_features.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 08_d_ni_tsp_features.png")

print(f"\n✓ NI CNN interpretability analysis complete.")
print(f"  Figures → {FIG_DIR}")
print(f"  Tables  → {TAB_DIR}")
print(f"\nIG completeness check (mean |delta|): "
      f"{np.mean([r['ig_delta'] for r in records]):.5f}")
print(f"Significant IG time points (FDR q<0.05): {sig_mask.sum()} / {T}")
print(f"Significant TSP features   (FDR q<0.05): {tsp_sig.sum()} / 96")
