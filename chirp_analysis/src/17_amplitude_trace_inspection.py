"""
17_amplitude_trace_inspection.py
=================================
Visual inspection of the chirp amplitude segment for WT vs 5xFAD.

Panels
------
  A  Individual subject traces (light) + group mean ± SE (bold)
  B  Difference signal (mean_5xFAD − mean_WT) ± propagated SE;
     gold shading = FDR-significant windows (Mann-Whitney U, BH q<0.05)
  C  Cohen's d across time — direct readout of effect magnitude and sign
     (dashed lines at d = ±0.2, ±0.5, ±0.8 threshold conventions)

Automatic feature identification
----------------------------------
  Contiguous FDR-significant windows are printed to console and saved as a
  table.  For each window the script reports:
    • Time range (s), length (ms)
    • Direction  (5xFAD > WT  or  WT > 5xFAD)
    • Peak |d| in window
    • Mean amplitude of each group in window (handcrafted feature candidate)

Outputs
-------
  results/figures/17_amplitude_inspection.png
  results/tables/17_significant_windows.csv

Usage (from chirp_analysis/ folder)
-------------------------------------
    python src/17_amplitude_trace_inspection.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

from dataset import ERGChirpDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_DIR   = os.path.join(ROOT, 'data', 'cache')
FIG_DIR     = os.path.join(ROOT, 'results', 'figures')
TAB_DIR     = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

FS           = 250.0   # Hz
SMOOTH_VIS   = 20      # Gaussian sigma for visualisation (samples = 80 ms)
SMOOTH_STAT  = 5       # Gaussian sigma for smoothing stat curves (samples)
C_WT         = '#2980b9'
C_FAD        = '#c0392b'


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load + aggregate to subject-level means
# ══════════════════════════════════════════════════════════════════════════════
print("Loading dataset …")
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

subjects_data = []
for subj, trials in sorted(trial_accum.items()):
    subjects_data.append({
        'subject': subj,
        'signal':  np.mean(trials, axis=0).astype(np.float32),
        'label':   subject_to_label[subj],
    })

signals = np.array([d['signal'] for d in subjects_data], dtype=np.float32)
labels  = np.array([d['label']  for d in subjects_data], dtype=np.int32)
N, T    = signals.shape
ts      = np.arange(T) / FS   # time axis in seconds

wt_sigs  = signals[labels == 0]   # [N_wt,  T]
fad_sigs = signals[labels == 1]   # [N_fad, T]
N_wt, N_fad = len(wt_sigs), len(fad_sigs)

print(f"  {N} subjects  (WT={N_wt}, 5xFAD={N_fad})  "
      f"T={T} samples  ({T/FS:.1f} s)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Group statistics (computed on RAW, un-smoothed signals)
# ══════════════════════════════════════════════════════════════════════════════
mean_wt  = wt_sigs.mean(axis=0)
mean_fad = fad_sigs.mean(axis=0)
se_wt    = wt_sigs.std(axis=0,  ddof=1) / np.sqrt(N_wt)
se_fad   = fad_sigs.std(axis=0, ddof=1) / np.sqrt(N_fad)

diff     = mean_fad - mean_wt                     # positive → 5xFAD > WT
se_diff  = np.sqrt(se_wt**2 + se_fad**2)          # propagated SE of difference

# Cohen's d at every sample
pooled_sd = np.sqrt((wt_sigs.var(axis=0, ddof=1) +
                     fad_sigs.var(axis=0, ddof=1)) / 2 + 1e-8)
cohens_d  = diff / pooled_sd                      # signed


# ══════════════════════════════════════════════════════════════════════════════
# 3. Pointwise Mann-Whitney U + BH FDR correction
# ══════════════════════════════════════════════════════════════════════════════
print("Computing pointwise Mann-Whitney U tests …")
pvals = np.array([
    mannwhitneyu(wt_sigs[:, i], fad_sigs[:, i], alternative='two-sided').pvalue
    for i in range(T)
])
_, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')
sig_mask_fdr = pvals_fdr < 0.05
sig_mask_unc = pvals < 0.05   # uncorrected (exploratory)

# Smoothed |d| ≥ 0.5 mask (medium effect size after light smoothing)
abs_d_sm = np.abs(gaussian_filter1d(cohens_d, sigma=SMOOTH_STAT))
sig_mask_d = abs_d_sm >= 0.5

def _contiguous_windows(mask):
    windows = []
    in_win = False
    for i, s in enumerate(mask):
        if s and not in_win:
            w_start = i; in_win = True
        elif not s and in_win:
            windows.append((w_start, i - 1)); in_win = False
    if in_win:
        windows.append((w_start, len(mask) - 1))
    return windows

sig_windows     = _contiguous_windows(sig_mask_fdr)
sig_windows_unc = _contiguous_windows(sig_mask_unc)
sig_windows_d   = _contiguous_windows(sig_mask_d)

print(f"  Significant windows (FDR q<0.05):        {len(sig_windows)}")
print(f"  Significant windows (uncorrected p<0.05): {len(sig_windows_unc)}")
print(f"  Windows with smoothed |d| ≥ 0.5:          {len(sig_windows_d)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Report + save significant windows
# ══════════════════════════════════════════════════════════════════════════════
rows_win = []
for w_start, w_end in sig_windows:
    t_start_s = w_start / FS
    t_end_s   = w_end   / FS
    length_ms = (w_end - w_start + 1) / FS * 1000
    wt_mean_w  = wt_sigs[:,  w_start:w_end+1].mean()
    fad_mean_w = fad_sigs[:, w_start:w_end+1].mean()
    peak_d     = cohens_d[w_start:w_end+1]
    peak_abs_d = float(np.abs(peak_d).max())
    direction  = '5xFAD > WT' if fad_mean_w > wt_mean_w else 'WT > 5xFAD'

    rows_win.append({
        'start_sample': w_start, 'end_sample': w_end,
        't_start_s':    round(t_start_s, 3),
        't_end_s':      round(t_end_s,   3),
        'length_ms':    round(length_ms, 1),
        'direction':    direction,
        'peak_abs_d':   round(peak_abs_d, 3),
        'wt_mean':      round(wt_mean_w,  4),
        'fad_mean':     round(fad_mean_w, 4),
    })
    print(f"  [{t_start_s:.2f}–{t_end_s:.2f} s  ({length_ms:.0f} ms)]  "
          f"{direction}  peak|d|={peak_abs_d:.2f}")

df_win = pd.DataFrame(rows_win)
df_win.to_csv(os.path.join(TAB_DIR, '17_significant_windows.csv'), index=False)
print(f"  Saved: 17_significant_windows.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Figure
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding figure …")

# Smooth ONLY for visualisation
sm = lambda x: gaussian_filter1d(x, sigma=SMOOTH_VIS)
sm_s = lambda x: gaussian_filter1d(x, sigma=SMOOTH_STAT)

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.subplots_adjust(hspace=0.12, left=0.07, right=0.97, top=0.93, bottom=0.06)

# ── Panel A: individual traces + group means ──────────────────────────────────
ax = axes[0]

for sig in wt_sigs:
    ax.plot(ts, sm(sig), color=C_WT,  lw=0.6, alpha=0.18)
for sig in fad_sigs:
    ax.plot(ts, sm(sig), color=C_FAD, lw=0.6, alpha=0.18)

ax.fill_between(ts, sm(mean_wt  - se_wt),  sm(mean_wt  + se_wt),
                alpha=0.30, color=C_WT)
ax.fill_between(ts, sm(mean_fad - se_fad), sm(mean_fad + se_fad),
                alpha=0.30, color=C_FAD)
ax.plot(ts, sm(mean_wt),  C_WT,  lw=2.5,
        label=f'Mean WT  (n={N_wt}, ±SE)')
ax.plot(ts, sm(mean_fad), C_FAD, lw=2.5,
        label=f'Mean 5xFAD  (n={N_fad}, ±SE)')

# Shade detection layers (back to front: |d|, uncorrected, FDR)
for w_start, w_end in sig_windows_d:
    ax.axvspan(ts[w_start], ts[min(w_end, T-1)],
               color='#f39c12', alpha=0.12, zorder=0,
               label='|d|≥0.5' if (w_start, w_end) == sig_windows_d[0] else '')
for w_start, w_end in sig_windows_unc:
    ax.axvspan(ts[w_start], ts[min(w_end, T-1)],
               color='gold', alpha=0.20, zorder=1,
               label='p<0.05 (unc.)' if (w_start, w_end) == sig_windows_unc[0] else '')
for w_start, w_end in sig_windows:
    ax.axvspan(ts[w_start], ts[min(w_end, T-1)],
               color='gold', alpha=0.45, zorder=2,
               label='FDR q<0.05' if (w_start, w_end) == sig_windows[0] else '')

ax.set_ylabel('Amplitude (norm.)', fontsize=11)
ax.set_title('A.  Individual subject traces — Chirp Amplitude Segment  '
             f'(smoothed σ={SMOOTH_VIS} smp = {SMOOTH_VIS/FS*1000:.0f} ms)',
             fontweight='bold', fontsize=11)
handles, lbls = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, lbls):
    if l not in seen: seen[l] = h
ax.legend(seen.values(), seen.keys(), fontsize=9, loc='upper right')
ax.grid(alpha=0.25)

# ── Panel B: difference signal ────────────────────────────────────────────────
ax = axes[1]

d_sm    = sm_s(diff)
se_d_sm = sm_s(se_diff)
ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6)

ax.fill_between(ts, d_sm - se_d_sm, d_sm + se_d_sm, alpha=0.20, color='#8e44ad')
ax.fill_between(ts, 0, d_sm, where=d_sm >= 0,
                color=C_FAD, alpha=0.40, label='5xFAD > WT')
ax.fill_between(ts, 0, d_sm, where=d_sm < 0,
                color=C_WT,  alpha=0.40, label='WT > 5xFAD')
ax.plot(ts, d_sm, color='#2c2c2c', lw=1.5, alpha=0.8)

for w_start, w_end in sig_windows_d:
    ax.axvspan(ts[w_start], ts[min(w_end, T-1)],
               color='#f39c12', alpha=0.12, zorder=0,
               label='|d|≥0.5' if (w_start, w_end) == sig_windows_d[0] else '')
for w_start, w_end in sig_windows_unc:
    ax.axvspan(ts[w_start], ts[min(w_end, T-1)],
               color='gold', alpha=0.20, zorder=1,
               label='p<0.05 (unc.)' if (w_start, w_end) == sig_windows_unc[0] else '')
for w_start, w_end in sig_windows:
    ax.axvspan(ts[w_start], ts[min(w_end, T-1)],
               color='gold', alpha=0.45, zorder=2,
               label='FDR q<0.05' if (w_start, w_end) == sig_windows[0] else '')

ax.set_ylabel('Δ amplitude\n(5xFAD − WT)', fontsize=11)
ax.set_title('B.  Difference signal (5xFAD − WT) ± propagated SE  '
             '│  shading: orange=|d|≥0.5  │  gold=p<0.05 (unc.)  │  dark gold=FDR q<0.05',
             fontweight='bold', fontsize=11)
handles, lbls = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, lbls):
    if l not in seen: seen[l] = h
ax.legend(seen.values(), seen.keys(), fontsize=9)
ax.grid(alpha=0.25)

# ── Panel C: Cohen's d ────────────────────────────────────────────────────────
ax = axes[2]

d_vis = sm_s(cohens_d)
ax.axhline(0,    color='gray',  lw=0.8, ls='--', alpha=0.6)
for val, ls, lbl in [(0.2, ':', 'd=0.2 (small)'),
                     (0.5, '--','d=0.5 (medium)'),
                     (0.8, '-', 'd=0.8 (large)')]:
    ax.axhline( val, color='#aaa', lw=0.8, ls=ls, alpha=0.7)
    ax.axhline(-val, color='#aaa', lw=0.8, ls=ls, alpha=0.7)
ax.text(ts[-1]*0.99, 0.82, 'large',  ha='right', va='bottom', fontsize=7, color='#888')
ax.text(ts[-1]*0.99, 0.52, 'medium', ha='right', va='bottom', fontsize=7, color='#888')
ax.text(ts[-1]*0.99, 0.22, 'small',  ha='right', va='bottom', fontsize=7, color='#888')

ax.fill_between(ts, 0, d_vis, where=d_vis >= 0,
                color=C_FAD, alpha=0.45, label="5xFAD > WT")
ax.fill_between(ts, 0, d_vis, where=d_vis < 0,
                color=C_WT,  alpha=0.45, label="WT > 5xFAD")
ax.plot(ts, d_vis, color='#2c2c2c', lw=1.5, alpha=0.85)

for w_start, w_end in sig_windows_d:
    ax.axvspan(ts[w_start], ts[min(w_end, T-1)],
               color='#f39c12', alpha=0.15, zorder=0,
               label='|d|≥0.5' if (w_start, w_end) == sig_windows_d[0] else '')
for w_start, w_end in sig_windows_unc:
    ax.axvspan(ts[w_start], ts[min(w_end, T-1)],
               color='gold', alpha=0.22, zorder=1,
               label='p<0.05 (unc.)' if (w_start, w_end) == sig_windows_unc[0] else '')
for w_start, w_end in sig_windows:
    ax.axvspan(ts[w_start], ts[min(w_end, T-1)],
               color='gold', alpha=0.45, zorder=2,
               label='FDR q<0.05' if (w_start, w_end) == sig_windows[0] else '')

ax.set_ylabel("Cohen's d\n(signed)", fontsize=11)
ax.set_xlabel('Time within amplitude segment (s)', fontsize=11)
ax.set_title("C.  Effect size (Cohen's d) — where and how much do groups differ?",
             fontweight='bold', fontsize=11)
handles, lbls = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, lbls):
    if l not in seen: seen[l] = h
ax.legend(seen.values(), seen.keys(), fontsize=9)
ax.grid(alpha=0.25)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))

# ── Annotate significant windows on panel C ───────────────────────────────────
for row in rows_win:
    mid = (row['t_start_s'] + row['t_end_s']) / 2
    pk  = row['peak_abs_d'] * (1 if row['direction'] == '5xFAD > WT' else -1)
    offset = 0.08 if pk > 0 else -0.08
    axes[2].annotate(
        f"{row['t_start_s']:.1f}–{row['t_end_s']:.1f} s\n({row['length_ms']:.0f} ms)",
        xy=(mid, pk), xytext=(mid, pk + offset),
        ha='center', va='bottom' if pk > 0 else 'top',
        fontsize=7, color='darkgoldenrod', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='darkgoldenrod', lw=1.0),
    )

plt.suptitle('Chirp Amplitude Segment — WT vs 5xFAD Group Comparison\n'
             f'N={N_wt} WT  |  N={N_fad} 5xFAD  |  '
             f'FDR sig.: {len(sig_windows)}  │  unc. p<0.05: {len(sig_windows_unc)}  │  '
             f'|d|≥0.5: {len(sig_windows_d)}  │  '
             f'Duration {T/FS:.1f} s  @  {FS:.0f} Hz',
             fontweight='bold', fontsize=12)

out_fig = os.path.join(FIG_DIR, '17_amplitude_inspection.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_fig}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Console summary of candidate features
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("CANDIDATE HAND-CRAFTED FEATURES")
print(f"{'='*65}")

# FDR windows
if len(sig_windows) == 0:
    print("\n  [FDR q<0.05]  No significant windows (low power, N=23 per group).")
else:
    print(f"\n  [FDR q<0.05]  {len(sig_windows)} window(s):")
    for _, row in df_win.iterrows():
        print(f"    {row['t_start_s']:.2f}–{row['t_end_s']:.2f} s  "
              f"({row['length_ms']:.0f} ms)  |  {row['direction']}  "
              f"|  peak |d| = {row['peak_abs_d']:.2f}")
        print(f"      → Mean amplitude (WT={row['wt_mean']:.3f}, 5xFAD={row['fad_mean']:.3f})")

# Uncorrected p<0.05 windows (exploratory)
print(f"\n  [Uncorrected p<0.05]  {len(sig_windows_unc)} window(s)  (exploratory — not corrected for multiple comparisons):")
for w_start, w_end in sig_windows_unc[:10]:   # cap at 10
    t0, t1     = w_start / FS, w_end / FS
    length_ms  = (w_end - w_start + 1) / FS * 1000
    pk_d       = cohens_d[w_start:w_end+1]
    peak_abs_d = float(np.abs(pk_d).max())
    direction  = '5xFAD > WT' if diff[w_start:w_end+1].mean() > 0 else 'WT > 5xFAD'
    print(f"    {t0:.2f}–{t1:.2f} s  ({length_ms:.0f} ms)  |  {direction}  |  peak |d|={peak_abs_d:.2f}")
if len(sig_windows_unc) > 10:
    print(f"    … (+{len(sig_windows_unc)-10} more)")

# |d|≥0.5 windows — most useful for feature candidates given low power
print(f"\n  [Smoothed |d| ≥ 0.5 — FEATURE CANDIDATES]  {len(sig_windows_d)} window(s):")
for w_start, w_end in sig_windows_d:
    t0, t1     = w_start / FS, w_end / FS
    length_ms  = (w_end - w_start + 1) / FS * 1000
    pk_d       = cohens_d[w_start:w_end+1]
    peak_abs_d = float(np.abs(pk_d).max())
    wt_mean_w  = wt_sigs[:,  w_start:w_end+1].mean()
    fad_mean_w = fad_sigs[:, w_start:w_end+1].mean()
    direction  = '5xFAD > WT' if fad_mean_w > wt_mean_w else 'WT > 5xFAD'
    print(f"    {t0:.2f}–{t1:.2f} s  ({length_ms:.0f} ms)  |  {direction}  |  peak |d|={peak_abs_d:.2f}")
    print(f"      Suggested features:")
    print(f"        • Mean amplitude [{t0:.2f}–{t1:.2f} s]  (WT={wt_mean_w:.3f}, 5xFAD={fad_mean_w:.3f})")
    print(f"        • RMS in window  •  Peak-to-peak in window  •  Hilbert envelope mean in window")

# Overall Cohen's d summary
abs_d = np.abs(cohens_d)
top10_idx = np.argsort(abs_d)[-10:][::-1]
print(f"\n  Top-10 samples by |d| (un-smoothed):")
for idx in top10_idx:
    print(f"    t={ts[idx]:.3f} s  (sample {idx})  d={cohens_d[idx]:+.3f}")

print(f"\n  Max |d| anywhere: {abs_d.max():.3f}  at t={ts[abs_d.argmax()]:.3f} s")
print(f"  Mean |d| across all samples: {abs_d.mean():.3f}")
print(f"\n  Figures → {FIG_DIR}")
print(f"  Tables  → {TAB_DIR}")
