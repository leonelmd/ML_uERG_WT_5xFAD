"""
19_virtual_blockade.py
======================
Virtual Pharmacological Blockade (Strategy 1).

Computational equivalent of applying APB/PDA to the biological retina to block
specific circuits. In the Chirp Amplitude segment (1 Hz sine sweep), the response 
can be decomposed into:
  1. Low-frequency Baseline Drift (< 0.5 Hz)
  2. Fundamental Tracking (0.5 - 1.5 Hz): Primarily photoreceptor and bipolar linear response.
  3. Harmonics / Nonlinearities (> 1.5 Hz): Inner retina and nonlinear pathway distortions.

We selectively "silence" these components by subtracting them from the raw 
signal, feeding the modified trace into the trained CNN, and quantifying the 
drop in 5xFAD prediction confidence.

Outputs
-------
  results/figures/19_virtual_blockade.png
  results/tables/19_virtual_blockade.csv
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

from dataset import ERGChirpDataset
from models  import ImprovedBinaryCNN

# ── Params ───────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_DIR   = os.path.join(ROOT, 'data', 'cache')

FIG_DIR = os.path.join(ROOT, 'results', 'figures')
TAB_DIR = os.path.join(ROOT, 'results', 'tables')
MOD_DIR = os.path.join(ROOT, 'results', 'models')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

FS = 250.0  # Sampling freq for chirp amplitude segment

# ── Filtering functions ──────────────────────────────────────────────────────
def butter_lowpass(cutoff, fs=FS, order=4):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_bandpass(lowcut, highcut, fs=FS, order=4):
    return butter(order, [lowcut, highcut], fs=fs, btype='band', analog=False)

def butter_highpass(cutoff, fs=FS, order=4):
    return butter(order, cutoff, fs=fs, btype='high', analog=False)

def extract_drift(data):
    b, a = butter_lowpass(cutoff=0.5)
    return filtfilt(b, a, data)

def extract_fundamental(data):
    b, a = butter_bandpass(lowcut=0.5, highcut=1.5)
    return filtfilt(b, a, data)

def extract_harmonics(data):
    b, a = butter_highpass(cutoff=1.5)
    return filtfilt(b, a, data)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading dataset...")
df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label = {r.Subject: (1 if '5xFAD' in r.Group else 0) for _, r in df_meta.iterrows()}

dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment='amplitude', cache_dir=CACHE_DIR)
all_trial_subjs = [s.split('_trial_')[0] for s in dataset.subjects]
valid_idx       = [i for i, s in enumerate(all_trial_subjs) if s in subject_to_label]

# Accumulate subject-level signals
from collections import defaultdict
trial_accum = defaultdict(list)
for i in valid_idx:
    sig, _, name = dataset[i]
    subj = name.split('_trial_')[0]
    arr = sig.float().numpy().flatten()
    trial_accum[subj].append(arr)

subjects, signals, labels = [], [], []
for subj, trials in sorted(trial_accum.items()):
    subjects.append(subj)
    signals.append(np.mean(trials, axis=0))
    labels.append(subject_to_label[subj])

signals = np.array(signals, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)
N, T = signals.shape
print(f"Loaded {N} subjects (WT={np.sum(labels==0)}, 5xFAD={np.sum(labels==1)}), T={T}")

# ── Load 5-fold ensemble ─────────────────────────────────────────────────────
print("Loading trained CNN ensemble...")
models = []
for fold in range(1, 6):
    m = ImprovedBinaryCNN().to(device)
    state = torch.load(os.path.join(MOD_DIR, f'12_improved_amplitude_fold_{fold}.pt'), map_location=device)
    m.load_state_dict(state)
    m.eval()
    models.append(m)

def ensemble_prob(signal_np):
    x = torch.tensor(signal_np, dtype=torch.float32).reshape(1, 1, -1).to(device)
    with torch.no_grad():
        probs = [F.softmax(m(x), dim=1)[0, 1].item() for m in models]
    return float(np.mean(probs))

# ── Virtual Blockade Experiment ──────────────────────────────────────────────
print("Running virtual pharmacological blockade...")
results = []
for i, (subj, sig, lbl) in enumerate(zip(subjects, signals, labels)):
    drift = extract_drift(sig)
    fund  = extract_fundamental(sig)
    harm  = extract_harmonics(sig)
    
    # Intact
    p_intact = ensemble_prob(sig)
    
    # Pharmacological blockades (silencing the reconstructed component)
    sig_no_drift = sig - drift
    sig_no_fund  = sig - fund
    sig_no_harm  = sig - harm
    
    p_no_drift = ensemble_prob(sig_no_drift)
    p_no_fund  = ensemble_prob(sig_no_fund)
    p_no_harm  = ensemble_prob(sig_no_harm)
    
    results.append({
        'Subject': subj,
        'Label': '5xFAD' if lbl == 1 else 'WT',
        'P_Intact': p_intact,
        'P_No_Drift': p_no_drift,
        'P_No_Fund': p_no_fund,
        'P_No_Harm': p_no_harm,
        'Drop_from_Drift': p_intact - p_no_drift if lbl == 1 else p_no_drift - p_intact,
        'Drop_from_Fund':  p_intact - p_no_fund if lbl == 1 else p_no_fund - p_intact,
        'Drop_from_Harm':  p_intact - p_no_harm if lbl == 1 else p_no_harm - p_intact
    })

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(TAB_DIR, '19_virtual_blockade.csv'), index=False)
print("  Saved 19_virtual_blockade.csv")

# ── Group statistics ─────────────────────────────────────────────────────────
df_fad = df_res[df_res['Label'] == '5xFAD']
print("\nMean confidence P(5xFAD) for true 5xFAD subjects:")
print(f"  Intact (baseline)   : {df_fad['P_Intact'].mean():.3f}")
print(f"  Blocked Drift       : {df_fad['P_No_Drift'].mean():.3f} (Drop: {df_fad['Drop_from_Drift'].mean():.3f})")
print(f"  Blocked Fundamental : {df_fad['P_No_Fund'].mean():.3f} (Drop: {df_fad['Drop_from_Fund'].mean():.3f})")
print(f"  Blocked Harmonics   : {df_fad['P_No_Harm'].mean():.3f} (Drop: {df_fad['Drop_from_Harm'].mean():.3f})")

# ── Figure: Visualization of Dictionary Projection & Drops ───────────────────
print("Generating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.35, wspace=0.3)
t = np.arange(T)

# Best correctly classified 5xFAD subject trace
best_fad = df_fad.loc[df_fad['P_Intact'].idxmax()]
best_idx = subjects.index(best_fad['Subject'])
sig = signals[best_idx]
fund = extract_fundamental(sig)
harm = extract_harmonics(sig)

ax = axes[0, 0]
ax.plot(t, sig, color='#c0392b', lw=1.5, alpha=0.3, label='Raw ERG (Chirp Amplitude)')
ax.plot(t, fund, color='#2980b9', lw=2, label='Fundamental (~1 Hz)')
ax.set_title(f'A. Biological Filter: Linear Response\nSubject {best_fad["Subject"]}')
ax.set_ylabel('Amplitude (norm.)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t, sig, color='#c0392b', lw=1.5, alpha=0.3, label='Raw ERG')
ax.plot(t, harm, color='#8e44ad', lw=2, label='Harmonics / Nonlinearities (> 1.5 Hz)')
ax.set_title(f'B. Biological Filter: Nonlinear Response\nSubject {best_fad["Subject"]}')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Drops in confidence for 5xFAD subjects
ax = axes[1, 0]
box_data = [df_fad['Drop_from_Drift'], df_fad['Drop_from_Fund'], df_fad['Drop_from_Harm']]
ax.boxplot(box_data, patch_artist=True,
          boxprops=dict(facecolor='#ecf0f1', color='black'),
          medianprops=dict(color='#c0392b', lw=2))
ax.axhline(0, color='gray', ls='--', alpha=0.5)
import numpy.random as rnd
for i, d in enumerate(box_data):
    x = rnd.normal(i + 1, 0.04, size=len(d))
    ax.scatter(x, d, alpha=0.7, color='#34495e', s=25)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Drift (<0.5Hz)', 'Fundamental (~1Hz)', 'Harmonics (>1.5Hz)'])
ax.set_ylabel('Drop in P(5xFAD)')
ax.set_title('C. Virtual Blockade Impact (True 5xFAD Subjects)\nPositive = CNN confidence fell when filter blocked')
ax.grid(axis='y', alpha=0.3)

# Compare P(5xFAD) distributions
ax = axes[1, 1]
vp = ax.violinplot([df_fad['P_Intact'], df_fad['P_No_Drift'], df_fad['P_No_Fund'], df_fad['P_No_Harm']],
                  showmedians=True)
for pc in vp['bodies']:
    pc.set_facecolor('#c0392b')
    pc.set_alpha(0.5)
vp['cmedians'].set_color('black')
ax.axhline(0.5, color='gray', ls='--', label='Classification Boundary')
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Intact', 'No Drift', 'No Fund.', 'No Harmonics'])
ax.set_ylabel('CNN Confidence: P(5xFAD)')
ax.set_title('D. Network Predictions Under Blockade\n(True 5xFAD Subjects)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Virtual Pharmacological Blockade via Band-Specific Silencing\nTesting linear vs non-linear retinal pathways in the Chirp amplitude response', fontweight='bold', fontsize=12)
plt.savefig(os.path.join(FIG_DIR, '19_virtual_blockade.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved 19_virtual_blockade.png")
print("\nDone!")
