"""
21_cure_spectral_analysis.py
============================
Bi-directional Manifold-Constrained Counterfactuals.

Analyzes the spectral and phase properties of the CNN's modifications for BOTH:
1. "The Cure": Pushing 5xFAD subjects towards WT (Target P(5xFAD) < 0.1)
2. "The Disease": Pushing WT subjects towards 5xFAD (Target P(5xFAD) > 0.9)

We compute the amplitude and phase relationships between the original responses
and the required Deltas.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
MOD_DIR     = os.path.join(ROOT, 'results', 'models')
FIG_DIR     = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
FS = 250.0  # Sampling freq

# ── Load data & Models ───────────────────────────────────────────────────────
df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label = {r.Subject: (1 if '5xFAD' in r.Group else 0) for _, r in df_meta.iterrows()}

dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment='amplitude', cache_dir=CACHE_DIR)
all_trial_subjs = [s.split('_trial_')[0] for s in dataset.subjects]
valid_idx       = [i for i, s in enumerate(all_trial_subjs) if s in subject_to_label]

from collections import defaultdict
trial_accum = defaultdict(list)
for i in valid_idx:
    sig, _, name = dataset[i]
    arr = sig.float().numpy().flatten()
    trial_accum[name.split('_trial_')[0]].append(arr)

subjects, signals, labels = [], [], []
for subj, trials in sorted(trial_accum.items()):
    subjects.append(subj)
    signals.append(np.mean(trials, axis=0))
    labels.append(subject_to_label[subj])

signals = np.array(signals, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)
T = signals.shape[1]

models = []
for fold in range(1, 6):
    m = ImprovedBinaryCNN().to(device)
    m.load_state_dict(torch.load(os.path.join(MOD_DIR, f'12_improved_amplitude_fold_{fold}.pt'), map_location=device))
    m.eval()
    for param in m.parameters(): param.requires_grad = False
    models.append(m)

def ensemble_prob_tensor(x_t):
    outs = [m(x_t) for m in models]
    probs = [F.softmax(out, dim=1)[0, 1] for out in outs]
    return torch.stack(probs).mean()

# ── Identify Top Candidates ──────────────────────────────────────────────────
fad_indices = np.where(labels == 1)[0]
wt_indices  = np.where(labels == 0)[0]

all_probs = [ensemble_prob_tensor(torch.tensor(signals[idx]).unsqueeze(0).unsqueeze(0).to(device)).item() for idx in range(len(subjects))]

fad_probs = [all_probs[idx] for idx in fad_indices]
wt_probs  = [all_probs[idx] for idx in wt_indices]

# Top highest confidence 5xFAD and mathematically most confident WT (lowest P)
top_fad_idx = fad_indices[np.argsort(fad_probs)[::-1]][:8]
top_wt_idx  = wt_indices[np.argsort(wt_probs)][:8]

def optimize_counterfactual(orig_signal_np, target_prob, maximize=False, lr=5e-3, steps=1000, alpha=0.5, beta=50.0):
    orig_t = torch.tensor(orig_signal_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    delta = torch.zeros_like(orig_t, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    best_loss = float('inf'); best_delta = None
    for _ in range(steps):
        optimizer.zero_grad()
        p_fad = ensemble_prob_tensor(orig_t + delta)
        loss_cls = F.mse_loss(p_fad, torch.tensor(target_prob, device=device))
        loss_l2 = torch.norm(delta) / delta.nelement()
        diff = delta[:, :, 1:] - delta[:, :, :-1]
        loss_smooth = torch.norm(diff) / diff.nelement()
        
        loss = loss_cls + alpha * loss_l2 + beta * loss_smooth
        loss.backward()
        optimizer.step()
        
        # Stop condition relies on direction
        reached_target = (p_fad.item() > 0.8) if maximize else (p_fad.item() < 0.2)
        
        if reached_target and loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta.detach().cpu().numpy().flatten()
            
    return best_delta if best_delta is not None else delta.detach().cpu().numpy().flatten()

print("Extracting spectral features for BOTH directions...")
freqs = np.fft.rfftfreq(T, 1.0/FS)
bin_1hz = np.argmin(np.abs(freqs - 1.0))

results = {'5xFAD_to_WT': {'delta_spectra': [], 'phases': []},
           'WT_to_5xFAD': {'delta_spectra': [], 'phases': []}}

for group_name, indices, target, maximize in [('5xFAD_to_WT', top_fad_idx, 0.1, False), 
                                              ('WT_to_5xFAD', top_wt_idx, 0.9, True)]:
    for idx in indices:
        orig = signals[idx]
        delta = optimize_counterfactual(orig, target, maximize)
        
        O_fft = np.fft.rfft(orig)
        D_fft = np.fft.rfft(delta)
        
        results[group_name]['delta_spectra'].append(np.abs(D_fft))
        
        phase_O = np.angle(O_fft[bin_1hz])
        phase_D = np.angle(D_fft[bin_1hz])
        
        diff_deg = np.degrees(np.angle(np.exp(1j * (phase_D - phase_O))))
        results[group_name]['phases'].append(diff_deg)

D_mean_fad = np.mean(results['5xFAD_to_WT']['delta_spectra'], axis=0)
D_mean_wt  = np.mean(results['WT_to_5xFAD']['delta_spectra'], axis=0)

# ── Visualization ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.subplots_adjust(wspace=0.3)

# Panel A: Spectral Power of the Deltas
ax1 = axes[0]
ax1.plot(freqs, D_mean_fad / np.max(D_mean_fad), color='#2980b9', lw=2.5, label='$\Delta$ (Cure 5xFAD $\\rightarrow$ WT)')
ax1.plot(freqs, D_mean_wt / np.max(D_mean_wt), color='#c0392b', lw=2.5, linestyle='--', label='$\Delta$ (Degrade WT $\\rightarrow$ 5xFAD)')
ax1.set_xlim(0, 5)
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Normalized Amplitude')
ax1.set_title('A. Frequency Domain of the CNN\'s Counterfactuals', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Panel B: Phase Shift at the Fundamental
ax2 = axes[1]
ax2.hist(results['5xFAD_to_WT']['phases'], bins=10, range=(-180, 180), color='#2980b9', alpha=0.6, label='Cure (5xFAD $\\rightarrow$ WT)')
ax2.hist(results['WT_to_5xFAD']['phases'], bins=10, range=(-180, 180), color='#c0392b', alpha=0.6, label='Degrade (WT $\\rightarrow$ 5xFAD)')

ax2.axvline(0, color='#2980b9', linestyle='--', lw=2, label='0° (Perfectly In-Phase)')
ax2.axvline(180, color='#c0392b', linestyle='-.', lw=2, label='180° (Out-of-Phase / Subtracting)')
ax2.axvline(-180, color='#c0392b', linestyle='-.', lw=2)

ax2.set_xlim(-180, 180)
ax2.set_xticks([-180, -90, 0, 90, 180])
ax2.set_xlabel('Phase Difference Between $\Delta$ and Original Signal (Degrees)')
ax2.set_ylabel('Number of Subjects')
ax2.set_title(f'B. Phase Shift Analysis at strictly {freqs[bin_1hz]:.2f} Hz', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('Bi-Directional Spectral Analysis of Counterfactuals\nTesting the symmetry of the CNN\'s generative edits', fontweight='bold')
out = os.path.join(FIG_DIR, '21_cure_spectral_analysis.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Done! Saved to {out}")
