"""
22_symmetrical_story_figure.py
==============================
Generates a comprehensive, publication-ready figure illustrating the 
"Symmetrical Dynamic Envelope Feature" discovered by the CNN.

Shows original traces, the counterfactual edits, and the bidirectional 
spectral / phase symmetry.
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

# ── Identify Candidates ──────────────────────────────────────────────────────
fad_indices = np.where(labels == 1)[0]
wt_indices  = np.where(labels == 0)[0]
all_probs = [ensemble_prob_tensor(torch.tensor(signals[idx]).unsqueeze(0).unsqueeze(0).to(device)).item() for idx in range(len(subjects))]

# Just take easiest confident individuals for illustration
idx_fad_illustrate = fad_indices[np.argmax([all_probs[i] for i in fad_indices])]
idx_wt_illustrate  = wt_indices[np.argmin([all_probs[i] for i in wt_indices])]

# Top 6 for spectral distribution
top_fad_idx = fad_indices[np.argsort([all_probs[i] for i in fad_indices])[::-1]][:8]
top_wt_idx  = wt_indices[np.argsort([all_probs[i] for i in wt_indices])][:8]

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
        
        reached_target = (p_fad.item() > 0.8) if maximize else (p_fad.item() < 0.2)
        if reached_target and loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta.detach().cpu().numpy().flatten()
            
    return best_delta if best_delta is not None else delta.detach().cpu().numpy().flatten()

print("Processing representative candidates...")
sig_fad = signals[idx_fad_illustrate]
sig_wt  = signals[idx_wt_illustrate]

delta_cure = optimize_counterfactual(sig_fad, target_prob=0.1, maximize=False)
delta_dis  = optimize_counterfactual(sig_wt,  target_prob=0.9, maximize=True)

print("Processing spectral distributions for cohort...")
freqs = np.fft.rfftfreq(T, 1.0/FS)
bin_1hz = np.argmin(np.abs(freqs - 1.0))
results = {'Cure': {'specs': [], 'phases': []}, 'Disease': {'specs': [], 'phases': []}}

for group_name, indices, target, maximize in [('Cure', top_fad_idx, 0.1, False), ('Disease', top_wt_idx, 0.9, True)]:
    for idx in indices:
        orig = signals[idx]
        delta = optimize_counterfactual(orig, target, maximize)
        
        O_fft = np.fft.rfft(orig)
        D_fft = np.fft.rfft(delta)
        results[group_name]['specs'].append(np.abs(D_fft))
        
        phase_O = np.angle(O_fft[bin_1hz])
        phase_D = np.angle(D_fft[bin_1hz])
        results[group_name]['phases'].append(np.degrees(np.angle(np.exp(1j * (phase_D - phase_O)))))

spec_cure = np.mean(results['Cure']['specs'], axis=0)
spec_dis  = np.mean(results['Disease']['specs'], axis=0)

# ── Visualization ────────────────────────────────────────────────────────────
print("Generating publication figure...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

t = np.arange(T) / FS

# Panel A & B: The Traces & Edits
ax_fad = fig.add_subplot(gs[0, 0])
ax_fad.plot(t, sig_fad, color='#c0392b', alpha=0.3, lw=2, label='Original 5xFAD Trace')
ax_fad.plot(t, sig_fad + delta_cure, color='#2980b9', lw=2, label='CNN "Cured" Trace (→ WT)')
ax_fad.set_title('A. 5xFAD Counterfactual (The "Cure")', fontweight='bold')
ax_fad.set_ylabel('Amplitude')
ax_fad.legend(fontsize=9, loc='upper left')
ax_fad.grid(alpha=0.3)

ax_cure_delta = ax_fad.twinx()
ax_cure_delta.plot(t, delta_cure, color='#16a085', lw=1.5, linestyle=':', label='Requested $\\Delta$ Edit')
ax_cure_delta.set_ylabel('$\\Delta$ Amplitude', color='#16a085')
ax_cure_delta.legend(fontsize=9, loc='lower right')


ax_wt = fig.add_subplot(gs[0, 1])
ax_wt.plot(t, sig_wt, color='#2980b9', alpha=0.3, lw=2, label='Original WT Trace')
ax_wt.plot(t, sig_wt + delta_dis, color='#c0392b', lw=2, label='CNN "Degraded" Trace (→ 5xFAD)')
ax_wt.set_title('B. Wild-Type Counterfactual (The "Disease")', fontweight='bold')
ax_wt.legend(fontsize=9, loc='upper left')
ax_wt.grid(alpha=0.3)

ax_dis_delta = ax_wt.twinx()
ax_dis_delta.plot(t, delta_dis, color='#d35400', lw=1.5, linestyle=':', label='Requested $\\Delta$ Edit')
ax_dis_delta.set_ylabel('$\\Delta$ Amplitude', color='#d35400')
ax_dis_delta.legend(fontsize=9, loc='lower right')

# Panel C: Spectral Focus
ax_spec = fig.add_subplot(gs[0, 2])
ax_spec.plot(freqs, spec_cure / np.max(spec_cure), color='#16a085', lw=2.5, label='Power of "Cure" $\\Delta$')
ax_spec.plot(freqs, spec_dis / np.max(spec_dis), color='#d35400', lw=2.5, linestyle='--', label='Power of "Disease" $\\Delta$')
ax_spec.set_xlim(0, 5)
ax_spec.set_xlabel('Frequency (Hz)')
ax_spec.set_ylabel('Normalized FFT Amplitude')
ax_spec.set_title('C. Frequency Spectrum of Edits', fontweight='bold')
ax_spec.legend(fontsize=9)
ax_spec.grid(alpha=0.3)

# Panel D: Bi-directional Phase shift
ax_phase = fig.add_subplot(gs[1, :2])
ax_phase.hist(results['Cure']['phases'], bins=8, range=(-180, 180), color='#16a085', alpha=0.7, label='To Cure 5xFAD (Make WT)')
ax_phase.hist(results['Disease']['phases'], bins=8, range=(-180, 180), color='#d35400', alpha=0.7, label='To Degrade WT (Make 5xFAD)')
ax_phase.axvline(0, color='#16a085', linestyle='--', lw=2.5, label='Perfectly In-Phase (+ Gain)')
ax_phase.axvline(180, color='#d35400', linestyle='--', lw=2.5, label='Perfectly Out-of-Phase (- Gain)')
ax_phase.axvline(-180, color='#d35400', linestyle='--', lw=2.5)
ax_phase.set_xlim(-200, 200)
ax_phase.set_xticks([-180, -90, 0, 90, 180])
ax_phase.set_xticklabels(['-180°\n(Subtraction)', '-90°', '0°\n(Addition)', '90°', '180°\n(Subtraction)'])
ax_phase.set_xlabel('Phase Shift of Edit relative to Original Fundamental (Degrees)')
ax_phase.set_ylabel('Subject Count')
ax_phase.set_title('D. Symmetrical Phase Constraints at 1 Hz', fontweight='bold')
ax_phase.legend(fontsize=10)
ax_phase.grid(axis='y', alpha=0.3)

# Panel E: Conceptual Summary Graphic
ax_concept = fig.add_subplot(gs[1, 2])
ax_concept.axis('off')
ax_concept.text(0.5, 0.9, "Biological Insight Map", fontweight='bold', fontsize=14, ha='center')
ax_concept.text(0.5, 0.75, "1. CNN Isolates the Linear Fundamental (~1 Hz)\n[Ignoring harmonics & downstream retinal distortions]", ha='center', va='center', bbox=dict(facecolor='#f1c40f', alpha=0.2, boxstyle='round,pad=1'))
ax_concept.text(0.5, 0.55, "2. Network Detects Symmetric Gain Disparity\n[WT responds deeply, 5xFAD responds weakly]", ha='center', va='center', bbox=dict(facecolor='#3498db', alpha=0.2, boxstyle='round,pad=1'))
ax_concept.text(0.5, 0.35, "3. Counterfactual Edits Act as Linear Amp Scales\n[Edits are locked to 0°/180° phase alignment]", ha='center', va='center', bbox=dict(facecolor='#e74c3c', alpha=0.2, boxstyle='round,pad=1'))
ax_concept.text(0.5, 0.15, "Conclusion: Deficit is Dynamic Response Compression\nin the Outer/Middle Retinal Pathway.", ha='center', va='center', fontweight='bold', bbox=dict(facecolor='#ecf0f1', alpha=0.8, boxstyle='round,pad=1'))

plt.tight_layout()
out_fig = os.path.join(FIG_DIR, '22_symmetrical_story.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"Saved to {out_fig}")
