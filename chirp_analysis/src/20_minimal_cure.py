"""
20_minimal_cure.py
==================
Manifold-Constrained Counterfactuals ("The Minimal Cure") (Strategy 3).

Instead of asking what optimal sequence maximizes P(5xFAD) globally, this script 
takes actual 5xFAD subject traces and asks: "What is the smallest physiological 
perturbation needed to fool the CNN into diagnosing this subject as Wild-Type?"

Procedure:
  1. Take a true 5xFAD subject trace.
  2. Use gradient descent on the input signal to minimize P(5xFAD).
  3. Strongly penalize:
     - L2 norm of the change (keep edits minimal).
     - Temporal derivative of the change (smoothness: don't just add high-frequency noise).
  4. The resulting "counterfactual" trace shows exactly what shapes the CNN
     wants to "fix" to declare the mouse healthy.

Outputs:
  results/figures/20_minimal_cure.png
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

FIG_DIR = os.path.join(ROOT, 'results', 'figures')
MOD_DIR = os.path.join(ROOT, 'results', 'models')
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

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

# ── Load 5-fold ensemble ─────────────────────────────────────────────────────
print("Loading trained CNN ensemble...")
models = []
for fold in range(1, 6):
    m = ImprovedBinaryCNN().to(device)
    state = torch.load(os.path.join(MOD_DIR, f'12_improved_amplitude_fold_{fold}.pt'), map_location=device)
    m.load_state_dict(state)
    m.eval()
    for param in m.parameters(): param.requires_grad = False
    models.append(m)

def ensemble_prob_tensor(x_t):
    """ Returns mean probability of class 1 (5xFAD) for tensor [1, 1, T] """
    outs = [m(x_t) for m in models]
    probs = [F.softmax(out, dim=1)[0, 1] for out in outs]
    return torch.stack(probs).mean()
    
# ── Identify Top 5xFAD Candidates ────────────────────────────────────────────
fad_indices = np.where(labels == 1)[0]
fad_probs = []
for idx in fad_indices:
    orig_t = torch.tensor(signals[idx]).unsqueeze(0).unsqueeze(0).to(device)
    fad_probs.append(ensemble_prob_tensor(orig_t).item())

# Sort by how confident the model is (highest to lowest)
top_fad_idx = fad_indices[np.argsort(fad_probs)[::-1]]

# ── Optimization Loop ────────────────────────────────────────────────────────
print("Generating Minimal Cures...")

def optimize_cure(orig_signal_np, target_prob=0.1, lr=5e-3, steps=1500, alpha=0.5, beta=50.0):
    """
    Finds perturbation delta that makes the model predict < target_prob.
    alpha: penalizes L2 norm of delta
    beta: penalizes first temporal derivative of delta (encourages smoothness)
    """
    orig_t = torch.tensor(orig_signal_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    delta = torch.zeros_like(orig_t, requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    best_delta = None
    best_loss = float('inf')
    
    for step in range(steps):
        optimizer.zero_grad()
        # The modified signal
        x_mod = orig_t + delta
        
        # Current prob of 5xFAD
        p_fad = ensemble_prob_tensor(x_mod)
        
        # Loss components
        # 1. Classification task: P(5xFAD) should approach target_prob (e.g. 0.1 for Wild-Type)
        loss_cls = F.mse_loss(p_fad, torch.tensor(target_prob, device=device))
        
        # 2. Sparsity/Magnitude penalty
        loss_l2 = torch.norm(delta) / delta.nelement()
        
        # 3. Smoothness penalty (TV regularization / First Derivative penalty)
        diff = delta[:, :, 1:] - delta[:, :, :-1]
        loss_smooth = torch.norm(diff) / diff.nelement()
        
        # Total loss
        loss = loss_cls + alpha * loss_l2 + beta * loss_smooth
        loss.backward()
        optimizer.step()
        
        if p_fad.item() < 0.2 and loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta.detach().cpu().numpy().flatten()
            
        if step % 300 == 0:
            pass # can print progress if needed
            
    if best_delta is None:
        best_delta = delta.detach().cpu().numpy().flatten()
    return best_delta, ensemble_prob_tensor(orig_t + torch.tensor(best_delta).unsqueeze(0).unsqueeze(0).to(device)).item()

num_to_plot = min(4, len(top_fad_idx))
cures = []

for i in range(num_to_plot):
    idx = top_fad_idx[i]
    subj = subjects[idx]
    orig_sig = signals[idx]
    p_orig = fad_probs[np.where(fad_indices == idx)[0][0]]
    
    print(f"  Optimizing Cure for {subj} (Initial P(5xFAD): {p_orig:.3f})")
    delta_np, p_new = optimize_cure(orig_sig)
    print(f"    Done! Final P(5xFAD): {p_new:.3f}")
    
    cures.append({
        'subject': subj,
        'orig': orig_sig,
        'delta': delta_np,
        'p_orig': p_orig,
        'p_new': p_new
    })

# ── Visualization ────────────────────────────────────────────────────────────
print("Generating visualization...")
fig, axes = plt.subplots(num_to_plot, 2, figsize=(15, 3.5 * num_to_plot))
fig.subplots_adjust(hspace=0.4)

t = np.arange(len(signals[0]))

for i, res in enumerate(cures):
    ax_trace = axes[i, 0] if num_to_plot > 1 else axes[0]
    ax_delta = axes[i, 1] if num_to_plot > 1 else axes[1]
    
    orig = res['orig']
    delta = res['delta']
    mod = orig + delta
    
    # Left subplot: Traces
    ax_trace.plot(t, orig, color='#c0392b', alpha=0.6, lw=1.5, label=f'Original 5xFAD (P={res["p_orig"]:.2f})')
    ax_trace.plot(t, mod, color='#2980b9', alpha=0.9, lw=1.5, label=f'Minimal Cure (P={res["p_new"]:.2f})')
    ax_trace.set_title(f'Counterfactual Transformation: {res["subject"]}', fontweight='bold')
    ax_trace.set_ylabel('Amplitude')
    ax_trace.legend(fontsize=9)
    ax_trace.grid(alpha=0.3)
    
    # Right subplot: The exact Delta edit
    ax_delta.plot(t, delta, color='#8e44ad', lw=2)
    ax_delta.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Shade regions of the delta
    y1 = np.where(delta > 0, delta, 0)
    y2 = np.where(delta < 0, delta, 0)
    ax_delta.fill_between(t, 0, y1, color='#27ae60', alpha=0.3, label='Added Amplitude')
    ax_delta.fill_between(t, 0, y2, color='#c0392b', alpha=0.3, label='Reduced Amplitude')
    
    ax_delta.set_title(f'The "Biological Edit" (Delta)\nAdded components needed to fool CNN', fontweight='bold')
    ax_delta.set_ylabel('$\Delta$ Amplitude')
    if i == 0: ax_delta.legend(fontsize=9)
    ax_delta.grid(alpha=0.3)

plt.suptitle('Manifold-Constrained Counterfactuals ("Minimal Cure")\nWhat exact wave shapes are missing from 5xFAD mice?', fontweight='bold', fontsize=14, y=0.98)
out_fig = os.path.join(FIG_DIR, '20_minimal_cure.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved {out_fig}")
print("Done!")
