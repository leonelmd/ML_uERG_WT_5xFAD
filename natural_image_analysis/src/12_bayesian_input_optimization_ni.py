"""
12_bayesian_input_optimization_ni.py
======================================
Optimal stimulus design for the Natural Image CNN (ImprovedNICNN_NoAge).

Finds the input signal that maximises / minimises P(5xFAD) across the ensemble
of 5 trained ImprovedNICNN_NoAge models (script 07, 07_improved_ni_cnn_fold_{1-5}.pt).

Architecture note
-----------------
Input: [B, 10, 2500] — 10 repetitions as channels, 2500 time points.
InstanceNorm1d(10) normalises each repetition independently over time.

Optimization strategy
---------------------
We optimize a single prototype waveform of shape (2500,) representing
the "ideal" mean NI response, then replicate it 10× to form the model input.
PCA is fitted on subject-level mean responses (average of 10 reps per subject).

Two complementary approaches:
  1. Gradient-based activation maximization — differentiable, full T=2500
     resolution, 3 random starts, TV + L2 regularization.
  2. Bayesian optimization in PCA-reduced space — GP surrogate with EI
     acquisition, K=15 components, ±3σ bounds; optimises ensemble mean score.

Outputs
-------
  results/figures/12_a_optimal_signals.png
  results/figures/12_b_pca_coefficients.png
  results/figures/12_c_convergence.png
  results/figures/12_d_pca_landscape.png
  results/tables/12_optimization_results.csv

Usage (from natural_image_analysis/ folder)
---------------------------------------------
    pip install scikit-optimize
    python src/12_bayesian_input_optimization_ni.py
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
from sklearn.decomposition import PCA

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

from models import NaturalImageDataset, ImprovedNICNN_NoAge, parse_ni_metadata

# ── Paths ──────────────────────────────────────────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'natural_image_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_PATH  = os.path.join(ROOT, 'data', 'cache', 'ni_dataset.pt')

FIG_DIR = os.path.join(ROOT, 'results', 'figures')
TAB_DIR = os.path.join(ROOT, 'results', 'tables')
MOD_DIR = os.path.join(ROOT, 'results', 'models')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
K_PCA        = 15
GRAD_LR      = 0.05
GRAD_STEPS   = 500
TV_WEIGHT    = 1e-4
L2_WEIGHT    = 1e-5
BO_N_CALLS   = 200
BO_N_INIT    = 50
GRID_N       = 30
RANDOM_STATE = 42
N_REPS       = 10   # repetition channels in model input

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load ensemble (5 folds, eval mode, frozen)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading ensemble models ...")
models = []
for fold in range(1, 6):
    m = ImprovedNICNN_NoAge(input_ch=N_REPS).to(device)
    m.load_state_dict(
        torch.load(os.path.join(MOD_DIR, f'07_improved_ni_cnn_fold_{fold}.pt'),
                   map_location=device, weights_only=True))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    models.append(m)
print(f"  Loaded {len(models)} fold models.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Load dataset — extract subject-level mean signals
# ══════════════════════════════════════════════════════════════════════════════
print("Loading dataset ...")
metadata = parse_ni_metadata(META_CSV)
dataset  = NaturalImageDataset(DATA_DIR, metadata,
                               cache_path=CACHE_PATH)

# Each item: x=[10, 2500], label, subject.  Compute mean across 10 reps → [2500].
all_means     = []   # [N, 2500]  — mean across reps for each subject
all_full      = []   # [N, 10, 2500] — full 10-rep tensors
binary_labels = []
subjects_list = []

for i in range(len(dataset)):
    x, _, lbl, subj = dataset[i]
    x_np  = x.float().numpy()     # [10, 2500]
    mean_i = x_np.mean(axis=0)    # [2500]
    all_means.append(mean_i)
    all_full.append(x_np)
    binary_labels.append(int(lbl.item() if hasattr(lbl, 'item') else lbl))
    subjects_list.append(subj)

all_means     = np.array(all_means,  dtype=np.float32)   # [N, 2500]
all_full      = np.array(all_full,   dtype=np.float32)   # [N, 10, 2500]
binary_labels = np.array(binary_labels, dtype=np.int32)
T             = all_means.shape[1]   # 2500
t             = np.arange(T)

wt_means  = all_means[binary_labels == 0]   # [N_wt,  2500]
fad_means = all_means[binary_labels == 1]   # [N_fad, 2500]

mean_wt  = wt_means.mean(axis=0)
mean_fad = fad_means.mean(axis=0)
se_wt    = wt_means.std(axis=0)  / np.sqrt(len(wt_means))
se_fad   = fad_means.std(axis=0) / np.sqrt(len(fad_means))

print(f"  N_total={len(all_means)}  N_WT={len(wt_means)}  N_5xFAD={len(fad_means)}")
print(f"  Signal length T={T}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Scoring functions
# ══════════════════════════════════════════════════════════════════════════════

def to_model_input(signal_np: np.ndarray) -> torch.Tensor:
    """
    (2500,) numpy array → [1, 10, 2500] tensor on device.
    The single prototype waveform is replicated across all 10 rep channels.
    """
    sig = torch.tensor(signal_np, dtype=torch.float32)
    return sig.reshape(1, 1, -1).expand(1, N_REPS, T).contiguous().to(device)


def ensemble_score(signal_np: np.ndarray) -> float:
    """Numpy (2500,) in → ensemble-mean P(5xFAD) float out. No gradients."""
    x = to_model_input(signal_np)
    with torch.no_grad():
        probs = [F.softmax(m(x), dim=1)[0, 1].item() for m in models]
    return float(np.mean(probs))


def ensemble_log_prob_diff(x_param: torch.Tensor) -> torch.Tensor:
    """
    Differentiable: mean log P(5xFAD) across ensemble.
    x_param: [1, 1, T] leaf with requires_grad=True.
    Expands to [1, 10, T] internally (grad flows back via expand).
    """
    x_in = x_param.expand(1, N_REPS, T)
    return torch.stack([
        F.log_softmax(m(x_in), dim=1)[0, 1] for m in models
    ]).mean()


# Sanity check
score_wt  = ensemble_score(mean_wt)
score_fad = ensemble_score(mean_fad)
print(f"\nSanity check:")
print(f"  P(5xFAD | mean_WT)    = {score_wt:.4f}")
print(f"  P(5xFAD | mean_5xFAD) = {score_fad:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PCA in z-scored space (fitted on subject mean signals)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFitting PCA ...")


def znorm(sigs: np.ndarray) -> np.ndarray:
    """Per-sample z-score: mirrors InstanceNorm1d."""
    mu  = sigs.mean(axis=1, keepdims=True)
    std = sigs.std(axis=1, keepdims=True) + 1e-8
    return (sigs - mu) / std


all_znorm  = znorm(all_means)                              # [N, 2500]
pca        = PCA(n_components=K_PCA, random_state=RANDOM_STATE)
pca.fit(all_znorm)
all_coeffs = pca.transform(all_znorm)                      # [N, K]
pc_std     = all_coeffs.std(axis=0)                        # [K]
bounds     = [(-3 * s, 3 * s) for s in pc_std]

wt_coeffs  = all_coeffs[binary_labels == 0]
fad_coeffs = all_coeffs[binary_labels == 1]

print(f"  Explained variance (K={K_PCA}): "
      f"{pca.explained_variance_ratio_.sum()*100:.1f}%")


def encode(sig: np.ndarray) -> np.ndarray:
    """(2500,) signal → PCA coefficients in z-scored space."""
    return pca.transform(znorm(sig.reshape(1, -1)))[0]


def decode(coeffs: np.ndarray, target_std: float = None) -> np.ndarray:
    """PCA coefficients → (2500,) signal. Optionally rescale std for display."""
    sig = pca.inverse_transform(coeffs.reshape(1, -1))[0].astype(np.float32)
    if target_std is not None:
        sig = sig * target_std / (sig.std() + 1e-8)
    return sig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Gradient-based activation maximization (3 starts)
# ══════════════════════════════════════════════════════════════════════════════
print("\nGradient ascent optimization for 5xFAD (3 starts) ...")

rng = np.random.default_rng(RANDOM_STATE)
starts = [
    ('mean_WT',    mean_wt.copy()),
    ('mean_5xFAD', mean_fad.copy()),
    ('noise_0.01', rng.normal(0, 0.01, T).astype(np.float32)),
]

grad_results = []

for label, init_signal in starts:
    print(f"  Start: {label} (init score={ensemble_score(init_signal):.4f}) ...",
          flush=True)
    x = nn.Parameter(
        torch.tensor(init_signal, dtype=torch.float32).reshape(1, 1, T).to(device))
    opt = torch.optim.Adam([x], lr=GRAD_LR)

    for step in range(GRAD_STEPS):
        opt.zero_grad()
        loss  = -ensemble_log_prob_diff(x)
        loss += TV_WEIGHT * torch.diff(x[0, 0]).abs().mean()
        loss += L2_WEIGHT * x.pow(2).mean()
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            sc = ensemble_score(x.detach().cpu().numpy().flatten())
            print(f"    step {step+1:4d}  P(5xFAD)={sc:.4f}", flush=True)

    # Dense convergence curve (re-run)
    scores_dense = []
    x2 = nn.Parameter(
        torch.tensor(init_signal, dtype=torch.float32).reshape(1, 1, T).to(device))
    opt2 = torch.optim.Adam([x2], lr=GRAD_LR)
    for step in range(GRAD_STEPS):
        opt2.zero_grad()
        loss  = -ensemble_log_prob_diff(x2)
        loss += TV_WEIGHT * torch.diff(x2[0, 0]).abs().mean()
        loss += L2_WEIGHT * x2.pow(2).mean()
        loss.backward()
        opt2.step()
        if (step + 1) % 10 == 0:
            with torch.no_grad():
                sc = ensemble_score(x2.detach().cpu().numpy().flatten())
            scores_dense.append(sc)

    final_np    = x.detach().cpu().numpy().flatten()
    final_score = ensemble_score(final_np)
    final_display = final_np * mean_fad.std() / (final_np.std() + 1e-8)

    grad_results.append({
        'label':        label,
        'signal':       final_display,
        'signal_raw':   final_np,
        'score':        final_score,
        'scores_dense': scores_dense,
    })
    print(f"  → Final P(5xFAD) = {final_score:.4f}")

best_grad      = max(grad_results, key=lambda r: r['score'])
grad_optimal   = best_grad['signal']
grad_best_score = best_grad['score']
print(f"\nBest gradient start (5xFAD): '{best_grad['label']}'  "
      f"P(5xFAD)={grad_best_score:.4f}")

# ── Gradient descent: minimise P(5xFAD) → optimal WT ─────────────────────────
print("\nGradient descent optimization for WT (3 starts) ...")

grad_results_wt = []
for label, init_signal in starts:
    print(f"  Start: {label} (init score={ensemble_score(init_signal):.4f}) ...",
          flush=True)
    x = nn.Parameter(
        torch.tensor(init_signal, dtype=torch.float32).reshape(1, 1, T).to(device))
    opt = torch.optim.Adam([x], lr=GRAD_LR)

    for step in range(GRAD_STEPS):
        opt.zero_grad()
        loss  = ensemble_log_prob_diff(x)   # minimise log P(5xFAD)
        loss += TV_WEIGHT * torch.diff(x[0, 0]).abs().mean()
        loss += L2_WEIGHT * x.pow(2).mean()
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            sc = ensemble_score(x.detach().cpu().numpy().flatten())
            print(f"    step {step+1:4d}  P(5xFAD)={sc:.4f}", flush=True)

    # Dense convergence curve (re-run)
    scores_dense_wt = []
    x2 = nn.Parameter(
        torch.tensor(init_signal, dtype=torch.float32).reshape(1, 1, T).to(device))
    opt2 = torch.optim.Adam([x2], lr=GRAD_LR)
    for step in range(GRAD_STEPS):
        opt2.zero_grad()
        loss  = ensemble_log_prob_diff(x2)
        loss += TV_WEIGHT * torch.diff(x2[0, 0]).abs().mean()
        loss += L2_WEIGHT * x2.pow(2).mean()
        loss.backward()
        opt2.step()
        if (step + 1) % 10 == 0:
            with torch.no_grad():
                sc = ensemble_score(x2.detach().cpu().numpy().flatten())
            scores_dense_wt.append(sc)

    final_np    = x.detach().cpu().numpy().flatten()
    final_score = ensemble_score(final_np)
    final_display = final_np * mean_wt.std() / (final_np.std() + 1e-8)

    grad_results_wt.append({
        'label':        label,
        'signal':       final_display,
        'signal_raw':   final_np,
        'score':        final_score,
        'scores_dense': scores_dense_wt,
    })
    print(f"  → Final P(5xFAD) = {final_score:.4f}")

best_grad_wt    = min(grad_results_wt, key=lambda r: r['score'])
grad_wt_optimal = best_grad_wt['signal']
grad_wt_score   = best_grad_wt['score']
print(f"\nBest gradient start (WT): '{best_grad_wt['label']}'  "
      f"P(5xFAD)={grad_wt_score:.4f}  P(WT)={1-grad_wt_score:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Bayesian optimization in PCA-reduced space
# ══════════════════════════════════════════════════════════════════════════════
print("\nBayesian optimization ...")

# scikit-optimize compatibility shim for sklearn >= 1.6
try:
    import sklearn.utils.validation as _suv
    if not hasattr(_suv, 'check_is_fitted'):
        _suv.check_is_fitted = lambda *a, **kw: None
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_OK = True
except ImportError:
    SKOPT_OK = False
    print("  WARNING: scikit-optimize not available. "
          "Run `pip install scikit-optimize` and rerun. "
          "BO section will be skipped.")

if SKOPT_OK:
    def bo_objective(params):
        coeffs = np.array(params, dtype=np.float32)
        sig    = decode(coeffs)
        return -ensemble_score(sig)

    dimensions = [Real(lo, hi, name=f'pc{k}') for k, (lo, hi) in enumerate(bounds)]

    print(f"  Running GP minimize: {BO_N_CALLS} calls, {BO_N_INIT} initial points ...")
    bo_result = gp_minimize(
        bo_objective,
        dimensions,
        n_calls=BO_N_CALLS,
        n_initial_points=BO_N_INIT,
        acq_func='EI',
        random_state=RANDOM_STATE,
        verbose=False,
    )

    bo_optimal    = decode(np.array(bo_result.x, dtype=np.float32),
                           target_std=mean_fad.std())
    bo_best_score = -bo_result.fun
    bo_func_vals  = np.array(bo_result.func_vals)
    bo_best_so_far = -np.minimum.accumulate(bo_func_vals)
    print(f"  BO best P(5xFAD) = {bo_best_score:.4f}")

    # ── BO for WT: minimise P(5xFAD) ─────────────────────────────────────────
    print(f"  Running BO for WT (minimise P(5xFAD)) ...")

    def bo_wt_objective(params):
        return ensemble_score(decode(np.array(params, dtype=np.float32)))

    bo_wt_result = gp_minimize(
        bo_wt_objective,
        dimensions,
        n_calls=BO_N_CALLS,
        n_initial_points=BO_N_INIT,
        acq_func='EI',
        random_state=RANDOM_STATE + 1,
        verbose=False,
    )
    bo_wt_optimal     = decode(np.array(bo_wt_result.x, dtype=np.float32),
                               target_std=mean_wt.std())
    bo_wt_score       = bo_wt_result.fun
    bo_wt_func_vals   = np.array(bo_wt_result.func_vals)
    bo_wt_best_so_far = np.minimum.accumulate(bo_wt_func_vals)
    print(f"  BO best P(5xFAD)={bo_wt_score:.4f}  P(WT)={1-bo_wt_score:.4f}")

else:
    bo_optimal       = grad_optimal.copy()
    bo_best_score    = grad_best_score
    bo_func_vals     = np.array([])
    bo_best_so_far   = np.array([])
    bo_wt_optimal    = grad_wt_optimal.copy()
    bo_wt_score      = grad_wt_score
    bo_wt_func_vals  = np.array([])
    bo_wt_best_so_far = np.array([])


# ══════════════════════════════════════════════════════════════════════════════
# 7. Figures
# ══════════════════════════════════════════════════════════════════════════════

C_WT  = '#2980b9'
C_FAD = '#c0392b'

# ── Figure A: Optimal Signals (2×2) ──────────────────────────────────────────
print("\nFigure A: optimal signals ...")

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.subplots_adjust(hspace=0.45, wspace=0.30)

# (0,0) CNN-preferred WT stimulus
ax = axes[0, 0]
ax.fill_between(t, mean_wt - se_wt, mean_wt + se_wt,
                alpha=0.25, color=C_WT)
ax.plot(t, mean_wt, color=C_WT, lw=1.5, alpha=0.7,
        label=f'Mean WT (n={len(wt_means)}, ±SE)')
ax.plot(t, bo_wt_optimal, color=C_WT, lw=2.5, ls='--',
        label=f'BO-optimal WT  [P(5xFAD)={bo_wt_score:.3f}]')
ax.set_title('A.  CNN-preferred WT stimulus\n'
             '(Bayesian optimisation: minimise P(5xFAD))',
             fontweight='bold', fontsize=10)
ax.set_ylabel('Response (norm.)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.text(0.97, 0.05, f'P(WT) = {1-bo_wt_score:.3f}',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=11, fontweight='bold', color=C_WT,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# (0,1) CNN-preferred 5xFAD stimulus
ax = axes[0, 1]
ax.fill_between(t, mean_fad - se_fad, mean_fad + se_fad,
                alpha=0.25, color=C_FAD)
ax.plot(t, mean_fad, color=C_FAD, lw=1.5, alpha=0.7,
        label=f'Mean 5xFAD (n={len(fad_means)}, ±SE)')
ax.plot(t, bo_optimal, color=C_FAD, lw=2.5, ls='--',
        label=f'BO-optimal 5xFAD  [P(5xFAD)={bo_best_score:.3f}]')
ax.set_title('B.  CNN-preferred 5xFAD stimulus\n'
             '(Bayesian optimisation: maximise P(5xFAD))',
             fontweight='bold', fontsize=10)
ax.set_ylabel('Response (norm.)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.text(0.97, 0.95, f'P(5xFAD) = {bo_best_score:.3f}',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=11, fontweight='bold', color=C_FAD,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# (1,0) Direct comparison
ax = axes[1, 0]
ax.plot(t, bo_wt_optimal, color=C_WT,  lw=2.5,
        label=f'WT-optimal   P(WT)={1-bo_wt_score:.3f}')
ax.plot(t, bo_optimal,    color=C_FAD, lw=2.5,
        label=f'5xFAD-optimal  P(5xFAD)={bo_best_score:.3f}')
ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_title('C.  Optimal stimuli — direct comparison\n'
             '(what the CNN "thinks" each class should look like)',
             fontweight='bold', fontsize=10)
ax.set_xlabel('Sample index (NI Response, 2500 pts @ 250 Hz = 10 s)', fontsize=9)
ax.set_ylabel('Response (norm.)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (1,1) Discriminative features: 5xFAD-opt minus WT-opt
ax = axes[1, 1]
diff = bo_optimal - bo_wt_optimal
ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.fill_between(t, 0, diff, where=diff >= 0,
                color=C_FAD, alpha=0.55, label='5xFAD-driving (positive)')
ax.fill_between(t, 0, diff, where=diff < 0,
                color=C_WT,  alpha=0.55, label='WT-driving (negative)')
ax.plot(t, diff, color='#2c2c2c', lw=1.2, alpha=0.7)
ax.set_title('D.  Discriminative features\n'
             '(5xFAD-optimal minus WT-optimal)',
             fontweight='bold', fontsize=10)
ax.set_xlabel('Sample index (NI Response, 2500 pts @ 250 Hz = 10 s)', fontsize=9)
ax.set_ylabel('Δ Response (5xFAD-opt − WT-opt)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle('Activation Maximization — What drives WT vs 5xFAD NI predictions?\n'
             'Bayesian optimisation in PCA-reduced space (K=15, ±3σ bounds)\n'
             'Input: prototype waveform replicated × 10 repetitions',
             fontweight='bold', fontsize=12)
plt.savefig(os.path.join(FIG_DIR, '12_a_optimal_signals.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 12_a_optimal_signals.png")


# ── Figure B: PCA Coefficients violin plots ───────────────────────────────────
print("Figure B: PCA coefficients ...")

N_PC_SHOW = 8
grad_coeffs_opt    = encode(best_grad['signal_raw'])
grad_wt_coeffs_opt = encode(best_grad_wt['signal_raw'])
bo_coeffs_opt      = encode(bo_optimal)
bo_wt_coeffs_opt   = encode(bo_wt_optimal)

fig, axes = plt.subplots(1, N_PC_SHOW, figsize=(16, 5), sharey=False)
fig.subplots_adjust(wspace=0.3)

for k, ax in enumerate(axes):
    wt_vals  = wt_coeffs[:, k]
    fad_vals = fad_coeffs[:, k]

    vp_wt  = ax.violinplot([wt_vals],  positions=[-0.2], widths=0.35,
                            showmedians=True, showextrema=False)
    vp_fad = ax.violinplot([fad_vals], positions=[0.2],  widths=0.35,
                            showmedians=True, showextrema=False)

    for pc in vp_wt['bodies']:
        pc.set_facecolor(C_WT); pc.set_alpha(0.6)
    for pc in vp_fad['bodies']:
        pc.set_facecolor(C_FAD); pc.set_alpha(0.6)
    vp_wt['cmedians'].set_color(C_WT)
    vp_fad['cmedians'].set_color(C_FAD)

    ax.scatter([0], [bo_wt_coeffs_opt[k]], marker='v', s=80,
               color=C_WT, zorder=5, label='BO WT-opt'  if k == 0 else '')
    ax.scatter([0], [bo_coeffs_opt[k]],    marker='^', s=80,
               color=C_FAD, zorder=5, label='BO FAD-opt' if k == 0 else '')

    ax.set_title(f'PC{k+1}', fontsize=9, fontweight='bold')
    ax.set_xticks([-0.2, 0.2])
    ax.set_xticklabels(['WT', '5xFAD'], fontsize=7, rotation=45)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5, ls='--')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [
    Patch(facecolor=C_WT,  alpha=0.6, label='WT'),
    Patch(facecolor=C_FAD, alpha=0.6, label='5xFAD'),
    Line2D([0], [0], marker='v', color=C_WT,  ls='', ms=8, label='BO WT-optimal'),
    Line2D([0], [0], marker='^', color=C_FAD, ls='', ms=8, label='BO FAD-optimal'),
]
axes[0].legend(handles=handles, fontsize=7, loc='upper right')

plt.suptitle('PCA Coefficients (z-scored space) — First 8 PCs\n'
             'WT (blue) vs 5xFAD (red)  |  ▼ BO WT-optimal  ▲ BO FAD-optimal',
             fontweight='bold', fontsize=11)
plt.savefig(os.path.join(FIG_DIR, '12_b_pca_coefficients.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 12_b_pca_coefficients.png")


# ── Figure C: Convergence ─────────────────────────────────────────────────────
print("Figure C: convergence ...")

colors_grad = ['#27ae60', '#8e44ad', '#e74c3c']

if SKOPT_OK:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
else:
    fig, ax_single = plt.subplots(1, 1, figsize=(7, 5))
    axes = [ax_single]

# Left: gradient ascent
ax = axes[0]
for res, color in zip(grad_results, colors_grad):
    steps = np.arange(10, GRAD_STEPS + 1, 10)
    ax.plot(steps, res['scores_dense'], color=color, lw=1.8,
            label=f"{res['label']}  (final={res['score']:.3f})")
ax.set_xlabel('Gradient step', fontsize=10)
ax.set_ylabel('P(5xFAD)', fontsize=10)
ax.set_title('A.  Gradient Ascent Convergence (5xFAD)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)

# Right: BO convergence
if SKOPT_OK:
    ax = axes[1]
    n_bo_calls = len(bo_func_vals)
    iters = np.arange(1, n_bo_calls + 1)
    ax.scatter(iters, -bo_func_vals, s=8, alpha=0.4, color='gray',
               label='All evaluations')
    ax.plot(iters, bo_best_so_far, '#e67e22', lw=2.0,
            label=f'Best so far (final={bo_best_score:.3f})')
    ax.axvline(BO_N_INIT, color='steelblue', ls='--', lw=1.5,
               label=f'End of random phase (n={BO_N_INIT})')
    ax.set_xlabel('BO iteration', fontsize=10)
    ax.set_ylabel('P(5xFAD)', fontsize=10)
    ax.set_title('B.  Bayesian Optimization Convergence (5xFAD)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

plt.suptitle('Optimization Convergence — Natural Image CNN\n'
             'Gradient ascent (left) vs Bayesian optimization (right)',
             fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '12_c_convergence.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 12_c_convergence.png")


# ── Figure D: PCA landscape (PC1 × PC2 grid) ─────────────────────────────────
print("Figure D: PCA landscape ...")

pc1_range = np.linspace(bounds[0][0], bounds[0][1], GRID_N)
pc2_range = np.linspace(bounds[1][0], bounds[1][1], GRID_N)
pc1_grid, pc2_grid = np.meshgrid(pc1_range, pc2_range)

# Build all grid coefficient vectors [GRID_N*GRID_N, K]
grid_coeffs_base = np.tile(bo_coeffs_opt, (GRID_N * GRID_N, 1)).astype(np.float32)
grid_coeffs_base[:, 0] = pc1_grid.ravel()
grid_coeffs_base[:, 1] = pc2_grid.ravel()

# Decode to signals [GRID_N*GRID_N, 2500]
grid_signals = np.array([decode(c) for c in grid_coeffs_base], dtype=np.float32)

# Replicate each signal 10× → [GRID_N*GRID_N, 10, 2500] for model forward pass
x_grid = (torch.tensor(grid_signals, dtype=torch.float32)  # [G, 2500]
          .unsqueeze(1)                                      # [G, 1, 2500]
          .expand(-1, N_REPS, -1)                           # [G, 10, 2500]
          .contiguous()
          .to(device))

with torch.no_grad():
    grid_scores = np.mean(
        [F.softmax(m(x_grid), dim=1)[:, 1].cpu().numpy() for m in models],
        axis=0
    )  # [GRID_N*GRID_N]
grid_scores_2d = grid_scores.reshape(GRID_N, GRID_N)

wt_pc1,  wt_pc2  = wt_coeffs[:, 0],  wt_coeffs[:, 1]
fad_pc1, fad_pc2 = fad_coeffs[:, 0], fad_coeffs[:, 1]

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(
    grid_scores_2d,
    origin='lower',
    extent=[pc1_range[0], pc1_range[-1], pc2_range[0], pc2_range[-1]],
    aspect='auto',
    cmap='RdBu_r',
    vmin=0, vmax=1,
    interpolation='bilinear',
)
plt.colorbar(im, ax=ax, label='P(5xFAD) — ensemble mean')

ax.scatter(wt_pc1,  wt_pc2,  c=C_WT,  s=25, alpha=0.8, zorder=3,
           edgecolors='white', lw=0.5, label=f'WT (n={len(wt_means)})')
ax.scatter(fad_pc1, fad_pc2, c=C_FAD, s=25, alpha=0.8, zorder=3,
           edgecolors='white', lw=0.5, label=f'5xFAD (n={len(fad_means)})')

ax.scatter([grad_coeffs_opt[0]], [grad_coeffs_opt[1]],
           marker='*', s=300, c='#27ae60', zorder=5, edgecolors='black', lw=0.8,
           label=f'Gradient opt.  P={grad_best_score:.3f}')
ax.scatter([bo_coeffs_opt[0]], [bo_coeffs_opt[1]],
           marker='D', s=120, c='#e67e22', zorder=5, edgecolors='black', lw=0.8,
           label=f'BO opt.  P={bo_best_score:.3f}')

ax.set_xlabel('PC1 coefficient', fontsize=11)
ax.set_ylabel('PC2 coefficient', fontsize=11)
ax.set_title('PCA Score Landscape — PC1 × PC2\n'
             'P(5xFAD) ensemble mean (other PCs fixed at BO-optimal)',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '12_d_pca_landscape.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 12_d_pca_landscape.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Results table
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving results table ...")

rows = []
rows.append({
    'signal': 'mean_WT', 'p_5xFAD': round(score_wt, 4),
    'method': 'reference', 'n_iterations': 0, 'init': '',
    'note': 'Group mean WT signal (channel-avg of 10 reps)',
})
rows.append({
    'signal': 'mean_5xFAD', 'p_5xFAD': round(score_fad, 4),
    'method': 'reference', 'n_iterations': 0, 'init': '',
    'note': 'Group mean 5xFAD signal (channel-avg of 10 reps)',
})
for res in grad_results:
    rows.append({
        'signal':       f'gradient_FAD_{res["label"]}',
        'p_5xFAD':      round(res['score'], 4),
        'method':       'gradient_ascent_FAD',
        'n_iterations': GRAD_STEPS,
        'init':         res['label'],
        'note':         f'lr={GRAD_LR}, TV_w={TV_WEIGHT}, L2_w={L2_WEIGHT}',
    })
for res in grad_results_wt:
    rows.append({
        'signal':       f'gradient_WT_{res["label"]}',
        'p_5xFAD':      round(res['score'], 4),
        'method':       'gradient_descent_WT',
        'n_iterations': GRAD_STEPS,
        'init':         res['label'],
        'note':         f'lr={GRAD_LR}, TV_w={TV_WEIGHT}, L2_w={L2_WEIGHT}',
    })
rows.append({
    'signal':       'bo_FAD_optimal',
    'p_5xFAD':      round(bo_best_score, 4),
    'method':       'bayesian_optimization_FAD' if SKOPT_OK else 'gradient_fallback',
    'n_iterations': BO_N_CALLS if SKOPT_OK else 0,
    'init':         f'K_PCA={K_PCA}, n_init={BO_N_INIT}' if SKOPT_OK else 'skopt_unavailable',
    'note':         'EI, maximise P(5xFAD)' if SKOPT_OK else 'skopt unavailable',
})
rows.append({
    'signal':       'bo_WT_optimal',
    'p_5xFAD':      round(bo_wt_score, 4),
    'method':       'bayesian_optimization_WT' if SKOPT_OK else 'gradient_fallback',
    'n_iterations': BO_N_CALLS if SKOPT_OK else 0,
    'init':         f'K_PCA={K_PCA}, n_init={BO_N_INIT}' if SKOPT_OK else 'skopt_unavailable',
    'note':         'EI, minimise P(5xFAD)' if SKOPT_OK else 'skopt unavailable',
})

df_out  = pd.DataFrame(rows)
out_csv = os.path.join(TAB_DIR, '12_optimization_results.csv')
df_out.to_csv(out_csv, index=False)
print(f"  Saved: {out_csv}")
print(df_out.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Bayesian Input Optimization — Natural Image CNN")
print(f"{'='*60}")
print(f"  P(5xFAD | mean_WT)         = {score_wt:.4f}")
print(f"  P(5xFAD | mean_5xFAD)      = {score_fad:.4f}")
print(f"  P(5xFAD | grad FAD opt)    = {grad_best_score:.4f}  [{best_grad['label']}]")
print(f"  P(5xFAD | grad WT  opt)    = {grad_wt_score:.4f}  [{best_grad_wt['label']}]"
      f"  → P(WT)={1-grad_wt_score:.4f}")
print(f"  P(5xFAD | BO  FAD opt)     = {bo_best_score:.4f}")
print(f"  P(5xFAD | BO  WT  opt)     = {bo_wt_score:.4f}  → P(WT)={1-bo_wt_score:.4f}")
print(f"\nFigures → {FIG_DIR}")
print(f"Tables  → {TAB_DIR}")
print(f"{'='*60}")
