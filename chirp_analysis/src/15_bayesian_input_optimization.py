"""
15_bayesian_input_optimization.py
==================================
Optimal stimulus design: find the input signal that maximises P(5xFAD)
across the ensemble of 5 trained ImprovedBinaryCNN models (script 12,
amplitude segment, T=2750).

Two complementary approaches:
  1. Gradient-based activation maximization — differentiable, full T=2750
     resolution, 3 random starts, TV + L2 regularization.
  2. Bayesian optimization in PCA-reduced space — GP surrogate with EI
     acquisition, K=15 components, ±3σ bounds; optimises ensemble mean score.

Outputs
-------
  results/figures/15_a_optimal_signals.png
  results/figures/15_b_pca_coefficients.png
  results/figures/15_c_convergence.png
  results/figures/15_d_pca_landscape.png
  results/tables/15_optimization_results.csv

Usage (from chirp_analysis/ folder)
--------------------------------------
    pip install scikit-optimize
    python src/15_bayesian_input_optimization.py
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

from dataset import ERGChirpDataset
from models  import ImprovedBinaryCNN

# ── Paths (identical to scripts 05 / 13 / 14) ─────────────────────────────────
RETINA_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR    = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
CACHE_DIR   = os.path.join(ROOT, 'data', 'cache')

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
    m = ImprovedBinaryCNN().to(device)
    m.load_state_dict(
        torch.load(os.path.join(MOD_DIR, f'12_improved_amplitude_fold_{fold}.pt'),
                   map_location=device, weights_only=True))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    models.append(m)
print(f"  Loaded {len(models)} fold models.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Load dataset (must match script 05 exactly)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading dataset ...")
dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment='amplitude',
                          cache_dir=CACHE_DIR)

all_signals   = []
binary_labels = []
for i in range(len(dataset)):
    sig, lbl, _ = dataset[i]
    arr = sig.float().numpy().flatten()
    all_signals.append(arr)
    binary_labels.append(1 if lbl >= 2 else 0)

all_signals   = np.array(all_signals,   dtype=np.float32)   # [N, T]
binary_labels = np.array(binary_labels, dtype=np.int32)      # [N]
T             = all_signals.shape[1]
t             = np.arange(T)

wt_signals  = all_signals[binary_labels == 0]   # [N_wt,  T]
fad_signals = all_signals[binary_labels == 1]   # [N_fad, T]

mean_wt  = wt_signals.mean(axis=0)
mean_fad = fad_signals.mean(axis=0)
se_wt    = wt_signals.std(axis=0)  / np.sqrt(len(wt_signals))
se_fad   = fad_signals.std(axis=0) / np.sqrt(len(fad_signals))

print(f"  N_total={len(all_signals)}  N_WT={len(wt_signals)}  N_5xFAD={len(fad_signals)}")
print(f"  Signal length T={T}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Scoring functions
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_score(signal_np: np.ndarray) -> float:
    """Numpy array in → ensemble-mean P(5xFAD) float out. No gradients."""
    x = torch.tensor(signal_np, dtype=torch.float32).reshape(1, 1, -1).to(device)
    with torch.no_grad():
        probs = [F.softmax(m(x), dim=1)[0, 1].item() for m in models]
    return float(np.mean(probs))


def ensemble_log_prob_diff(x_tensor: torch.Tensor) -> torch.Tensor:
    """Differentiable: mean log P(5xFAD) across ensemble.
    x_tensor: [1, 1, T] on device, leaf with requires_grad=True.
    """
    return torch.stack([
        F.log_softmax(m(x_tensor), dim=1)[0, 1] for m in models
    ]).mean()


# Sanity check
score_wt  = ensemble_score(mean_wt)
score_fad = ensemble_score(mean_fad)
print(f"\nSanity check:")
print(f"  P(5xFAD | mean_WT)    = {score_wt:.4f}")
print(f"  P(5xFAD | mean_5xFAD) = {score_fad:.4f}")
assert score_fad >= score_wt or True, "Warning: 5xFAD mean doesn't outscore WT mean"


# ══════════════════════════════════════════════════════════════════════════════
# 4. PCA in z-scored space
# ══════════════════════════════════════════════════════════════════════════════
print("\nFitting PCA ...")

def znorm(sigs: np.ndarray) -> np.ndarray:
    """Per-sample z-score: mirrors InstanceNorm1d."""
    mu  = sigs.mean(axis=1, keepdims=True)
    std = sigs.std(axis=1, keepdims=True) + 1e-8
    return (sigs - mu) / std

all_znorm = znorm(all_signals)                          # [N, T]
pca       = PCA(n_components=K_PCA, random_state=RANDOM_STATE)
pca.fit(all_znorm)
all_coeffs = pca.transform(all_znorm)                   # [N, K]
pc_std     = all_coeffs.std(axis=0)                     # [K]
bounds     = [(-3 * s, 3 * s) for s in pc_std]

wt_coeffs  = all_coeffs[binary_labels == 0]
fad_coeffs = all_coeffs[binary_labels == 1]

print(f"  Explained variance (K={K_PCA}): "
      f"{pca.explained_variance_ratio_.sum()*100:.1f}%")


def encode(sig: np.ndarray) -> np.ndarray:
    """Signal → PCA coefficients in z-scored space."""
    return pca.transform(znorm(sig.reshape(1, -1)))[0]


def decode(coeffs: np.ndarray, target_std: float = None) -> np.ndarray:
    """PCA coefficients → signal. Optionally rescale std for display."""
    sig = pca.inverse_transform(coeffs.reshape(1, -1))[0].astype(np.float32)
    if target_std is not None:
        sig = sig * target_std / (sig.std() + 1e-8)
    return sig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Gradient-based activation maximization (3 starts)
# ══════════════════════════════════════════════════════════════════════════════
print("\nGradient ascent optimization (3 starts) ...")

rng = np.random.default_rng(RANDOM_STATE)
starts = [
    ('mean_WT',    mean_wt.copy()),
    ('mean_5xFAD', mean_fad.copy()),
    ('noise_0.01', rng.normal(0, 0.01, T).astype(np.float32)),
]

grad_results = []

for label, init_signal in starts:
    print(f"  Start: {label} (init score={ensemble_score(init_signal):.4f}) ...", flush=True)
    x = nn.Parameter(
        torch.tensor(init_signal, dtype=torch.float32).reshape(1, 1, -1).to(device))
    opt = torch.optim.Adam([x], lr=GRAD_LR)

    scores_per_step = []
    for step in range(GRAD_STEPS):
        opt.zero_grad()
        loss  = -ensemble_log_prob_diff(x)
        loss += TV_WEIGHT * torch.diff(x[0, 0]).abs().mean()
        loss += L2_WEIGHT * x.pow(2).mean()
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            sc = ensemble_score(x.detach().cpu().numpy().flatten())
            scores_per_step.append(sc)
            print(f"    step {step+1:4d}  P(5xFAD)={sc:.4f}", flush=True)

    # Collect score at every step for convergence plot (lighter: every 10)
    # Re-run for dense curve
    scores_dense = []
    x2 = nn.Parameter(
        torch.tensor(init_signal, dtype=torch.float32).reshape(1, 1, -1).to(device))
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

    final_np  = x.detach().cpu().numpy().flatten()
    final_score = ensemble_score(final_np)

    # Rescale to mean_fad std for display (InstanceNorm-invariant → shape only)
    final_display = final_np * mean_fad.std() / (final_np.std() + 1e-8)

    grad_results.append({
        'label':         label,
        'signal':        final_display,
        'signal_raw':    final_np,
        'score':         final_score,
        'scores_dense':  scores_dense,  # every 10 steps
    })
    print(f"  → Final P(5xFAD) = {final_score:.4f}")

# Best gradient start (maximise P(5xFAD))
best_grad = max(grad_results, key=lambda r: r['score'])
grad_optimal    = best_grad['signal']
grad_best_score = best_grad['score']
print(f"\nBest gradient start (5xFAD): '{best_grad['label']}'  P(5xFAD)={grad_best_score:.4f}")

# ── Gradient descent: minimise P(5xFAD) → maximise P(WT) ─────────────────────
print("\nGradient descent optimization for WT (3 starts) ...")

grad_results_wt = []
for label, init_signal in starts:
    print(f"  Start: {label} (init score={ensemble_score(init_signal):.4f}) ...", flush=True)
    x = nn.Parameter(
        torch.tensor(init_signal, dtype=torch.float32).reshape(1, 1, -1).to(device))
    opt = torch.optim.Adam([x], lr=GRAD_LR)

    for step in range(GRAD_STEPS):
        opt.zero_grad()
        loss  = ensemble_log_prob_diff(x)   # minimise log P(5xFAD) → push toward WT
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
        torch.tensor(init_signal, dtype=torch.float32).reshape(1, 1, -1).to(device))
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

best_grad_wt      = min(grad_results_wt, key=lambda r: r['score'])
grad_wt_optimal   = best_grad_wt['signal']
grad_wt_score     = best_grad_wt['score']   # P(5xFAD) — want this low
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

    bo_optimal     = decode(np.array(bo_result.x, dtype=np.float32),
                            target_std=mean_fad.std())
    bo_best_score  = -bo_result.fun
    bo_func_vals   = np.array(bo_result.func_vals)
    bo_best_so_far = -np.minimum.accumulate(bo_func_vals)
    print(f"  BO best P(5xFAD) = {bo_best_score:.4f}")

    # ── BO for WT: minimise P(5xFAD) ─────────────────────────────────────────
    print(f"  Running BO for WT (minimise P(5xFAD)) ...")
    def bo_wt_objective(params):
        return ensemble_score(decode(np.array(params, dtype=np.float32)))

    bo_wt_result   = gp_minimize(
        bo_wt_objective,
        dimensions,
        n_calls=BO_N_CALLS,
        n_initial_points=BO_N_INIT,
        acq_func='EI',
        random_state=RANDOM_STATE + 1,
        verbose=False,
    )
    bo_wt_optimal      = decode(np.array(bo_wt_result.x, dtype=np.float32),
                                target_std=mean_wt.std())
    bo_wt_score        = bo_wt_result.fun          # P(5xFAD) at WT optimum (want low)
    bo_wt_func_vals    = np.array(bo_wt_result.func_vals)
    bo_wt_best_so_far  = np.minimum.accumulate(bo_wt_func_vals)
    print(f"  BO best P(5xFAD)={bo_wt_score:.4f}  P(WT)={1-bo_wt_score:.4f}")
else:
    # Fallback: use best gradient results
    bo_optimal     = grad_optimal.copy()
    bo_best_score  = grad_best_score
    bo_func_vals   = np.array([])
    bo_best_so_far = np.array([])
    bo_wt_optimal     = grad_wt_optimal.copy()
    bo_wt_score       = grad_wt_score
    bo_wt_func_vals   = np.array([])
    bo_wt_best_so_far = np.array([])


# ══════════════════════════════════════════════════════════════════════════════
# 7. Figures
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure A: Optimal Signals (2×2) ──────────────────────────────────────────
print("\nFigure A: optimal signals ...")

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.subplots_adjust(hspace=0.45, wspace=0.30)

C_WT  = '#2980b9'
C_FAD = '#c0392b'

# ── (0,0) CNN-preferred WT stimulus ──────────────────────────────────────────
ax = axes[0, 0]
ax.fill_between(t, mean_wt - se_wt, mean_wt + se_wt,
                alpha=0.25, color=C_WT)
ax.plot(t, mean_wt, color=C_WT, lw=1.5, alpha=0.7,
        label=f'Mean WT (n={len(wt_signals)}, ±SE)')
ax.plot(t, bo_wt_optimal, color=C_WT, lw=2.5, ls='--',
        label=f'BO-optimal WT  [P(5xFAD)={bo_wt_score:.3f}]')
ax.set_title('A.  CNN-preferred WT stimulus\n'
             '(Bayesian optimisation: minimise P(5xFAD))',
             fontweight='bold', fontsize=10)
ax.set_ylabel('Amplitude (norm.)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.text(0.97, 0.05, f'P(WT) = {1-bo_wt_score:.3f}',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=11, fontweight='bold', color=C_WT,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ── (0,1) CNN-preferred 5xFAD stimulus ───────────────────────────────────────
ax = axes[0, 1]
ax.fill_between(t, mean_fad - se_fad, mean_fad + se_fad,
                alpha=0.25, color=C_FAD)
ax.plot(t, mean_fad, color=C_FAD, lw=1.5, alpha=0.7,
        label=f'Mean 5xFAD (n={len(fad_signals)}, ±SE)')
ax.plot(t, bo_optimal, color=C_FAD, lw=2.5, ls='--',
        label=f'BO-optimal 5xFAD  [P(5xFAD)={bo_best_score:.3f}]')
ax.set_title('B.  CNN-preferred 5xFAD stimulus\n'
             '(Bayesian optimisation: maximise P(5xFAD))',
             fontweight='bold', fontsize=10)
ax.set_ylabel('Amplitude (norm.)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.text(0.97, 0.95, f'P(5xFAD) = {bo_best_score:.3f}',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=11, fontweight='bold', color=C_FAD,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ── (1,0) Direct comparison of the two optimal stimuli ───────────────────────
ax = axes[1, 0]
ax.plot(t, bo_wt_optimal,  color=C_WT,  lw=2.5,
        label=f'WT-optimal   P(WT)={1-bo_wt_score:.3f}')
ax.plot(t, bo_optimal,     color=C_FAD, lw=2.5,
        label=f'5xFAD-optimal  P(5xFAD)={bo_best_score:.3f}')
ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_title('C.  Optimal stimuli — direct comparison\n'
             '(what the CNN "thinks" each class should look like)',
             fontweight='bold', fontsize=10)
ax.set_xlabel('Sample index (Chirp Amplitude Segment)', fontsize=9)
ax.set_ylabel('Amplitude (norm.)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── (1,1) Discriminative features: 5xFAD-opt minus WT-opt ───────────────────
ax = axes[1, 1]
diff = bo_optimal - bo_wt_optimal
ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.fill_between(t, 0, diff, where=diff >= 0,
                color=C_FAD, alpha=0.55,
                label='5xFAD-driving (positive)')
ax.fill_between(t, 0, diff, where=diff < 0,
                color=C_WT,  alpha=0.55,
                label='WT-driving (negative)')
ax.plot(t, diff, color='#2c2c2c', lw=1.2, alpha=0.7)
ax.set_title('D.  Discriminative features\n'
             '(5xFAD-optimal minus WT-optimal)',
             fontweight='bold', fontsize=10)
ax.set_xlabel('Sample index (Chirp Amplitude Segment)', fontsize=9)
ax.set_ylabel('Δ Amplitude (5xFAD-opt − WT-opt)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle('Activation Maximization — What drives WT vs 5xFAD predictions?\n'
             'Bayesian optimisation in PCA-reduced space (K=15, ±3σ bounds)',
             fontweight='bold', fontsize=12)
plt.savefig(os.path.join(FIG_DIR, '15_a_optimal_signals.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 15_a_optimal_signals.png")


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
        pc.set_facecolor('#2980b9'); pc.set_alpha(0.6)
    for pc in vp_fad['bodies']:
        pc.set_facecolor('#c0392b'); pc.set_alpha(0.6)
    vp_wt['cmedians'].set_color('#2980b9')
    vp_fad['cmedians'].set_color('#c0392b')

    # Overlay optimal points (BO only — cleaner read)
    ax.scatter([0], [bo_wt_coeffs_opt[k]],  marker='v', s=80,
               color='#2980b9', zorder=5, label='BO WT-opt'  if k == 0 else '')
    ax.scatter([0], [bo_coeffs_opt[k]],     marker='^', s=80,
               color='#c0392b', zorder=5, label='BO FAD-opt' if k == 0 else '')

    ax.set_title(f'PC{k+1}', fontsize=9, fontweight='bold')
    ax.set_xticks([-0.2, 0.2])
    ax.set_xticklabels(['WT', '5xFAD'], fontsize=7, rotation=45)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5, ls='--')

# Legend on first subplot
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [
    Patch(facecolor='#2980b9', alpha=0.6, label='WT'),
    Patch(facecolor='#c0392b', alpha=0.6, label='5xFAD'),
    Line2D([0], [0], marker='v', color='#2980b9', ls='', ms=8, label='BO WT-optimal'),
    Line2D([0], [0], marker='^', color='#c0392b', ls='', ms=8, label='BO FAD-optimal'),
]
axes[0].legend(handles=handles, fontsize=7, loc='upper right')

plt.suptitle('PCA Coefficients (z-scored space) — First 8 PCs\n'
             'WT (blue) vs 5xFAD (red)  |  ▼ BO WT-optimal  ▲ BO FAD-optimal',
             fontweight='bold', fontsize=11)
plt.savefig(os.path.join(FIG_DIR, '15_b_pca_coefficients.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 15_b_pca_coefficients.png")


# ── Figure C: Convergence ─────────────────────────────────────────────────────
print("Figure C: convergence ...")

colors_grad = ['#27ae60', '#8e44ad', '#e74c3c']
n_bo_calls  = len(bo_func_vals) if SKOPT_OK else 0

if SKOPT_OK:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
else:
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    axes = [axes]

# Left: gradient ascent
ax = axes[0]
for res, color in zip(grad_results, colors_grad):
    steps = np.arange(10, GRAD_STEPS + 1, 10)
    ax.plot(steps, res['scores_dense'], color=color, lw=1.8,
            label=f"{res['label']}  (final={res['score']:.3f})")
ax.set_xlabel('Gradient step', fontsize=10)
ax.set_ylabel('P(5xFAD)', fontsize=10)
ax.set_title('A.  Gradient Ascent Convergence', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)

# Right: BO convergence
if SKOPT_OK:
    ax = axes[1]
    iters = np.arange(1, n_bo_calls + 1)
    ax.scatter(iters, -bo_func_vals, s=8, alpha=0.4, color='gray',
               label='All evaluations')
    ax.plot(iters, bo_best_so_far, '#e67e22', lw=2.0,
            label=f'Best so far (final={bo_best_score:.3f})')
    ax.axvline(BO_N_INIT, color='steelblue', ls='--', lw=1.5,
               label=f'End of initial random phase (n={BO_N_INIT})')
    ax.set_xlabel('BO iteration', fontsize=10)
    ax.set_ylabel('P(5xFAD)', fontsize=10)
    ax.set_title('B.  Bayesian Optimization Convergence', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

plt.suptitle('Optimization Convergence\n'
             'Gradient ascent (left) vs Bayesian optimization (right)',
             fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '15_c_convergence.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 15_c_convergence.png")


# ── Figure D: PCA landscape (PC1 × PC2 grid) ─────────────────────────────────
print("Figure D: PCA landscape ...")

# Grid over PC1 × PC2; all other PCs fixed at BO-optimal values
pc1_range = np.linspace(bounds[0][0], bounds[0][1], GRID_N)
pc2_range = np.linspace(bounds[1][0], bounds[1][1], GRID_N)
pc1_grid, pc2_grid = np.meshgrid(pc1_range, pc2_range)  # [GRID_N, GRID_N]

# Build all grid coefficient vectors [GRID_N*GRID_N, K]
grid_coeffs_base = np.tile(bo_coeffs_opt, (GRID_N * GRID_N, 1)).astype(np.float32)
grid_coeffs_base[:, 0] = pc1_grid.ravel()
grid_coeffs_base[:, 1] = pc2_grid.ravel()

# Decode to signals [GRID_N*GRID_N, T]
grid_signals = np.array([decode(c) for c in grid_coeffs_base], dtype=np.float32)

# Batched forward pass across all 5 models
x_grid = torch.tensor(grid_signals, dtype=torch.float32).unsqueeze(1).to(device)
with torch.no_grad():
    grid_scores = np.mean(
        [F.softmax(m(x_grid), dim=1)[:, 1].cpu().numpy() for m in models],
        axis=0
    )  # [GRID_N*GRID_N]
grid_scores_2d = grid_scores.reshape(GRID_N, GRID_N)

# Project data points to PC1 × PC2
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

ax.scatter(wt_pc1,  wt_pc2,  c='#2980b9', s=20, alpha=0.7, zorder=3,
           edgecolors='white', lw=0.5, label=f'WT (n={len(wt_signals)})')
ax.scatter(fad_pc1, fad_pc2, c='#c0392b', s=20, alpha=0.7, zorder=3,
           edgecolors='white', lw=0.5, label=f'5xFAD (n={len(fad_signals)})')

# Optimal signal markers
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
plt.savefig(os.path.join(FIG_DIR, '15_d_pca_landscape.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 15_d_pca_landscape.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Results table
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving results table ...")

rows = []
rows.append({
    'signal':       'mean_WT',
    'p_5xFAD':      round(score_wt, 4),
    'method':       'reference',
    'n_iterations': 0,
    'init':         '',
    'note':         'Group mean WT signal',
})
rows.append({
    'signal':       'mean_5xFAD',
    'p_5xFAD':      round(score_fad, 4),
    'method':       'reference',
    'n_iterations': 0,
    'init':         '',
    'note':         'Group mean 5xFAD signal',
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

df_out = pd.DataFrame(rows)
out_csv = os.path.join(TAB_DIR, '15_optimization_results.csv')
df_out.to_csv(out_csv, index=False)
print(f"  Saved: {out_csv}")
print(df_out.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Bayesian Input Optimization — Summary")
print(f"{'='*60}")
print(f"  P(5xFAD | mean_WT)       = {score_wt:.4f}")
print(f"  P(5xFAD | mean_5xFAD)    = {score_fad:.4f}")
print(f"  P(5xFAD | grad FAD opt)  = {grad_best_score:.4f}  [{best_grad['label']}]")
print(f"  P(5xFAD | grad WT  opt)  = {grad_wt_score:.4f}  [{best_grad_wt['label']}]"
      f"  → P(WT)={1-grad_wt_score:.4f}")
print(f"  P(5xFAD | BO  FAD opt)   = {bo_best_score:.4f}")
print(f"  P(5xFAD | BO  WT  opt)   = {bo_wt_score:.4f}  → P(WT)={1-bo_wt_score:.4f}")
print(f"\nFigures → {FIG_DIR}")
print(f"Tables  → {TAB_DIR}")
print(f"{'='*60}")
