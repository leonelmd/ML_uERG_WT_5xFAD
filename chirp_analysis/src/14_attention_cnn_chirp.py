"""
14_attention_cnn_chirp.py
==========================
Trains AttentionBinaryCNN (temporal attention pooling) on two chirp segments
and compares them:

  • amplitude  (samples 6000–8750, T=2750, T'≈22 attention positions)
  • full       (samples 0–8750,    T=8750, T'≈68 attention positions)

The full segment spans all chirp phases (flash → frequency sweep →
amplitude sweep).  Attention pooling can therefore reveal WHICH phase
the model focuses on — a biological question that TemporalStatPool could
never answer.

Key properties of AttentionBinaryCNN
--------------------------------------
• Attention weights [B, T'] are the interpretable heatmap — no Grad-CAM needed.
• ~4 K parameters (vs ~7 K for ImprovedBinaryCNN) — better suited to N≈46.
• Statistical test on T'=22 or 68 positions has far more power than 2750 tests.

Protocol
--------
• 5-fold stratified, subject-disjoint CV  (same split as scripts 04 & 05).
• Primary metric: pooled cross-validated AUC.
• Attention weights collected from held-out subjects using best-epoch model.

Outputs
-------
  results/figures/14_a_segment_comparison.png   — amplitude vs full AUC/Acc
  results/figures/14_b_attention_maps.png        — group-level attention heatmaps
  results/figures/14_c_attention_stats.png       — statistical test on T' positions
  results/tables/14_segment_comparison.csv
  results/tables/14_attention_records_{seg}.csv  — per-subject attention weights
  results/models/14_attn_{seg}_fold_{k}.pt

Usage (from chirp_analysis/ folder)
-------------------------------------
    python src/14_attention_cnn_chirp.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

from dataset import ERGChirpDataset
from models  import AttentionBinaryCNN

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
os.makedirs(MOD_DIR, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEGMENTS     = ['amplitude', 'frequency', 'flash', 'full']
BATCH_SIZE   = 16
EPOCHS       = 80
LR           = 5e-4
N_FOLDS      = 5
RANDOM_STATE = 42
SMOOTH_SIGMA = 10    # smoothing for attention visualisation

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}\n")

# ── Metadata ─────────────────────────────────────────────────────────────────
df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label = {r.Subject: (1 if '5xFAD' in r.Group else 0)
                    for _, r in df_meta.iterrows()}


# ══════════════════════════════════════════════════════════════════════════════
# Training + evaluation function
# ══════════════════════════════════════════════════════════════════════════════

def run_segment(segment):
    print(f"\n{'='*60}")
    print(f"Segment: {segment.upper()}")

    dataset = ERGChirpDataset(DATA_DIR, META_CSV, segment=segment,
                              cache_dir=CACHE_DIR)

    all_trial_subjs = [s.split('_trial_')[0] for s in dataset.subjects]
    valid_idx       = [i for i, s in enumerate(all_trial_subjs)
                       if s in subject_to_label]
    unique_subjects = sorted(set(all_trial_subjs[i] for i in valid_idx))
    subj_labels     = [subject_to_label[s] for s in unique_subjects]

    # Signal length and expected T' (attention positions after conv backbone)
    sample_sig, _, _ = dataset[valid_idx[0]]
    T = int(sample_sig.shape[-1])
    # Compute T' analytically: stride=2×4×2×4×2 = 128, then Conv3 stride=2 → ÷2 total
    # Exact: apply conv arithmetic
    t1 = (T + 2*7 - 15)//2 + 1      # Conv1 stride=2, k=15, p=7
    t2 = t1 // 4                      # MaxPool(4)
    t3 = (t2 + 2*5 - 11)//2 + 1      # Conv2 stride=2, k=11, p=5
    t4 = t3 // 4                      # MaxPool(4)
    T_prime = (t4 + 2*3 - 7)//2 + 1  # Conv3 stride=2, k=7,  p=3
    print(f"  Input T={T}  →  Attention positions T'={T_prime}")

    torch.manual_seed(RANDOM_STATE);  np.random.seed(RANDOM_STATE)

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
    pooled = {}          # subject → {y_true, y_prob}
    fold_accs, fold_aucs = [], []
    all_records = []     # per-subject held-out attribution records

    for fold, (tr_subj_idx, vl_subj_idx) in enumerate(
            skf.split(unique_subjects, subj_labels), 1):

        tr_subs = {unique_subjects[i] for i in tr_subj_idx}
        vl_subs = {unique_subjects[i] for i in vl_subj_idx}
        tr_idx  = [i for i in valid_idx if all_trial_subjs[i] in tr_subs]
        vl_idx  = [i for i in valid_idx if all_trial_subjs[i] in vl_subs]

        tr_loader = DataLoader(Subset(dataset, tr_idx),
                               batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True)
        vl_loader = DataLoader(Subset(dataset, vl_idx),
                               batch_size=BATCH_SIZE, shuffle=False)

        model     = AttentionBinaryCNN().to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=EPOCHS)

        best_auc, best_acc = 0.0, 0.0
        best_votes, best_state = None, None

        for epoch in range(EPOCHS):
            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            for signals, lbls4, _ in tr_loader:
                bz = torch.tensor([1 if l >= 2 else 0
                                   for l in lbls4]).to(device)
                sig_d = signals.to(device)
                sig_d = sig_d + torch.randn_like(sig_d) * 0.01  # augment
                optimizer.zero_grad()
                criterion(model(sig_d), bz).backward()
                optimizer.step()
            scheduler.step()

            # ── Validate ─────────────────────────────────────────────────────
            model.eval()
            votes = {}
            with torch.no_grad():
                for signals, lbls4, subjs in vl_loader:
                    probs = F.softmax(
                        model(signals.to(device)), dim=1)[:, 1].cpu().tolist()
                    for prob, s in zip(probs, subjs):
                        orig = s.split('_trial_')[0]
                        votes.setdefault(
                            orig, {'label': subject_to_label[orig],
                                   'probs': []})
                        votes[orig]['probs'].append(prob)

            sl       = [v['label'] for v in votes.values()]
            sp_probs = [np.mean(v['probs']) for v in votes.values()]
            sp       = [1 if p >= 0.5 else 0 for p in sp_probs]
            acc = accuracy_score(sl, sp)
            auc = roc_auc_score(sl, sp_probs) if len(np.unique(sl)) > 1 else 0.5

            if (auc > best_auc) or (auc == best_auc and acc > best_acc):
                best_auc, best_acc = auc, acc
                best_votes = {s: {'y_true': v['label'],
                                  'y_prob': float(np.mean(v['probs']))}
                              for s, v in votes.items()}
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}

        # ── Save model ────────────────────────────────────────────────────────
        torch.save(best_state,
                   os.path.join(MOD_DIR, f'14_attn_{segment}_fold_{fold}.pt'))
        fold_accs.append(best_acc)
        fold_aucs.append(best_auc)
        pooled.update(best_votes)
        print(f"  Fold {fold}  Acc={best_acc:.1%}  AUC={best_auc:.3f}")

        # ── Collect attention weights on held-out subjects ────────────────────
        model.load_state_dict(best_state)
        model.eval()

        subj_data = {}
        for idx in vl_idx:
            sig, _, name = dataset[idx]
            base = name.split('_trial_')[0]
            arr  = (sig.float().numpy() if hasattr(sig, 'numpy')
                    else np.array(sig, dtype=np.float32))
            subj_data.setdefault(base, []).append(arr.flatten())

        for subj, sigs in subj_data.items():
            mean_sig = np.mean(sigs, axis=0)          # [T]
            x        = torch.tensor(mean_sig,
                                    dtype=torch.float32)[None, None].to(device)
            with torch.no_grad():
                logits, attn_t = model(x, return_attn=True)
                prob_fad = F.softmax(logits, dim=1)[0, 1].item()
                pred     = logits.argmax(1).item()

            attn_raw = attn_t[0].cpu().float().numpy()   # [T']
            # Upsample T' → T for overlay
            attn_up  = np.interp(
                np.linspace(0, T_prime - 1, T),
                np.arange(T_prime), attn_raw)             # [T]

            all_records.append({
                'subject' : subj,
                'label'   : subject_to_label[subj],
                'pred'    : pred,
                'prob_fad': prob_fad,
                'signal'  : mean_sig,
                'attn_raw': attn_raw,    # [T'] — use for stats
                'attn_up' : attn_up,     # [T]  — use for overlay
            })

    # ── Pooled metrics ────────────────────────────────────────────────────────
    y_true = np.array([pooled[s]['y_true'] for s in unique_subjects
                       if s in pooled])
    y_prob = np.array([pooled[s]['y_prob'] for s in unique_subjects
                       if s in pooled])
    pooled_auc = roc_auc_score(y_true, y_prob)
    pooled_acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    print(f"  → Pooled AUC={pooled_auc:.3f}  "
          f"Fold-mean AUC={np.mean(fold_aucs):.3f}  "
          f"Fold-mean Acc={np.mean(fold_accs):.1%}")

    return {
        'segment'        : segment,
        'T'              : T,
        'T_prime'        : T_prime,
        'pooled_auc'     : pooled_auc,
        'pooled_acc'     : pooled_acc,
        'fold_mean_auc'  : np.mean(fold_aucs),
        'fold_std_auc'   : np.std(fold_aucs, ddof=1),
        'fold_mean_acc'  : np.mean(fold_accs),
        'fold_std_acc'   : np.std(fold_accs, ddof=1),
        'records'        : all_records,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Run both segments
# ══════════════════════════════════════════════════════════════════════════════
results = {}
for seg in SEGMENTS:
    results[seg] = run_segment(seg)

# Save segment comparison table
rows = [{
    'Segment'      : seg,
    'Pooled_AUC'   : results[seg]['pooled_auc'],
    'Pooled_Acc'   : results[seg]['pooled_acc'],
    'Fold_Mean_AUC': results[seg]['fold_mean_auc'],
    'Fold_Std_AUC' : results[seg]['fold_std_auc'],
    'Fold_Mean_Acc': results[seg]['fold_mean_acc'],
    'Fold_Std_Acc' : results[seg]['fold_std_acc'],
    'T_prime'      : results[seg]['T_prime'],
} for seg in SEGMENTS]
df_comp = pd.DataFrame(rows)
df_comp.to_csv(os.path.join(TAB_DIR, '14_segment_comparison.csv'), index=False)

# Save per-subject attention records for each segment
for seg in SEGMENTS:
    recs  = results[seg]['records']
    T_p   = results[seg]['T_prime']
    rows_ = []
    for r in recs:
        row = {'subject': r['subject'], 'label': r['label'],
               'pred': r['pred'], 'prob_fad': r['prob_fad']}
        for k, v in enumerate(r['attn_raw']):
            row[f'attn_pos_{k}'] = float(v)
        rows_.append(row)
    pd.DataFrame(rows_).to_csv(
        os.path.join(TAB_DIR, f'14_attention_records_{seg}.csv'), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Identify best segment
# ══════════════════════════════════════════════════════════════════════════════
best_seg = max(SEGMENTS, key=lambda s: results[s]['pooled_auc'])
print(f"\nBest segment by pooled AUC: {best_seg.upper()} "
      f"(AUC={results[best_seg]['pooled_auc']:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A: Segment Comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure A: Segment comparison ...")

# Load ImprovedBinaryCNN amplitude baseline (from script 04 if available)
ref_auc = None
ref_csv = os.path.join(TAB_DIR, '04_chirp_segment_comparison.csv')
if os.path.exists(ref_csv):
    df04 = pd.read_csv(ref_csv)
    row04 = df04[df04['Segment'] == 'amplitude']
    if not row04.empty:
        ref_auc = float(row04['Pooled_AUC'].values[0])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x_pos = np.arange(len(SEGMENTS))
colors = ['#2980b9', '#e67e22', '#8e44ad', '#27ae60'][:len(SEGMENTS)]
labels_nice = [s.capitalize() for s in SEGMENTS]

# Left: Pooled AUC
ax = axes[0]
bars = ax.bar(x_pos, [results[s]['pooled_auc'] for s in SEGMENTS],
              color=colors, width=0.5, edgecolor='black', alpha=0.85)
ax.scatter(x_pos, [results[s]['fold_mean_auc'] for s in SEGMENTS],
           marker='D', s=60, color='black', zorder=5, label='Fold-mean AUC')
if ref_auc is not None:
    ax.axhline(ref_auc, color='#c0392b', ls='--', lw=1.5,
               label=f'ImprovedCNN amplitude (AUC={ref_auc:.3f})')
ax.axhline(0.5, color='gray', ls=':', lw=1.2, label='Chance')
ax.set_xticks(x_pos); ax.set_xticklabels(labels_nice, fontsize=11)
ax.set_ylim(0, 1.1); ax.set_ylabel('Pooled AUC', fontsize=11)
ax.set_title('Pooled AUC — AttentionBinaryCNN', fontweight='bold')
ax.legend(fontsize=9)
for i, s in enumerate(SEGMENTS):
    ax.text(i, results[s]['pooled_auc'] + 0.02,
            f"{results[s]['pooled_auc']:.3f}",
            ha='center', fontweight='bold', fontsize=11)
best_idx = SEGMENTS.index(best_seg)
ax.get_xticklabels()[best_idx].set_fontweight('bold')
ax.get_xticklabels()[best_idx].set_color('darkblue')

# Right: Fold-mean Accuracy ± SD
ax = axes[1]
ax.bar(x_pos,
       [results[s]['fold_mean_acc'] for s in SEGMENTS],
       yerr=[results[s]['fold_std_acc']  for s in SEGMENTS],
       color=colors, width=0.5, edgecolor='black', alpha=0.85, capsize=8)
ax.axhline(0.5, color='gray', ls=':', lw=1.2, label='Chance')
ax.set_xticks(x_pos); ax.set_xticklabels(labels_nice, fontsize=11)
ax.set_ylim(0, 1.1); ax.set_ylabel('Fold-mean Subject Accuracy', fontsize=11)
ax.set_title('Fold-mean Accuracy — AttentionBinaryCNN', fontweight='bold')
ax.legend(fontsize=9)
for i, s in enumerate(SEGMENTS):
    m = results[s]['fold_mean_acc']; e = results[s]['fold_std_acc']
    ax.text(i, m + e + 0.02, f'{m:.1%}',
            ha='center', fontweight='bold', fontsize=11)

plt.suptitle('Attention CNN — Segment Comparison: '
             + ' vs '.join(s.capitalize() for s in SEGMENTS) + '\n'
             + '(5-fold subject-disjoint CV, WT vs 5xFAD)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '14_a_segment_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 14_a_segment_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figures B & C for BOTH segments (attention maps + statistics)
# ══════════════════════════════════════════════════════════════════════════════

def smean(arr, sigma=SMOOTH_SIGMA):
    return gaussian_filter1d(arr.mean(axis=0), sigma=sigma)

def sse(arr, sigma=SMOOTH_SIGMA):
    return gaussian_filter1d(arr.std(axis=0)/np.sqrt(arr.shape[0]), sigma=sigma)


for seg in SEGMENTS:
    res     = results[seg]
    recs    = res['records']
    T       = res['T']
    T_prime = res['T_prime']
    t       = np.arange(T)
    t_prime = np.arange(T_prime)

    wt_rec  = [r for r in recs if r['label'] == 0]
    fad_rec = [r for r in recs if r['label'] == 1]

    wt_sigs  = np.array([r['signal']   for r in wt_rec])   # [N_wt,  T]
    fad_sigs = np.array([r['signal']   for r in fad_rec])  # [N_fad, T]
    wt_attn  = np.array([r['attn_raw'] for r in wt_rec])   # [N_wt,  T']
    fad_attn = np.array([r['attn_raw'] for r in fad_rec])  # [N_fad, T']
    wt_attn_up  = np.array([r['attn_up'] for r in wt_rec])
    fad_attn_up = np.array([r['attn_up'] for r in fad_rec])

    # ── Statistical test on T' attention positions ─────────────────────────────
    pvals_attn = np.array([
        mannwhitneyu(wt_attn[:, ti], fad_attn[:, ti],
                     alternative='two-sided').pvalue
        for ti in range(T_prime)
    ])
    _, pvals_fdr, _, _ = multipletests(pvals_attn, method='fdr_bh')
    sig_attn = pvals_fdr < 0.05

    n1, n2 = len(wt_rec), len(fad_rec)
    effect_r = np.array([
        abs(mannwhitneyu(wt_attn[:, ti], fad_attn[:, ti]).statistic
            - n1*n2/2) / (n1*n2/2)
        for ti in range(T_prime)
    ])

    sig_pos = np.where(sig_attn)[0]
    print(f"\n{seg.upper()}: {len(sig_pos)} / {T_prime} attention positions "
          f"significant (FDR q<0.05)  →  {sig_pos.tolist()}")

    # Map significant T' positions back to input sample ranges
    stride = T / T_prime
    for pos in sig_pos:
        lo, hi = int(pos * stride), int((pos + 1) * stride)
        diff   = fad_attn[:, pos].mean() - wt_attn[:, pos].mean()
        print(f"    T'[{pos}] = input samples [{lo}:{hi}]  "
              f"ΔAttn(5xFAD-WT)={diff:.4f}  r={effect_r[pos]:.3f}  "
              f"q={pvals_fdr[pos]:.4f}")

    # Save stats
    stat_rows = [{
        'segment'    : seg,
        'attn_pos'   : i,
        'input_start': int(i * stride),
        'input_end'  : int((i+1) * stride),
        'WT_mean'    : wt_attn[:, i].mean(),
        'FAD_mean'   : fad_attn[:, i].mean(),
        'p_raw'      : pvals_attn[i],
        'p_BH'       : pvals_fdr[i],
        'effect_r'   : effect_r[i],
        'significant': bool(sig_attn[i]),
    } for i in range(T_prime)]
    pd.DataFrame(stat_rows).to_csv(
        os.path.join(TAB_DIR, f'14_attention_stats_{seg}.csv'), index=False)

    # Best representatives
    correct_wt  = [r for r in wt_rec  if r['pred'] == 0] or wt_rec
    correct_fad = [r for r in fad_rec if r['pred'] == 1] or fad_rec
    best_wt  = min(correct_wt,  key=lambda r: r['prob_fad'])
    best_fad = max(correct_fad, key=lambda r: r['prob_fad'])

    # ── FIGURE B: Attention maps ───────────────────────────────────────────────
    print(f"  Figure B ({seg}): attention maps ...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    # Row 0: mean signals
    for ax, sigs, color, lbl, n in [
            (axes[0,0], wt_sigs,  '#2980b9', 'WT',    len(wt_rec)),
            (axes[0,1], fad_sigs, '#c0392b', '5xFAD', len(fad_rec))]:
        m_, s_ = smean(sigs), sse(sigs)
        ax.fill_between(t, m_-s_, m_+s_, alpha=0.2, color=color)
        ax.plot(t, m_, color=color, lw=1.5, label=f'{lbl} (n={n})')
        ax.set_ylabel('Amplitude (norm.)', fontsize=9)
        ax.set_title(f'Mean ERG — {lbl}', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Row 1: mean attention weights at T' positions
    ax = axes[1, 0]
    m_wt_a  = wt_attn.mean(axis=0)
    m_fad_a = fad_attn.mean(axis=0)
    e_wt_a  = wt_attn.std(axis=0)  / np.sqrt(len(wt_rec))
    e_fad_a = fad_attn.std(axis=0) / np.sqrt(len(fad_rec))
    ax.fill_between(t_prime, m_wt_a - e_wt_a, m_wt_a + e_wt_a,
                    alpha=0.2, color='#2980b9')
    ax.fill_between(t_prime, m_fad_a - e_fad_a, m_fad_a + e_fad_a,
                    alpha=0.2, color='#c0392b')
    ax.plot(t_prime, m_wt_a,  '#2980b9', lw=2, label=f'WT (n={len(wt_rec)})')
    ax.plot(t_prime, m_fad_a, '#c0392b', lw=2, label=f'5xFAD (n={len(fad_rec)})')
    ax.axhline(1/T_prime, color='gray', ls='--', lw=1,
               label=f'Uniform (1/{T_prime})')
    for pos in sig_pos:
        ax.axvspan(pos-0.5, pos+0.5, color='gold', alpha=0.5, zorder=0,
                   label='FDR q<0.05' if pos == sig_pos[0] else '')
    ax.set_xlabel(f'Attention position (T\'={T_prime})', fontsize=9)
    ax.set_ylabel('Attention weight', fontsize=9)
    ax.set_title('Mean Attention Weights per Group', fontweight='bold')
    handles, labels_ = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels_):
        if l not in seen: seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=7)
    ax.grid(alpha=0.3)

    # Panel (1,1): difference + significance
    ax = axes[1, 1]
    diff_attn = m_fad_a - m_wt_a
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.plot(t_prime, diff_attn, color='#8e44ad', lw=2,
            label='ΔAttn = 5xFAD − WT')
    ax.fill_between(t_prime, 0, diff_attn, where=diff_attn > 0,
                    color='#c0392b', alpha=0.25)
    ax.fill_between(t_prime, 0, diff_attn, where=diff_attn < 0,
                    color='#2980b9', alpha=0.25)
    ax.bar(t_prime, -np.log10(np.clip(pvals_fdr, 1e-10, 1)) * 0.001,
           color=['gold' if s else 'lightgray' for s in sig_attn],
           alpha=0.8, label='-log₁₀(q) [scaled]')
    for pos in sig_pos:
        ax.axvline(pos, color='gold', lw=2, alpha=0.7)
    ax.set_xlabel(f'Attention position (T\'={T_prime})', fontsize=9)
    ax.set_ylabel('ΔAttention  (5xFAD − WT)', fontsize=9)
    ax.set_title('Attention Difference + Significance', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Row 2: overlay on representative traces (upsampled attention as heatmap)
    for ax, rec, lbl in [
            (axes[2,0], best_wt,
             f"Best WT  (p_5xFAD={best_wt['prob_fad']:.2f})"),
            (axes[2,1], best_fad,
             f"Best 5xFAD  (p_5xFAD={best_fad['prob_fad']:.2f})")]:
        sig_trace = rec['signal']
        attn_up   = rec['attn_up']
        attn_up   = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
        y_lo = sig_trace.min() - 0.15*(sig_trace.max()-sig_trace.min())
        y_hi = sig_trace.max() + 0.15*(sig_trace.max()-sig_trace.min())
        cam_2d = np.tile(attn_up[np.newaxis, :], (30, 1))
        ax.imshow(cam_2d, aspect='auto',
                  extent=[t[0], t[-1], y_lo, y_hi],
                  cmap='YlOrRd', alpha=0.6, vmin=0, vmax=1,
                  origin='lower', interpolation='bilinear')
        c = '#2980b9' if rec['label'] == 0 else '#c0392b'
        ax.plot(t, sig_trace, color=c, lw=1.5, zorder=5, label=lbl)
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel('Sample index', fontsize=9)
        ax.set_ylabel('Amplitude (norm.)', fontsize=9)
        ax.set_title(f'Attention Overlay — {lbl}', fontweight='bold', fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # For full segment: annotate chirp phase boundaries on signal axes
    if seg == 'full':
        phase_boundaries = {'Freq. start': 1875, 'Amp. start': 6000}
        for ax_row in [axes[0], axes[2]]:   # signal overlay rows only
            for ax_ in ax_row:
                for name_, sample_ in phase_boundaries.items():
                    ax_.axvline(sample_, color='k', ls=':', lw=1,
                                alpha=0.5, label=name_)

    plt.suptitle(f'Attention-Pooling CNN — {seg.capitalize()} Segment\n'
                 f'Temporal Attention Weights  (T\'={T_prime} positions, '
                 f'{len(sig_pos)} significant after FDR correction)',
                 fontweight='bold', fontsize=11)
    plt.savefig(os.path.join(FIG_DIR, f'14_b_attention_maps_{seg}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 14_b_attention_maps_{seg}.png")

    # ── FIGURE C: Statistical analysis at T' positions ────────────────────────
    print(f"  Figure C ({seg}): attention statistics ...")

    fig, axes_c = plt.subplots(2, 1, figsize=(13, 9))
    fig.subplots_adjust(hspace=0.45)

    # Panel 1: -log10(FDR q) bar chart
    ax = axes_c[0]
    bar_colors = ['#c0392b' if fad_attn[:,i].mean() > wt_attn[:,i].mean()
                  else '#2980b9' for i in range(T_prime)]
    bars = ax.bar(t_prime,
                  -np.log10(np.clip(pvals_fdr, 1e-10, 1)),
                  color=bar_colors, alpha=0.75, edgecolor='black',
                  linewidth=0.5)
    thresh = -np.log10(0.05)
    ax.axhline(thresh, color='gold', ls='--', lw=2,
               label=f'FDR q=0.05  (−log₁₀={thresh:.2f})')
    ax.set_xlabel(f'Attention position index (0 to {T_prime-1})', fontsize=10)
    ax.set_ylabel('−log₁₀(FDR q)', fontsize=10)
    ax.set_title('A.  Statistical Significance of Attention Weights  '
                 '(red = 5xFAD > WT, blue = WT > 5xFAD)',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    # Annotate significant positions
    for pos in sig_pos:
        ax.text(pos, -np.log10(pvals_fdr[pos]) + 0.05,
                f'{pos}', ha='center', fontsize=7, fontweight='bold',
                color='darkgoldenrod')

    # Panel 2: Violin plots for top-6 by raw p-value
    ax = axes_c[1]
    top6 = np.argsort(pvals_attn)[:min(6, T_prime)]
    positions_v = np.arange(len(top6))
    vp_wt  = ax.violinplot([wt_attn[:, i]  for i in top6],
                            positions=positions_v - 0.2, widths=0.35,
                            showmedians=True)
    vp_fad = ax.violinplot([fad_attn[:, i] for i in top6],
                            positions=positions_v + 0.2, widths=0.35,
                            showmedians=True)
    for pc in vp_wt['bodies']:
        pc.set_facecolor('#2980b9'); pc.set_alpha(0.6)
    for pc in vp_fad['bodies']:
        pc.set_facecolor('#c0392b'); pc.set_alpha(0.6)
    for part in ['cbars','cmins','cmaxes','cmedians']:
        vp_wt[part].set_color('#2980b9')
        vp_fad[part].set_color('#c0392b')

    ax.set_xticks(positions_v)
    stride_ = T / T_prime
    ax.set_xticklabels(
        [f"T'[{i}]\n(~samp {int(i*stride_)}–{int((i+1)*stride_)})"
         for i in top6], fontsize=8)
    ax.set_ylabel('Attention weight', fontsize=10)
    ax.set_title(f'B.  Top-{len(top6)} Attention Positions by Raw p-Value  '
                 f'(blue=WT, red=5xFAD)', fontweight='bold')
    for v_pos, i in zip(positions_v, top6):
        p = pvals_fdr[i]
        y_top = max(wt_attn[:,i].max(), fad_attn[:,i].max())
        label = (f'q={p:.3f}' if p >= 0.001 else 'q<0.001') \
                if p < 0.1 else f'q={p:.2f}'
        color = 'darkgreen' if p < 0.05 else ('darkorange' if p < 0.1 else 'gray')
        ax.text(v_pos, y_top * 1.08, label,
                ha='center', fontsize=7, fontweight='bold', color=color)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#2980b9', alpha=0.6, label='WT'),
                       Patch(color='#c0392b', alpha=0.6, label='5xFAD')],
              fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(1/T_prime, color='gray', ls='--', lw=1,
               label=f'Uniform (1/{T_prime})')

    plt.suptitle(f'Attention Weight Statistics — {seg.capitalize()} Segment\n'
                 f'Mann-Whitney U + BH FDR  |  T\'={T_prime} positions  |  '
                 f'N_WT={len(wt_rec)}, N_5xFAD={len(fad_rec)}',
                 fontweight='bold', fontsize=11)
    plt.savefig(os.path.join(FIG_DIR, f'14_c_attention_stats_{seg}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 14_c_attention_stats_{seg}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ATTENTION CNN — SUMMARY")
print(df_comp.to_string(index=False))
print(f"\n✓ Figures → {FIG_DIR}")
print(f"✓ Tables  → {TAB_DIR}")
print(f"✓ Models  → {MOD_DIR}")
