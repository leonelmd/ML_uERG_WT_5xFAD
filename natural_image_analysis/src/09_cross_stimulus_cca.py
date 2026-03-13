"""
09_cross_stimulus_cca.py
========================
Cross-Stimulus Shared Deficit Mining via Canonical Correlation Analysis (CCA)
(Strategy 4).

Extracts the 96-dimensional bottleneck latent features (TemporalStatPool) from
both the best Chirp CNN (amplitude segment) and the best Natural Image CNN
for the exact same subjects.

We run Canonical Correlation Analysis (CCA) to find the projection where
Chirp latents and Natural Image latents perfectly correlate. This shared latent
dimension represents the stimulus-invariant genotypic deficit.

Outputs
-------
  results/figures/09_cross_stimulus_cca.png
  results/tables/09_cca_components.csv
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cross_decomposition import CCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NI_ROOT  = os.path.dirname(THIS_DIR)
ML_ROOT  = os.path.dirname(NI_ROOT)
sys.path.insert(0, ML_ROOT)

from natural_image_analysis.src.models import ImprovedNICNN_NoAge, NaturalImageDataset, parse_ni_metadata
from chirp_analysis.src.dataset import ERGChirpDataset
from chirp_analysis.src.models import ImprovedBinaryCNN

RETINA_ROOT   = os.path.abspath(os.path.join(ML_ROOT, '..'))
NI_DATA_DIR   = os.path.join(RETINA_ROOT, 'natural_image_analysis', 'processed_data')
CH_DATA_DIR   = os.path.join(RETINA_ROOT, 'chirp_analysis', 'processed_data')
CH_CACHE_DIR  = os.path.join(ML_ROOT, 'chirp_analysis', 'data', 'cache')
NI_CACHE_PATH = os.path.join(NI_ROOT, 'data', 'cache', 'ni_dataset.pt')
META_CSV    = os.path.join(ROOT, 'data', 'metadata.csv')
META_CSV     = os.path.join(ML_ROOT, 'chirp_analysis', 'data', 'metadata.csv')

NI_MOD_DIR = os.path.join(NI_ROOT, 'results', 'models')
CH_MOD_DIR = os.path.join(ML_ROOT, 'chirp_analysis', 'results', 'models')
FIG_DIR    = os.path.join(NI_ROOT, 'results', 'figures')
TAB_DIR    = os.path.join(NI_ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load Datasets and Metadata ───────────────────────────────────────────────
print("Parsing metadata …")
df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
subject_to_label = {r.Subject: (1 if '5xFAD' in r.Group else 0) for _, r in df_meta.iterrows()}

print("Loading Chirp Dataset …")
ch_dataset = ERGChirpDataset(CH_DATA_DIR, META_CSV, segment='amplitude', cache_dir=CH_CACHE_DIR)
ch_trial_accum = {}
for i in range(len(ch_dataset)):
    sig, _, name = ch_dataset[i]
    subj = name.split('_trial_')[0]
    if subj in subject_to_label:
        arr = sig.float().numpy().flatten()
        ch_trial_accum.setdefault(subj, []).append(arr)
ch_signals_subj = {s: np.mean(v, axis=0) for s, v in ch_trial_accum.items()}

print("Loading Natural Images Dataset …")
ni_metadata = parse_ni_metadata(META_CSV)
ni_dataset = NaturalImageDataset(NI_DATA_DIR, ni_metadata, cache_path=NI_CACHE_PATH)
ni_signals_subj = {}
for i in range(len(ni_dataset)):
    sig, _, lbl, subj = ni_dataset[i]
    ni_signals_subj[subj] = sig.float().numpy()

common_subjects = sorted(list(set(ch_signals_subj.keys()) & set(ni_signals_subj.keys())))
labels = np.array([subject_to_label[s] for s in common_subjects])
print(f"Found {len(common_subjects)} common subjects (WT={np.sum(labels==0)}, 5xFAD={np.sum(labels==1)})")

# ── Load Ensembles ───────────────────────────────────────────────────────────
print("Loading trained CNN models for latent extraction...")
ch_models = []
for fold in range(1, 6):
    m = ImprovedBinaryCNN().to(device)
    state = torch.load(os.path.join(CH_MOD_DIR, f'05_cnn_fold_{fold}.pt'), map_location=device)
    m.load_state_dict(state)
    m.eval()
    ch_models.append(m)

ni_models = []
for fold in range(1, 6):
    m = ImprovedNICNN_NoAge().to(device)
    state = torch.load(os.path.join(NI_MOD_DIR, f'07_improved_ni_cnn_fold_{fold}.pt'), map_location=device)
    m.load_state_dict(state)
    m.eval()
    ni_models.append(m)

# ── Extract Latents ──────────────────────────────────────────────────────────
def extract_tsp_latents(signal_np, model_list, is_ni=False):
    """ Extract the 96-dimensional TemporalStatPool features, averaged over ensemble """
    # signal_np shape:
    # Chirp: [1, T] -> requires [1, 1, T] 
    # NI: [10, T] -> requires [1, 10, T]
    sig_t = torch.tensor(signal_np, dtype=torch.float32).to(device)
    if is_ni:
        sig_t = sig_t.unsqueeze(0) # [1, 10, T]
    else:
        sig_t = sig_t.unsqueeze(0).unsqueeze(0) # [1, 1, T]
        
    feats = []
    for model in model_list:
        _store = {}
        def _feat_hook(m, inp, out):
            _store['feat'] = out.detach()
        h = model.conv[-1].register_forward_hook(_feat_hook)
        
        with torch.no_grad():
            model(sig_t)
        
        h.remove()
        A = _store['feat'][0] # [32, 22] or similar
        tsp = torch.cat([A.mean(dim=-1), A.amax(dim=-1), A.std(dim=-1, unbiased=False)], dim=0) # [96]
        feats.append(tsp.cpu().float().numpy())
    return np.mean(feats, axis=0)

print("Extracting TemporalStatPool bottleneck features...")
X_ch = np.array([extract_tsp_latents(ch_signals_subj[s], ch_models, is_ni=False) for s in common_subjects])
X_ni = np.array([extract_tsp_latents(ni_signals_subj[s], ni_models, is_ni=True)  for s in common_subjects])

# ── Canonical Correlation Analysis (CCA) ─────────────────────────────────────
print("Performing CCA (n_components=1)...")
# Optional: Z-score before CCA for stability
from sklearn.preprocessing import StandardScaler
sc_ch = StandardScaler()
sc_ni = StandardScaler()
X_ch_z = sc_ch.fit_transform(X_ch)
X_ni_z = sc_ni.fit_transform(X_ni)

cca = CCA(n_components=1, max_iter=1000)
cca.fit(X_ch_z, X_ni_z)
X_ch_c, X_ni_c = cca.transform(X_ch_z, X_ni_z)

# Canonical Correlation coefficient
r = np.corrcoef(X_ch_c[:, 0], X_ni_c[:, 0])[0, 1]

# ── Save weights ─────────────────────────────────────────────────────────────
# Weights mapping from original 96 features to CCA dimension 1
ch_weights = cca.x_weights_[:, 0]
ni_weights = cca.y_weights_[:, 0]

# Feature names
stat_names = (['mean'] * 32 + ['max'] * 32 + ['std'] * 32)
feat_names = [f'{s}_ch{k%32}' for k, s in enumerate(stat_names)]

df_weights = pd.DataFrame({
    'Feature': feat_names,
    'Stat_Type': stat_names,
    'Channel': [i % 32 for i in range(96)],
    'Chirp_CCA_Weight': ch_weights,
    'NI_CCA_Weight': ni_weights
})
df_weights.to_csv(os.path.join(TAB_DIR, '09_cca_components.csv'), index=False)
print("  Saved 09_cca_components.csv")

# ── Visualization ────────────────────────────────────────────────────────────
print("Generating visualization...")
fig = plt.figure(figsize=(16, 6))
fig.subplots_adjust(wspace=0.3)

# Panel A: Canonical Correlation Scatter
ax1 = fig.add_subplot(1, 2, 1)
wt_mask  = labels == 0
fad_mask = labels == 1
C_WT, C_FAD = '#2980b9', '#c0392b'

ax1.scatter(X_ch_c[wt_mask, 0], X_ni_c[wt_mask, 0], color=C_WT, s=60, alpha=0.8, edgecolors='white', label='WT')
ax1.scatter(X_ch_c[fad_mask, 0], X_ni_c[fad_mask, 0], color=C_FAD, s=60, alpha=0.8, edgecolors='white', label='5xFAD')
ax1.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5, label='Identity Line')
ax1.set_xlabel('Chirp Shared Latent (CCA Dimension 1)', fontsize=10)
ax1.set_ylabel('Natural Image Shared Latent (CCA Dimension 1)', fontsize=10)
ax1.set_title(f'A. Cross-Stimulus CCA Canonical Variates\n(Correlation: r = {r:.3f})', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Panel B: Weights comparison across the 96 features (violin plots)
ax2 = fig.add_subplot(1, 2, 2)
vp_ch = ax2.violinplot([np.abs(ch_weights[0:32]), np.abs(ch_weights[32:64]), np.abs(ch_weights[64:96])], positions=[0, 3, 6], showmedians=True)
vp_ni = ax2.violinplot([np.abs(ni_weights[0:32]), np.abs(ni_weights[32:64]), np.abs(ni_weights[64:96])], positions=[1, 4, 7], showmedians=True)

for pc in vp_ch['bodies']: pc.set_facecolor('#8e44ad'); pc.set_alpha(0.6)
vp_ch['cmedians'].set_color('black')
for pc in vp_ni['bodies']: pc.set_facecolor('#27ae60'); pc.set_alpha(0.6)
vp_ni['cmedians'].set_color('black')

ax2.set_xticks([0.5, 3.5, 6.5])
ax2.set_xticklabels(['Mean Filters', 'Max Filters', 'Std Filters'])
ax2.set_ylabel('Absolute CCA Feature Loading (Weight)')
ax2.set_title('B. CCA Loadings (Importance) for Shared Axis\nChirp (Purple) vs Natural Image (Green)', fontweight='bold')

from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color='#8e44ad', alpha=0.6, label='Chirp Weights'),
                    Patch(color='#27ae60', alpha=0.6, label='Natural Image Weights')], fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Cross-Stimulus Shared Deficit Mining via Canonical Correlation Analysis\n'
             'Finding the exact latent dimension perfectly correlated across both biological stimulus conditions', fontweight='bold', fontsize=12)
plt.savefig(os.path.join(FIG_DIR, '09_cross_stimulus_cca.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Done! Results saved in {TAB_DIR} and {FIG_DIR}")
