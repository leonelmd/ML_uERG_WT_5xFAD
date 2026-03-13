"""
10_stat_features_hc.py
=======================
Statistical analysis of Natural Image Hand-crafted features across 4 groups.
Normality-based test selection (Parametric vs Non-Parametric).

Outputs:
- results/figures/10_ni_hc_stats_distribution.png
- results/tables/10_ni_hc_stats.csv
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(ROOT, 'data')
FIG_DIR  = os.path.join(ROOT, 'results', 'figures')
TAB_DIR  = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
df_meta  = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
df_feats = pd.read_csv(os.path.join(DATA_DIR, 'hand_crafted_features.csv'))
df = pd.merge(df_meta, df_feats, on='Subject')

GROUP_ORDER = ['WT young', 'WT adult', '5xFAD young', '5xFAD adult']
FEATURES = [
    'Signal_P2P', 'Signal_RMS', 'Signal_Max', 'Signal_Min', 'Signal_Std',
    'Power_Total', 'Power_Low', 'Power_Mid', 'Power_High'
]

# Ensure Group is categorical
df['Group'] = pd.Categorical(df['Group'], categories=GROUP_ORDER, ordered=True)

# ── Statistical Analysis ──────────────────────────────────────────────────────
stat_results = []
for feat in FEATURES:
    if feat not in df.columns: continue
    
    group_data = [df[df['Group'] == g][feat].dropna().values for g in GROUP_ORDER]
    
    # Shapiro-Wilk Normality Test
    is_normal = True
    for gd in group_data:
        if len(gd) < 3: is_normal = False; break
        _, p_norm = stats.shapiro(gd)
        if p_norm < 0.05: is_normal = False; break
            
    # Main Test Selection
    if is_normal:
        test_name = "One-way ANOVA"
        p_val_main = stats.f_oneway(*group_data)[1]
        ph_res = sp.posthoc_tukey(df[['Group', feat]].dropna(), val_col=feat, group_col='Group')
    else:
        test_name = "Kruskal-Wallis"
        p_val_main = stats.kruskal(*group_data)[1]
        ph_res = sp.posthoc_dunn(df[['Group', feat]].dropna(), val_col=feat, group_col='Group', p_adjust='bonferroni')

    sig_comps = []
    for i, g1 in enumerate(GROUP_ORDER):
        for j, g2 in enumerate(GROUP_ORDER):
            if i < j:
                p_adj = ph_res.loc[g1, g2]
                if p_adj < 0.05: sig_comps.append(f"{g1} vs {g2} (p={p_adj:.4f})")
    
    row = {
        'Feature': feat,
        'Normality': 'Normal' if is_normal else 'Non-Normal',
        'Main_Test': test_name,
        'Main_P': p_val_main,
        'Significant_Comps': "; ".join(sig_comps)
    }
    for g in GROUP_ORDER: row[f'Mean_{g}'] = df[df['Group'] == g][feat].mean()
    stat_results.append(row)

pd.DataFrame(stat_results).to_csv(os.path.join(TAB_DIR, '10_ni_hc_stats.csv'), index=False)

# ── Visualization ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = ['#2563EB', '#1D4ED8', '#E11D48', '#BE123C']

fig, axes = plt.subplots(3, 3, figsize=(24, 22), facecolor='#F8FAFC')
axes = axes.flatten()

for i, feat in enumerate(FEATURES):
    ax = axes[i]
    if feat not in df.columns: continue
    sns.violinplot(data=df, x='Group', y=feat, ax=ax, hue='Group', palette=COLORS, inner='box', alpha=0.7, legend=False)
    sns.stripplot(data=df, x='Group', y=feat, ax=ax, color='black', alpha=0.3, size=4, jitter=True)
    
    res_row = [r for r in stat_results if r['Feature'] == feat][0]
    p_val_main = res_row['Main_P']
    test_type = res_row['Main_Test']
    title_color = '#BE123C' if p_val_main < 0.05 else '#334155'
    ax.set_title(f"{feat}\n{test_type} P={p_val_main:.4f}", fontsize=14, fontweight='bold', color=title_color)
    
    # Build ph_res
    df_clean = df[['Group', feat]].dropna()
    is_normal = (res_row['Normality'] == 'Normal')
    if is_normal:
        ph_res = sp.posthoc_tukey(df_clean, val_col=feat, group_col='Group')
    else:
        ph_res = sp.posthoc_dunn(df_clean, val_col=feat, group_col='Group', p_adjust='bonferroni')

    y_max = df[feat].max(); y_min = df[feat].min(); y_range = y_max - y_min
    h_level = y_max + 0.1 * y_range
    
    sig_pairs = []
    for idx1, g1 in enumerate(GROUP_ORDER):
        for idx2, g2 in enumerate(GROUP_ORDER):
            if idx1 < idx2:
                p_adj = ph_res.loc[g1, g2]
                if p_adj < 0.05: sig_pairs.append((idx1, idx2, p_adj))
    sig_pairs.sort(key=lambda x: abs(x[0]-x[1]))

    for i1, i2, p_adj in sig_pairs:
        stars = '***' if p_adj < 0.001 else ('**' if p_adj < 0.01 else '*')
        ax.plot([i1, i1, i2, i2], [h_level, h_level+0.02*y_range, h_level+0.02*y_range, h_level], lw=1.5, c='#475569')
        ax.text((i1+i2)*0.5, h_level+0.02*y_range, f"{stars} (p={p_adj:.3f})", ha='center', va='bottom', color='#BE123C', fontsize=11, fontweight='bold')
        h_level += 0.08 * y_range 
            
    ax.set_ylim(top=h_level + 0.05 * y_range)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['WT\nYoung', 'WT\nAdult', '5xFAD\nYoung', '5xFAD\nAdult'], fontsize=11)
    for spine in ax.spines.values(): spine.set_color('#E2E8F0')

# Clear extra axes
for j in range(len(FEATURES), len(axes)): axes[j].axis('off')

fig.suptitle('Natural Image Hand-crafted Features: Statistical Survey', 
             fontsize=26, fontweight='bold', y=0.98, color='#1E293B')
fig.text(0.5, 0.94, 'Normality check (Shapiro-Wilk) determines Parametric (ANOVA+Tukey) vs Non-Parametric (Kruskal+Dunn).\nMultiple comparisons corrected via Tukey or Bonferroni.', 
         ha='center', fontsize=14, color='#475569', style='italic')

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig_out = os.path.join(FIG_DIR, '10_ni_hc_stats_distribution.png')
plt.savefig(fig_out, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
plt.close()
print(f"Distribution plot saved: {fig_out}")
