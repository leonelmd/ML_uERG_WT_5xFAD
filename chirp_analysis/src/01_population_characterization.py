"""
01_population_characterization.py
====================================
Characterizes the study population by:
  1. Group-level demographics (n, age-months, sex if available)
  2. Signal amplitude statistics per group
  3. Summary table saved to results/tables/

Usage (from chirp_analysis/ folder):
    python src/01_population_characterization.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)          # chirp_analysis/
META_CSV = os.path.join(ROOT, 'data', 'metadata.csv')
HC_CSV   = os.path.join(ROOT, 'data', 'hand_crafted_features.csv')
FIG_DIR  = os.path.join(ROOT, 'results', 'figures')
TAB_DIR  = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading CSVs …")
df_meta = pd.read_csv(META_CSV)
df_meta['Subject'] = df_meta['Subject'].str.strip()
df_hc   = pd.read_csv(HC_CSV)
df_hc['Subject'] = df_hc['Subject'].str.strip()
df = pd.merge(df_meta, df_hc, on='Subject')
df['Age_Months'] = df['Age (Days)'] / 30.44
df['Label']    = df['Group'].str.contains('5xFAD').astype(int)
df['AgeGroup'] = df['Group'].str.contains('adult').map({True: 'Adult', False: 'Young'})
df['Genotype'] = df['Group'].str.contains('5xFAD').map({True: '5xFAD', False: 'WT'})

print(f"Dataset: {len(df)} tissue pieces\n")

# ── Summary table ──────────────────────────────────────────────────────────────
summary_rows = []
for group_name, gdf in df.groupby('Group', sort=False):
    row = {
        'Group':           group_name,
        'N':               len(gdf),
        'Age_Months_Mean': gdf['Age_Months'].mean() if 'Age_Months' in gdf else np.nan,
        'Age_Months_Std':  gdf['Age_Months'].std()  if 'Age_Months' in gdf else np.nan,
        'Flash_RMS_Mean':  gdf['Flash_RMS'].mean(),
        'Flash_RMS_Std':   gdf['Flash_RMS'].std(),
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(TAB_DIR, '01_population_summary.csv'), index=False)
print(summary_df.to_string(index=False))
print()

# Overall counts
print("Overall counts:")
print(f"  WT    : {(df['Genotype']=='WT').sum()}")
print(f"  5xFAD : {(df['Genotype']=='5xFAD').sum()}")
print(f"  Young : {(df['AgeGroup']=='Young').sum()}")
print(f"  Adult : {(df['AgeGroup']=='Adult').sum()}")
print()

# ── Figure 1: Sample sizes per group ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Population Characterization — Chirp Stimulus',
             fontsize=15, fontweight='bold', y=1.02)

COLORS = {'WT-Young': '#2196F3', 'WT-Adult': '#0D47A1',
          '5xFAD-Young': '#F44336', '5xFAD-Adult': '#B71C1C'}

group_order = ['WT young', 'WT adult', '5xFAD young', '5xFAD adult']
group_labels = ['WT\nYoung', 'WT\nAdult', '5xFAD\nYoung', '5xFAD\nAdult']
colors = ['#2196F3', '#0D47A1', '#F44336', '#B71C1C']
counts = [len(df[df['Group'] == g]) for g in group_order]

# Panel A: N per group
ax = axes[0]
bars = ax.bar(group_labels, counts, color=colors, edgecolor='black', linewidth=1.2, 
              alpha=0.85, width=0.5)
for bar, n in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2, str(n),
            ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Tissue Pieces', fontsize=12)
ax.set_title('A: Sample Size per Group', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(counts) * 1.25)

# Panel B: Age distribution
ax = axes[1]
if 'Age_Months' in df.columns and df['Age_Months'].notna().any():
    for g, color, label in zip(group_order, colors, group_labels):
        ages = df[df['Group'] == g]['Age_Months'].dropna()
        if len(ages):
            ax.scatter([label.replace('\n', ' ')] * len(ages), ages,
                       color=color, alpha=0.7, s=60, zorder=2)
    for g, color, label in zip(group_order, colors, group_labels):
        ages = df[df['Group'] == g]['Age_Months'].dropna()
        if len(ages):
            ax.plot([label.replace('\n', ' ')], [ages.mean()],
                    'D', color='black', markersize=10, zorder=3)
    ax.set_ylabel('Age (months)', fontsize=12)
    ax.set_title('B: Age Distribution per Group', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Age data not available', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.set_title('B: Age Distribution', fontsize=12, fontweight='bold')

# Panel C: Flash amplitude (proxy for retinal sensitivity)
ax = axes[2]
for g, color in zip(group_order, colors):
    vals = df[df['Group'] == g]['Flash_RMS'].dropna()
    ax.scatter([g.replace(' ', '\n')] * len(vals), vals,
               color=color, alpha=0.6, s=50, zorder=2)
    ax.errorbar(g.replace(' ', '\n'), vals.mean(), yerr=vals.std(),
                fmt='D', color='black', capsize=6, markersize=8, zorder=3)
ax.set_ylabel('Flash RMS Amplitude (a.u.)', fontsize=12)
ax.set_title('C: Flash Response Amplitude', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(range(4))
ax.set_xticklabels(['WT\nYoung', 'WT\nAdult', '5xFAD\nYoung', '5xFAD\nAdult'], fontsize=10)

plt.tight_layout()
out = os.path.join(FIG_DIR, '01_population_characterization.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved figure: {out}")
print(f"✓ Saved table:  {TAB_DIR}/population_summary.csv")
