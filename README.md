# Retinal ERG Genotype Classification via Machine Learning

> **Binary classification of Wild-Type (WT) vs. 5xFAD (Alzheimer's model) retinas
> from multi-electrode ERG recordings.**
> Default benchmark: **"No Age"** — model inputs are signal-only, isolating
> genotypic deficits from age-related drift.

Two visual stimulation paradigms are analysed in parallel:
**Chirp** (35 s flash + frequency sweep + amplitude sweep) and **Natural Image** (video of a natural landscape).
Three modelling strategies are compared: end-to-end CNN, hand-crafted features + ML,
and complexity/entropy features + ML.

---

## Data Requirements

### What's already in this repo (no extra data needed)

| Location | Contents |
|----------|----------|
| `chirp_analysis/data/` | `metadata.csv`, `hand_crafted_features.csv`, `complexity_features.csv` |
| `natural_image_analysis/data/` | same three CSVs for the NI dataset |
| `chirp_analysis/results/models/` | trained model weights (`12_improved_full_fold_*.pt`, attention models) |
| `natural_image_analysis/results/models/` | trained model weights (`07_improved_ni_cnn_fold_*.pt`) |

### What you need to run CNN training / interpretability (~25 GB)

The raw MEA electrode time-series live **outside** this repo.
Ask the author for the processed H5 files, then place them at:

```
/Users/<you>/retina/chirp_analysis/processed_data/          # 46 subjects × 360 MB
/Users/<you>/retina/natural_image_analysis/processed_data/   # 45 subjects × 107 MB
```

If your data lives elsewhere, edit the `DATA_DIR` variable near the top of each
training/interpretability script (e.g. `chirp_analysis/src/05_train_binary_cnn.py`).

> **Note:** All subject/group metadata is
> self-contained in `data/metadata.csv` inside each analysis folder.
> Only subjects whose H5 contains `electrode_mean/snr_7/data` (written by the
> Julia pipeline when ≥ 1 electrode passes SNR ≥ 7) are included.  Subjects
> that fail this criterion are absent from `metadata.csv` and are not used in
> any analysis (CNN, HC, or complexity).

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run without H5 data (local-only mode)
Runs only the steps that use the pre-computed CSVs already in the repo —
HC/complexity ML, statistical surveys, and comparison figures:
```bash
python run_all.py --local-only
```

Steps included in `--local-only`:

| Step | What it does |
|:----:|:-------------|
| 1 | Population demographics (metadata.csv only) |
| 6 | Chirp hand-crafted ML (LogReg / SVM / RF / k-NN) |
| 7 | Chirp complexity ML |
| 8 | Chirp final comparison figure (CNN + HC + HC+GT + Complexity) |
| 11 | NI hand-crafted ML |
| 12 | NI complexity ML |
| 13 | NI comparison figure |
| 15 | Cross-stimulus ML value figure |
| 16–19 | Statistical surveys (4-group, HC + complexity, chirp + NI) |

> Steps 8 and 15 read the CNN probabilities from pre-computed CSV files in
> `results/tables/` (already present). If those files are missing, the CNN
> panel will fall back to a synthetic placeholder.

### Run everything (requires H5 data)
```bash
python run_all.py                  # skip steps whose outputs already exist
python run_all.py --force          # force-retrain everything
python run_all.py --chirp          # chirp steps only
python run_all.py --ni             # NI steps only
python run_all.py --improved       # improved-CNN steps only (20–21)
python run_all.py --interp         # interpretability chain only (22–32)
python run_all.py --step 29        # single step by number
python run_all.py --from-step 22   # resume from a specific step
```

---

## Key Results

### Chirp stimulus — Pooled cross-validated AUC  *(N = 46)*

| Method | AUC |
|--------|-----|
| HC + Gain-Tracking (k-NN) | **0.810** |
| Hand-crafted ML (Log. Regression) | 0.735 |
| Complexity ML (Log. Regression) | 0.729 |
| Improved CNN — full segment | 0.590 |
| Original CNN — amplitude segment | 0.452 |

### Natural Image stimulus — Pooled cross-validated AUC

| Method | N | AUC |
|--------|---|-----|
| Hand-crafted ML (Log. Regression) | 42 | **0.782** |
| Improved CNN | 42 | 0.752 |
| Complexity ML (k-NN) | 42 | 0.736 |
| Original CNN | 42 | 0.609 |

> All methods use the same 42 subjects that pass the SNR ≥ 7 quality gate.
> FRCMSE entropy curves were computed via the Julia pipeline for all 42 subjects
> (the original 22 adult subjects plus the 20 young subjects added in this revision).

> **Architectural note (steps 20–21):** The original CNN used `AdaptiveAvgPool1d(1)`,
> which collapses the time series to a global mean — discarding all temporal
> structure. Replacing it with `TemporalStatPool` (mean + max + std) and adding
> per-sample `InstanceNorm1d` at input yielded:
> - **Chirp full**: AUC 0.452 → **0.590** (+0.138)
> - **Natural Image**: AUC 0.609 → **0.752** (+0.143)

---

## CNN Input Structure: Chirp vs Natural Image

The two CNN pipelines differ fundamentally in how they present data to the model,
reflecting the nature of each stimulus:

### Chirp — trial-level with subject-level voting

```
10 repetitions × 1 electrode-average signal (1, 2750)
       ↓
Train on individual repetitions as independent samples
       ↓
At inference: average 10 per-repetition probabilities → subject AUC
```

- Each repetition is one `(1, 2750)` tensor (amplitude segment, electrode-averaged)
- Training pool: ~460 samples from 46 subjects
- **Why trial-level?** The key discriminative feature is the *within-repetition*
  gain trajectory (temporal dynamics of the amplitude sweep). Each repetition
  independently carries that signal; stacking them as channels adds no extra
  information beyond what the model already sees.

### Natural Image — all-repetitions stacked as channels

```
10 repetitions of the same 10 s video scene
       ↓
Stacked into one (10, 2500) tensor per subject
       ↓
Train directly at subject level
```

- Each subject is one `(10, 2500)` tensor (all reps as input channels)
- Training pool: 48 samples from 48 subjects
- **Why stacked?** The 10 repetitions are near-identical presentations of the
  same scene. Stacking lets the CNN exploit *cross-repetition reliability* —
  whether a given spatial/temporal response pattern recurs consistently across
  trials is itself a discriminative feature. Tested empirically: trial-level
  approach for NI gives pooled AUC 0.533 vs 0.663 for stacked (−0.13).

> **Design rule:** use trial-level + voting when the discriminative cue lives
> *within* each repetition's temporal structure; use stacked channels when
> cross-repetition consistency is itself the signal.

---

## Pipeline

### Phase 0 — Feature Precomputation (requires H5 data)

Run once per dataset to generate the feature CSVs consumed by all ML scripts.

| Script | H5? | Description | Key output |
|:-------|:---:|:------------|:-----------|
| `chirp_analysis/src/00_compute_hc_features.py` | ✓ | 13 HC features from `electrode_mean/snr_7/data` | `chirp_analysis/data/hand_crafted_features.csv` |
| `chirp_analysis/src/00_compute_complexity_features.py` | ✓ | 8 nAUC/LRS features from pre-computed FRCMSE curves | `chirp_analysis/data/complexity_features.csv` |
| `natural_image_analysis/src/00_compute_hc_features.py` | ✓ | 9 HC features (signal + spectral) | `natural_image_analysis/data/hand_crafted_features.csv` |
| `natural_image_analysis/src/00_compute_complexity_features.py` | ✓ | 8 nAUC/LRS features from pre-computed FRCMSE curves | `natural_image_analysis/data/complexity_features.csv` |

Signal source: `electrode_mean/snr_7/data` in each subject's processed H5. Subjects without this key (no electrode passed SNR ≥ 7) are skipped.
Complexity source: `FRCMSE/0.2/electrode_mean/snr_7/curve` in each subject's entropy H5 (pre-computed by the Julia pipeline).

> The pre-computed CSVs are already checked in; re-run these only if you have new subjects or regenerated H5 files.

---

### Phase 1 — Chirp Foundation (`chirp_analysis/`)

| Step | Script | H5? | Description | Key output |
|:----:|:-------|:---:|:------------|:-----------|
| 1 | `01_population_characterization.py` | | Demographics, flash stats | `01_population_summary.csv` |
| 2 | `02_train_4class_cnn.py` | ✓ | 4-class (Genotype × Age) screen | `02_4class_summary.png` |
| 3 | `03_age_effect_comparison.py` | ✓ | Justifies "No Age" default | `03_age_effect_comparison.png` |
| 4 | `04_compare_chirp_segments.py` | ✓ | Selects `amplitude` segment | `04_chirp_segment_comparison.png` |
| 5 | `05_train_binary_cnn.py` | ✓ | Original CNN baseline | `05_cnn_probs.csv` |
| 6 | `06_handcrafted_ml.py` | | Peak / power / spectral features + ML | `06_handcrafted_results.csv` |
| 7 | `07_complexity_ml.py` | | nAUC / LRS entropy + ML | `07_complexity_results.csv` |
| 9 | `09_sanity_checks.py` | ✓ | Leakage check + age-confound test | `09_age_classification_results.csv` |
| 20 | `12_improved_chirp_cnn.py` | ✓ | **Improved CNN** (TemporalStatPool) | `12_improved_chirp_comparison.csv` |

### Phase 2 — Natural Image (`natural_image_analysis/`)

| Step | Script | H5? | Description | Key output |
|:----:|:-------|:---:|:------------|:-----------|
| 10 | `01_train_ni_cnn.py` | ✓ | Original NI CNN baseline | `01_ni_cnn_noage_probs.csv` |
| 11 | `02_handcrafted_ml.py` | | Signal / spectral features + ML | `02_ni_handcrafted_results.csv` |
| 12 | `03_complexity_ml.py` | | nAUC / LRS entropy + ML | `03_ni_complexity_results.csv` |
| 13 | `04_ni_comparison.py` | | CNN vs ML comparison figure | `04_ni_comparison.png` |
| 14 | `05_fusion_model.py` | ✓ | Dual-input CNN (Chirp + NI) | `05_fusion_noage_probs.csv` |
| 15 | `06_combined_comparison.py` | | Cross-stimulus ML value figure | `ROC_Value_of_ML.png` |
| 21 | `07_improved_ni_cnn.py` | ✓ | **Improved NI CNN** | `07_ni_cnn_comparison.csv` |

### Phase 3 — Statistical Surveys

| Step | Script | H5? | Description |
|:----:|:-------|:---:|:------------|
| 16 | `chirp_analysis/src/10_stat_features_hc.py` | | Chirp HC: 4-group stats |
| 17 | `chirp_analysis/src/11_stat_features_comp.py` | | Chirp complexity: 4-group stats |
| 18 | `natural_image_analysis/src/10_stat_features_hc.py` | | NI HC: 4-group stats |
| 19 | `natural_image_analysis/src/11_stat_features_comp.py` | | NI complexity: 4-group stats |

### Phase 4 — CNN Interpretability (Chirp)

> All steps 22–30 require the trained models from step 20 and H5 electrode data.

| Step | Script | Description | Key output |
|:----:|:-------|:------------|:-----------|
| 22 | `13_cnn_interpretability.py` | Grad-CAM + Integrated Gradients | `13_b_gradcam.png`, `13_c_integrated_grads.png` |
| 23 | `14_attention_cnn_chirp.py` | Segment + spatial attention maps | `14_a_segment_comparison.png` |
| 24 | `15_bayesian_input_optimization.py` | Optimal input stimulus (Bayesian) | `15_a_optimal_signals.png` |
| 25 | `19_virtual_blockade.py` | Band-selective silencing | `19_virtual_blockade.png` |
| 26 | `20_minimal_cure.py` | Counterfactual edits | `20_minimal_cure.png` |
| 27 | `21_cure_spectral_analysis.py` | Counterfactual spectral analysis | `21_cure_spectral_analysis.png` |
| 28 | `22_symmetrical_story_figure.py` | Counterfactual summary figure | `22_symmetrical_story.png` |
| 29 | `23_gain_tracking_hc.py` | **HC + Gain-Tracking: AUC=0.810** | `23_gain_tracking_comparison.png` |
| 30 | `24_muted_gain_tracking.py` | 3-level evidence synthesis | `24_muted_gain_tracking.png` |

### Phase 5 — CNN Interpretability (NI)

| Step | Script | H5? | Description | Key output |
|:----:|:-------|:---:|:------------|:-----------|
| 31 | `08_ni_cnn_interpretability.py` | ✓ | NI Grad-CAM + Integrated Gradients | `08_b_ni_gradcam.png` |
| 32 | `12_bayesian_input_optimization_ni.py` | ✓ | NI optimal input stimulus | `12_a_optimal_signals.png` |

### Final Comparison

| Step | Script | Description | Key output |
|:----:|:-------|:------------|:-----------|
| 8 | `08_chirp_comparison.py` | CNN + HC + Complexity + HC+GT (4-method ROC) | `08_chirp_comparison.png` |

---

## Scientific Design

### "No Age" Standard
Three input cases were compared in step 3:

| Case | Input |
|------|-------|
| Baseline (No Age) | ERG signal only |
| Binary Age | Signal + Young/Adult flag |
| Continuous Age | Signal + age in days |

Continuous/binary age degrades out-of-sample performance (the model learns
age-correlated signal drift rather than genotypic deficit). **"No Age"** is
therefore the default for all benchmark comparisons.

### Cross-validation
All splits are **strictly subject-disjoint**: every tissue piece belongs to
exactly one validation fold; all trials from that piece are kept together.
Step 9 (`09_sanity_checks.py`) verifies this with an explicit intersection test.

Paired preps from the same animal (e.g. `-t1` / `-t2` suffix) are treated as
**independent subjects** — different recording sessions, different retinal placements.

### Improved CNN architecture (steps 20–21)
The original `BinaryCNN_NoAge` used `AdaptiveAvgPool1d(1)` as the final
temporal aggregation, reducing any-length feature maps to a single global mean.
This loses all temporal ordering, peak information, and variability.

`ImprovedBinaryCNN` / `ImprovedNICNN_NoAge` replace this with:
```
TemporalStatPool: concat([mean, max, std], dim=time) → 3 × C features
```
and add `InstanceNorm1d(in_channels, affine=True)` at the input.
Channel counts reduced (8/16/32 vs. 16/32/64), dropout increased (0.5 vs. 0.3).

### Gain-tracking interpretation (steps 22–30)
The interpretability chain identifies the fundamental-frequency gain trajectory
in the amplitude sweep as the dominant discriminative cue.  This motivates four
new hand-crafted features (`AmpFund_RMS`, `AmpFund_frac`, `AmpEnv_slope_norm`,
`AmpEnv_late_early`) whose combination with the existing HC features raises
chirp AUC from 0.735 to **0.810**.

---

## Feature Definitions

### Hand-crafted (Chirp)
- **Flash branch**: peak max/min, P2P amplitude, RMS
- **Chirp frequency sweep**: RMS, std, spectral power (low / mid / high bands)
- **Chirp amplitude sweep**: RMS, max, P2P
- **Gain-tracking** (step 29): `AmpFund_RMS`, `AmpFund_frac`, `AmpEnv_slope_norm`, `AmpEnv_late_early`

### Hand-crafted (Natural Image)
- Signal max/min, P2P, RMS, std; spectral power bands

### Complexity / Entropy
- **nAUC**: normalised area under the multiscale entropy curve at τ = 15, 30, 45, all
- **LRS**: linear regression slope of the entropy curve (rate of change)

---

## Requirements
```
torch >= 1.13
scikit-learn >= 1.1
pandas, numpy, matplotlib, seaborn, h5py
```

---

*Analysis pipeline — WT vs 5xFAD ERG genotype classification.*
