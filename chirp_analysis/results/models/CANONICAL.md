# Canonical / Default Model

## ★ DEFAULT MODEL ★

| Property | Value |
|----------|-------|
| **Class** | `ImprovedBinaryCNN` (defined in `src/models.py`) |
| **Segment** | `full` (samples 0–8750 of the chirp, 35 s at 250 Hz) |
| **Weights** | `12_improved_full_fold_{1-5}.pt` |
| **Pooled AUC** | **0.590** (5-fold subject-disjoint cross-validation, 46 subjects, clean cache) |
| **Trained by** | `src/12_improved_chirp_cnn.py` |

### How to load

```python
from models import load_default_ensemble

MOD_DIR = 'results/models'               # adjust to your root
models  = load_default_ensemble(MOD_DIR) # returns list of 5 eval-mode models
```

### Why this model?

* `BinaryCNN_NoAge` (scripts 02/05) achieves pooled AUC ≈ 0.529 (near chance) due to
  `AdaptiveAvgPool1d(1)` destroying all temporal structure.
* `ImprovedBinaryCNN` fixes this with `TemporalStatPool` ([mean, max, std]) +
  `InstanceNorm1d` at input + reduced channel counts + Dropout 0.5.
* AUC 0.590 is the **highest achieved on the full segment** with a CNN (clean 46-subject cache).

---

## All saved weights

| File prefix | Class | Segment | Pooled AUC | Script | Status |
|-------------|-------|---------|-----------|--------|--------|
| `02_cnn_noage_fold_*` | `BinaryCNN_NoAge` | amplitude | ~0.529 | 02 | **DEPRECATED** |
| `05_cnn_fold_*` | `BinaryCNN_NoAge` | amplitude | ~0.529 | 05 | **DEPRECATED** |
| `12_improved_amplitude_fold_*` | `ImprovedBinaryCNN` | amplitude | 0.565 | 12 | ok |
| `12_improved_full_fold_*` | `ImprovedBinaryCNN` | full | **0.590** | 12 | **★ CANONICAL** |
| `12_improved_amplitude_norm_fold_*` | `ImprovedBinaryCNN` | amplitude_norm | see table | 12 | ok |
| `12_improved_flash_fold_*` | `ImprovedBinaryCNN` | flash | see table | 12 | ok |
| `12_improved_frequency_fold_*` | `ImprovedBinaryCNN` | frequency | see table | 12 | ok |
| `14_attn_amplitude_fold_*` | `AttentionBinaryCNN` | amplitude | 0.584 | 14 | ok (lower AUC) |
| `14_attn_*_fold_*` | `AttentionBinaryCNN` | others | see table | 14 | ok (lower AUC) |

**Rule: for any new analysis, use `12_improved_full_fold_*` with `ImprovedBinaryCNN`
unless you have a specific reason to use another segment or architecture.**
