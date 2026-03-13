"""
run_all.py — Master Pipeline Runner
====================================
Runs the complete machine learning analysis in the correct order.
Standardizes on a "No Age" baseline as established in step 03.

Sequence (Chirp):
  01: Population Demographics
  02: 4-Class CNN (Age x Genotype)
  03: Age Effect Comparison (Baseline Confirmation)
  04: Segment Comparison (amplitude selection)
  05: Binary CNN — original baseline
  06: Hand-crafted ML
  07: Complexity ML
  09: Sanity Checks

Sequence (Natural Image):
  10: NI CNN — original baseline
  11: Hand-crafted ML
  12: Complexity ML
  13: NI Comparison Figure
  14: Fusion Model (Dual-Input)
  15: Cross-Stimulus Final Comparison

Statistical Survey:
  16: Chirp HC Stats (4-group comparison)
  17: Chirp Comp Stats
  18: NI HC Stats
  19: NI Comp Stats

Architecture-Improved CNNs (steps 20-21):
  20: Improved Chirp CNN  — TemporalStatPool + InstanceNorm, all segments
  21: Improved NI CNN     — same architectural fixes applied to NI

CNN Interpretability — Chirp (steps 22-28):
  22: Grad-CAM + Integrated Gradients
  23: Attention CNN (segment + spatial attention)
  24: Bayesian Input Optimization (optimal chirp signal)
  25: Virtual Blockade (band-selective silencing)
  26: Counterfactual Edits (minimal perturbation)
  27: Counterfactual Spectral Analysis
  28: Symmetrical Story Figure (counterfactual summary)

Gain-Tracking Discovery (steps 29-30):
  29: HC + Gain-Tracking features  — AUC=0.810
  30: Muted Gain-Tracking Synthesis (3-level evidence)

CNN Interpretability — Natural Image (steps 31-32):
  31: NI Grad-CAM + Integrated Gradients
  32: NI Bayesian Input Optimization

Final Comparisons (step 08):
  08: Chirp comprehensive comparison (CNN + HC + Complexity + HC+GT)
      NOTE: step 08 runs last so it can include HC+Gain-Tracking results.
      Use --step 8 to re-run this figure in isolation.

Usage
-----
    python run_all.py                  # skip training if results exist
    python run_all.py --force          # force-retrain everything
    python run_all.py --local-only     # only steps that need NO H5 data (no 25 GB required)
    python run_all.py --chirp          # chirp analysis only
    python run_all.py --ni             # natural image analysis only
    python run_all.py --step 5         # single step by number
    python run_all.py --improved       # run only improved-CNN steps (20-21)
    python run_all.py --interp         # run only interpretability steps (22-32)
    python run_all.py --from-step 22   # start from a specific step
"""

import argparse
import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Steps that run entirely from pre-computed CSVs in data/ and results/tables/.
# They do NOT load H5 electrode data and do NOT require the ~25 GB processed_data/.
LOCAL_ONLY_STEPS = {1, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19}

# (number, label, subdir, script, expected_outputs_relative_to_subdir)
STEPS = [
    # ── CHIRP ANALYSIS: FOUNDATION ─────────────────────────────────────────────
    (1,  'Population Characterization',
         'chirp_analysis', 'src/01_population_characterization.py',
         ['results/tables/01_population_summary.csv']),

    (2,  '4-Class CNN (Age x Genotype)',
         'chirp_analysis', 'src/02_train_4class_cnn.py',
         ['results/tables/02_4class_segment_summary.csv', 'results/figures/02_4class_summary.png']),

    (3,  'Age Effect Comparison (Baseline Confirmation)',
         'chirp_analysis', 'src/03_age_effect_comparison.py',
         ['results/tables/03_age_effect_comparison.csv', 'results/figures/03_age_effect_comparison.png']),

    (4,  'Compare Chirp Segments (selection of amplitude)',
         'chirp_analysis', 'src/04_compare_chirp_segments.py',
         ['results/figures/04_chirp_segment_comparison.png']),

    (5,  'Binary CNN (Original Baseline)',
         'chirp_analysis', 'src/05_train_binary_cnn.py',
         ['results/tables/05_cnn_fold_results.csv', 'results/tables/05_cnn_probs.csv',
          'results/figures/05_cnn_training_curves.png']),

    (6,  'Hand-crafted ML — Chirp',
         'chirp_analysis', 'src/06_handcrafted_ml.py',
         ['results/tables/06_handcrafted_results.csv']),

    (7,  'Complexity ML — Chirp',
         'chirp_analysis', 'src/07_complexity_ml.py',
         ['results/tables/07_complexity_results.csv']),

    (9,  'Sanity Checks (Leakage + Age Classification)',
         'chirp_analysis', 'src/09_sanity_checks.py',
         ['results/tables/09_age_classification_results.csv']),

    # ── NATURAL IMAGE ANALYSIS ────────────────────────────────────────────────
    (10, 'NI CNN (No Age/Sex baseline)',
         'natural_image_analysis', 'src/01_train_ni_cnn.py',
         ['results/tables/01_ni_cnn_noage_probs.csv', 'results/figures/01_ni_cnn_training_curves.png']),

    (11, 'Hand-crafted ML — NI',
         'natural_image_analysis', 'src/02_handcrafted_ml.py',
         ['results/tables/02_ni_handcrafted_results.csv']),

    (12, 'Complexity ML — NI',
         'natural_image_analysis', 'src/03_complexity_ml.py',
         ['results/tables/03_ni_complexity_results.csv']),

    (13, 'NI Comparison Figure',
         'natural_image_analysis', 'src/04_ni_comparison.py',
         ['results/figures/04_ni_comparison.png']),

    (14, 'Dual-Input Fusion CNN (Chirp+NI)',
         'natural_image_analysis', 'src/05_fusion_model.py',
         ['results/tables/05_fusion_noage_probs.csv']),

    (15, 'Cross-Stimulus Final Comparison + ML Value Figure',
         'natural_image_analysis', 'src/06_combined_comparison.py',
         ['results/figures/06_combined_comparison.png', 'results/figures/ROC_Value_of_ML.png']),

    # ── STATISTICAL SURVEY ────────────────────────────────────────────────────
    (16, 'Chirp Hand-crafted Stats (4-group)',
         'chirp_analysis', 'src/10_stat_features_hc.py',
         ['results/tables/10_chirp_hc_stats.csv', 'results/figures/10_chirp_hc_stats_distribution.png']),

    (17, 'Chirp Complexity Stats (4-group)',
         'chirp_analysis', 'src/11_stat_features_comp.py',
         ['results/tables/11_chirp_comp_stats.csv', 'results/figures/11_chirp_comp_stats_distribution.png']),

    (18, 'NI Hand-crafted Stats (4-group)',
         'natural_image_analysis', 'src/10_stat_features_hc.py',
         ['results/tables/10_ni_hc_stats.csv', 'results/figures/10_ni_hc_stats_distribution.png']),

    (19, 'NI Complexity Stats (4-group)',
         'natural_image_analysis', 'src/11_stat_features_comp.py',
         ['results/tables/11_ni_comp_stats.csv', 'results/figures/11_ni_comp_stats_distribution.png']),

    # ── ARCHITECTURE-IMPROVED CNNs ────────────────────────────────────────────
    # TemporalStatPool [mean+max+std], InstanceNorm1d, channels 8/16/32,
    # dropout=0.5, weight_decay=1e-2.  Pooled CV AUC reported throughout.
    (20, 'Improved Chirp CNN (TemporalStatPool + InstanceNorm) — all segments',
         'chirp_analysis', 'src/12_improved_chirp_cnn.py',
         ['results/tables/12_improved_chirp_comparison.csv',
          'results/figures/12_improved_chirp_cnn.png']),

    (21, 'Improved NI CNN (TemporalStatPool + InstanceNorm)',
         'natural_image_analysis', 'src/07_improved_ni_cnn.py',
         ['results/tables/07_ni_cnn_comparison.csv',
          'results/figures/07_improved_ni_cnn.png']),

    # ── CNN INTERPRETABILITY — CHIRP ──────────────────────────────────────────
    # Steps 22-28 unpack *what* the improved chirp CNN has learned:
    #   Grad-CAM → attention maps → Bayesian optimal signal → blockade experiment
    #   → counterfactual edits → story figure.
    # Taken together they point to the fundamental-frequency gain trajectory
    # as the dominant cue — motivating the gain-tracking features in step 29.
    (22, 'Chirp CNN Interpretability — Grad-CAM + Integrated Gradients',
         'chirp_analysis', 'src/13_cnn_interpretability.py',
         ['results/figures/13_b_gradcam.png',
          'results/figures/13_c_integrated_grads.png']),

    (23, 'Chirp Attention CNN (Segment + Spatial Attention)',
         'chirp_analysis', 'src/14_attention_cnn_chirp.py',
         ['results/figures/14_a_segment_comparison.png']),

    (24, 'Chirp Bayesian Input Optimization (optimal stimulus)',
         'chirp_analysis', 'src/15_bayesian_input_optimization.py',
         ['results/figures/15_a_optimal_signals.png',
          'results/figures/15_b_pca_coefficients.png']),

    (25, 'Chirp Virtual Blockade — Band-Selective Silencing',
         'chirp_analysis', 'src/19_virtual_blockade.py',
         ['results/figures/19_virtual_blockade.png',
          'results/tables/19_virtual_blockade.csv']),

    (26, 'Chirp Counterfactual Edits (minimal perturbation to fool CNN)',
         'chirp_analysis', 'src/20_minimal_cure.py',
         ['results/figures/20_minimal_cure.png']),

    (27, 'Chirp Counterfactual Spectral Analysis',
         'chirp_analysis', 'src/21_cure_spectral_analysis.py',
         ['results/figures/21_cure_spectral_analysis.png']),

    (28, 'Chirp Symmetrical Story Figure (counterfactual summary)',
         'chirp_analysis', 'src/22_symmetrical_story_figure.py',
         ['results/figures/22_symmetrical_story.png']),

    # ── GAIN-TRACKING DISCOVERY ───────────────────────────────────────────────
    # The interpretability chain (22-28) identifies the fundamental-frequency
    # gain trajectory as the key cue.  Step 29 tests this directly as hand-
    # crafted features; step 30 synthesises the three levels of evidence.
    (29, 'Chirp Gain-Tracking HC — AUC=0.810 (best chirp result)',
         'chirp_analysis', 'src/23_gain_tracking_hc.py',
         ['results/tables/23_gain_tracking_results.csv',
          'results/tables/23_hc_gain_probs.csv',
          'results/figures/23_gain_tracking_comparison.png']),

    (30, 'Chirp Muted Gain-Tracking Synthesis (3-level evidence)',
         'chirp_analysis', 'src/24_muted_gain_tracking.py',
         ['results/figures/24_muted_gain_tracking.png']),

    # ── CNN INTERPRETABILITY — NATURAL IMAGE ──────────────────────────────────
    (31, 'NI CNN Interpretability — Grad-CAM + Integrated Gradients',
         'natural_image_analysis', 'src/08_ni_cnn_interpretability.py',
         ['results/figures/08_b_ni_gradcam.png',
          'results/figures/08_c_ni_integrated_grads.png']),

    (32, 'NI Bayesian Input Optimization (optimal natural-image response)',
         'natural_image_analysis', 'src/12_bayesian_input_optimization_ni.py',
         ['results/figures/12_a_optimal_signals.png',
          'results/figures/12_b_pca_coefficients.png']),

    # ── FINAL CHIRP COMPARISON ────────────────────────────────────────────────
    # Runs LAST so it can include HC+Gain-Tracking results (step 29).
    # Produces a single 4-method ROC + bar chart + confusion-matrix figure.
    (8,  'Chirp Final Comparison — CNN + HC + Complexity + HC+Gain-Tracking',
         'chirp_analysis', 'src/08_chirp_comparison.py',
         ['results/figures/08_chirp_comparison.png']),
]

def clean_old_files():
    print("\n🧹 Cleaning old/obsolete files (including old numbering)...")
    dirs = ['chirp_analysis/results/figures', 'chirp_analysis/results/tables',
            'natural_image_analysis/results/figures', 'natural_image_analysis/results/tables']
    
    for d in dirs:
        path = os.path.join(BASE_DIR, d)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.png') or f.endswith('.csv')]
            for f in files:
                # Keep current results if we don't want a full wipe, but here we wipe them
                os.remove(os.path.join(path, f))
    print("Done.\n")

def run_step(num, label, subdir, script, outputs, force_retrain=False):
    cwd = os.path.join(BASE_DIR, subdir)
    
    # Check if outputs exist
    outputs_exist = True
    for out in outputs:
        if not os.path.exists(os.path.join(cwd, out)):
            outputs_exist = False
            break
            
    if outputs_exist and not force_retrain:
        print(f"  STEP {num:02d}: {label} [SKIPPED - Results exist]")
        return True

    cmd = [sys.executable, script]
    print(f"\n{'='*75}")
    print(f"  STEP {num:02d}: {label}")
    print(f"  Script: {script}")
    if force_retrain: print(f"  (Forcing retrain)")
    print(f"{'='*75}\n")
    
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"\n❌  Step {num} FAILED (exit code {result.returncode})\n")
        return False
    print(f"\n✓  Step {num} done  [{elapsed:.1f}s]\n")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run all analysis steps.')
    parser.add_argument('--clean',     action='store_true', help='Wipe all results first.')
    parser.add_argument('--force',     action='store_true', help='Force retrain all steps.')
    parser.add_argument('--step',      type=int, default=None, help='Run only a specific step number.')
    parser.add_argument('--chirp',     action='store_true', help='Run chirp steps only.')
    parser.add_argument('--ni',        action='store_true', help='Run NI/Fusion steps only.')
    parser.add_argument('--improved',  action='store_true', help='Run improved-CNN steps only (20-21).')
    parser.add_argument('--interp',     action='store_true', help='Run interpretability steps only (22-32).')
    parser.add_argument('--local-only', action='store_true',
                        help='Run only steps that work without the 25 GB H5 data '
                             f'(steps {sorted(LOCAL_ONLY_STEPS)}). '
                             'Skips all CNN training and interpretability steps.')
    parser.add_argument('--from-step', type=int, default=None,
                        help='Start from this position in the pipeline (by list order, not step number).')
    args = parser.parse_args()

    local_only = getattr(args, 'local_only', False)

    if args.clean:
        clean_old_files()
        if not (args.force or args.step or args.chirp or args.ni
                or args.improved or args.interp or local_only):
            return

    steps_to_run = list(STEPS)

    if args.step:
        steps_to_run = [s for s in STEPS if s[0] == args.step]
    elif args.improved:
        steps_to_run = [s for s in STEPS if s[0] in (20, 21)]
    elif args.interp:
        steps_to_run = [s for s in STEPS if 22 <= s[0] <= 32]
    elif local_only:
        steps_to_run = [s for s in STEPS if s[0] in LOCAL_ONLY_STEPS]
    elif args.chirp:
        steps_to_run = [s for s in STEPS if s[2] == 'chirp_analysis']
    elif args.ni:
        steps_to_run = [s for s in STEPS if s[2] == 'natural_image_analysis']

    if not args.step and args.from_step is not None:
        # Skip steps that appear before position from_step in the list
        all_nums = [s[0] for s in STEPS]
        start_pos = next((i for i, n in enumerate(all_nums) if n == args.from_step), 0)
        skip_nums = set(all_nums[:start_pos])
        steps_to_run = [s for s in steps_to_run if s[0] not in skip_nums]

    print(f"\nMaster Pipeline Configuration:")
    print(f"  Force Retrain: {args.force}")
    if local_only:
        print(f"  Mode         : LOCAL-ONLY (no H5 data needed)")
    print(f"  Total Steps  : {len(steps_to_run)}")
    
    failed = []
    t_start = time.time()
    for s in steps_to_run:
        ok = run_step(s[0], s[1], s[2], s[3], s[4], force_retrain=args.force)
        if not ok:
            failed.append(s[0])
            break

    elapsed = time.time() - t_start
    print(f"\n{'='*75}")
    if failed:
        print(f"Pipeline ABORTED at step {failed[0]}")
    else:
        print(f"✅ Pipeline FINISHED in {elapsed/60:.1f} minutes")
    print(f"{'='*75}\n")

if __name__ == '__main__':
    main()
