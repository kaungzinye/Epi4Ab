# Slide Evidence Assets

This file collects the concrete evidence assets you can use in slides.

## Slide 16 - EpiScan proof-of-concept / dead end

Static PNGs already available:

- `EpiScan-Private/plots/db1_test_results_visualization.png`
- `EpiScan-Private/plots/mAb159_vs_SARS-CoV-2_Spike_RBD_analysis.png`
- `EpiScan-Private/plots/mAb311_vs_SARS-CoV-2_Spike_RBD_analysis.png`
- `EpiScan-Private/plots/prediction_comparison.png`

Supporting docs/code for the brittleness story:

- `EpiScan-Private/scripts/3_generate_cdr_masks.py`
- `EpiScan-Private/scripts/4_generate_antigen_encoding.py`
- `EpiScan-Private/EpiScan/EpiScan/commands/epimapping.py`

## Slide 17 - Epi4Ab pipeline / internal infrastructure

There is not currently a packaged PNG in this repo for the earlier HER2 / 12-structure claim.

Closest existing artifact in-repo:

- `examples/output_inference/2025-11-26_GNNResNet_4/test_record/1n8z_BAC_final_result.txt`

Useful supporting docs:

- `EPI4AB_DEEP_DIVE.md`
- `docs/PIPELINE_DOCUMENTATION.md`

## Slide 18 - Phase 1 scaled synthetic calibration (Epi4Ab)

Preprocessing / scale-up evidence:

- `logs/epi4ab-preprocess-full-autodetect-35666537.out`
  - shows: `200` candidates -> `191/200` passed preprocess gate -> `191/191` passed nodes/edges -> `191/191` passed fill-edge

ProteinMPNN target-generation evidence:

- `logs/proteinmpnn-scores-full-autodetect-35671566.out`
  - shows successful writes of `proteinmpnn_scores.parquet`
  - final failure was only `8vtd_BAC` due to `scores=104 residues=105`

Scaled held-out Phase 1 run (190 usable complexes):

- Dashboard HTML (plots hub):
  - `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/plots/runs/phase1_scale190_s42/html/dashboard.html`
- Per-PDB summary CSV:
  - `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/phase1_scale190_noab_seed42/2026-03-10_GNNResNet_1/test_record/regression_summary.csv`
- Aggregate test metric file:
  - `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/phase1_scale190_noab_seed42/2026-03-10_GNNResNet_1/evaluation_mean_test_whole.txt`
- Aggregate train metric file:
  - `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/phase1_scale190_noab_seed42/2026-03-10_GNNResNet_1/evaluation_mean_train_whole.txt`
- Aggregate validate metric file:
  - `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/phase1_scale190_noab_seed42/2026-03-10_GNNResNet_1/evaluation_mean_validate_whole.txt`

Key numbers from that run:

- train/test split: `152 / 38`
- test MSE: `0.575773`
- macro Spearman: `~0.786`
- micro Spearman: `~0.812`

## Important note on images

Right now:

- `EpiScan` already has ready-made PNGs.
- `Epi4Ab` evidence is mainly in `dashboard.html` plus CSV / log files.

If you want slide-ready PNGs for the Epi4Ab Phase 1 dashboard, the next step is to export screenshots or render static plots from the dashboard/CSV.
