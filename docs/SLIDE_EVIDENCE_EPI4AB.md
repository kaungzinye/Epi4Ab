# Epi4Ab Slide Evidence

Use this file as a quick index for slide-ready evidence assets from the Epi4Ab runs.

## Phase 1 pilot run (small held-out split)

Source run:
- `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/phase1_rawnll_split_20260225/2026-02-25_GNNResNet_1`

Slide-ready PNGs:
- `docs/slide_assets/phase1_pilot/summary_card.png`
- `docs/slide_assets/phase1_pilot/scatter_true_vs_pred.png`
- `docs/slide_assets/phase1_pilot/per_pdb_metrics.png`

Interactive / raw evidence:
- Dashboard HTML: `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/phase1_rawnll_split_20260225/2026-02-25_GNNResNet_1/dashboard.html`
- Summary CSV: `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/phase1_rawnll_split_20260225/2026-02-25_GNNResNet_1/test_record/regression_summary.csv`

## Phase 1 scaled synthetic run (190 usable complexes)

Source run:
- `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/phase1_scale190_noab_seed42/2026-03-10_GNNResNet_1`

Slide-ready PNGs:
- `docs/slide_assets/phase1_scale190/summary_card.png`
- `docs/slide_assets/phase1_scale190/scatter_true_vs_pred.png`
- `docs/slide_assets/phase1_scale190/per_pdb_metrics.png`

Interactive / raw evidence:
- Dashboard HTML: `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/phase1_scale190_noab_seed42/2026-03-10_GNNResNet_1/dashboard.html`
- Summary CSV: `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/phase1_scale190_noab_seed42/2026-03-10_GNNResNet_1/test_record/regression_summary.csv`
- Aggregate test MSE: `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/phase1_scale190_noab_seed42/2026-03-10_GNNResNet_1/evaluation_mean_test_whole.txt`

## Scale-up pipeline evidence

These are useful if you want to justify the synthetic Phase 1 expansion itself.

- Preprocess log (200 -> 191 gated):
  - `logs/epi4ab-preprocess-full-autodetect-35666537.out`
- ProteinMPNN log (191 attempted -> 190 success, 1 residue mismatch):
  - `logs/proteinmpnn-scores-full-autodetect-35671566.out`
  - `logs/proteinmpnn-scores-full-autodetect-35671566.err`
