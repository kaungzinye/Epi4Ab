# Epi4Ab -> Seqitope (Regression) Proposal and Current Implementation

This document describes the current plan (and the parts already implemented) to adapt Epi4Ab from its legacy classification workflow to a Seqitope regression workflow.

If you only need run commands, see `docs/PIPELINE_QUICKSTART.md`.

## Executive Summary

- **Phase 1 (implemented):** train a regression model to predict **ProteinMPNN per-residue NLL** (synthetic calibration target). This is *not* epitope learning.
- **Phase 2 (next):** fine-tune the same regression model to predict **Seqitope per-residue score** in `[0,1]` (the real epitope signal).

The point of Phase 1 is de-risking and initialization: verify the end-to-end regression pipeline (graph inputs, residue alignment, MSE training, inference outputs) and optionally warm-start representations before Phase 2.

## What Changed vs Legacy Epi4Ab

Legacy Epi4Ab:
- Classification output (3-class / interface labels) based on `node_label_pi.parquet` / `isInterface`.
- Visualization focused on confusion matrix, F1, etc.

Current Seqitope path:
- Regression output (1 scalar per residue) with `out_label=1`.
- Phase 1 targets: `proteinmpnn_scores.parquet` (`score` = raw ProteinMPNN NLL).
- Phase 2 targets: `node_label_seqitope.parquet` (`score` in `[0,1]`).
- Visualization is regression-first (`scripts/visualize_results.py`).

## Pipeline Overview (Current)

### Step A: Preprocessing (graphs + gating)

Produces:
- `processed_data/<pdb_id>/...` (antigen-only `lig.pdb`, sequences, and feature intermediates)
- `nodes_edges/<pdb_id>/...` (graph inputs: `node_feature.parquet`, `edge_index.parquet`, `edge_attribute_*.parquet`)

Robustness improvements included:
- Antigen chain auto-detection (`--autodetect_antigen_chain`) to prevent extracting the antibody chain as antigen.
- Gating/quarantine between steps (`scripts/pipeline_gate.py`), producing filtered metadata under `OUT_BASE/gates/...`.

Entry point:
- `slurm/preprocess_full_autodetect.sbatch`

### Step B: Phase 1 targets (ProteinMPNN)

For each PDB, Phase 1 creates:
- `nodes_edges/<pdb_id>/proteinmpnn_scores.parquet`

Format:
- `resId` (int)
- `score` (float) = **ProteinMPNN NLL** for the native amino acid at that residue

Interpretation:
- low NLL: native residue is structurally plausible under ProteinMPNN
- high NLL: native residue is structurally implausible (ProteinMPNN is "surprised")

This value is a **sequence-structure plausibility signal**, not an epitope probability.

Entry point:
- `slurm/proteinmpnn_scores_full_autodetect.sbatch`

### Step C: Phase 1 training (synthetic calibration)

Training configuration:
- `out_label=1`
- `loss_function=mse`
- `target_type=proteinmpnn`
- `target_file=proteinmpnn_scores.parquet`
- `target_column=score`
- `output_activation=identity` (raw regression output, because NLL is not in `[0,1]`)

Entry points:
- Parameters: `parameters_phase1_proteinmpnn_regression.txt`
- SLURM: `slurm/train_phase1_proteinmpnn_regression.sbatch`

### Step D: Phase 1 inference + regression dashboard

Why: Phase 1 training does not automatically emit per-PDB `*_final_result.txt` files. To visualize regression outputs, run inference with the trained model.

Entry points:
- Inference: `slurm/inference_phase1_proteinmpnn_regression.sbatch`
- Visualization: `scripts/visualize_results.py`

Outputs:
- `.../test_record/<pdb_id>_final_result.txt` with `pred_score` and `true_score`
- `.../dashboard.html` (regression plots)
- `.../test_record/regression_summary.csv` (per-PDB summary)

### Step E: Phase 2 (next) - fine-tune on Seqitope

Goal: predict **Seqitope per-residue score** in `[0,1]`.

Labels to produce per PDB:
- `nodes_edges/<pdb_id>/node_label_seqitope.parquet` with:
  - `resId` (int)
  - `score` (float in `[0,1]`)

Phase 2 training configuration:
- `target_type=seqitope`
- `target_file=node_label_seqitope.parquet`
- `target_column=score`
- `output_activation=sigmoid` (now the output is a probability-like score)

Phase 2 planning doc:
- `docs/SEQITOPE_PHASE2_PLAN.md`

## What Phase 1 Proved (and What It Did Not)

Phase 1 proved:
- The end-to-end regression pipeline works with the current Epi4Ab codebase:
  - preprocessing -> per-residue regression targets -> regression training -> regression inference -> per-residue outputs
- Residue alignment via `resId` is consistent through the pipeline.
- Training is stable (loss decreases; no NaNs; outputs are generated).

Phase 1 did NOT prove:
- Epitope prediction accuracy.
- Biological relationship between ProteinMPNN NLL and antibody binding.

Phase 2 is the first point where we can measure whether the model predicts the real Seqitope target.

## How To Run (Reference)

Use the runbook:
- `docs/PIPELINE_QUICKSTART.md`

Key SLURM entrypoints:
- Preprocess (gated): `slurm/preprocess_full_autodetect.sbatch`
- ProteinMPNN targets (gated): `slurm/proteinmpnn_scores_full_autodetect.sbatch`
- Phase 1 train: `slurm/train_phase1_proteinmpnn_regression.sbatch`
- Phase 1 inference: `slurm/inference_phase1_proteinmpnn_regression.sbatch`

## Notes on ESM2 / use_pretrained

If `use_pretrained` is enabled, ESM2 weights are loaded via HuggingFace Transformers at runtime.
Compute nodes are offline; weights must be cached ahead of time (see `docs/PHASE1_REGRESSION_PIPELINE.md`).

## Containerization Note

Leonardo compute nodes typically do not run Docker. If a container is required, use Apptainer/Singularity.
