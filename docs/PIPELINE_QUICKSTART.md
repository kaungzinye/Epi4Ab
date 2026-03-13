# Pipeline Quickstart


Goal:
1) Preprocess structures into graphs (with gating/quarantine)
2) Phase 1: train a regression model on ProteinMPNN per-residue NLL (synthetic calibration)
3) Phase 2: fine-tune on Seqitope per-residue scores (the real target)

## Step 0: One-time setup

### 0.1 Load module and create a venv

```bash
module purge
module load python/3.11.7

cd /leonardo_work/EUHPC_D29_035/Epi4Ab
python -m venv venv
source venv/bin/activate
```

### 0.2 Install Python deps (use the pinned lockfile)

Do NOT rely on `requirements.txt` (it’s the older unpinned list).

Use the pinned snapshot from the working environment:

```bash
pip install -r requirements.lock.txt
```

### 0.3 Non-Python deps (required for preprocessing)

You need a PyMOL executable for `preprocess/nodes_edges.py`.
Set it via `.env` or your shell environment:

```bash
export PYMOL_EXECUTABLE=/path/to/pymol
```

### 0.4 Offline note (only matters if `use_pretrained` is enabled)

If `use_pretrained` is enabled in training params, the code uses HuggingFace Transformers ESM2.

Compute nodes are offline, so ESM2 weights/tokenizer files must already exist in the HuggingFace cache.
Recommended env vars for SLURM jobs:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Optional but recommended: make the cache shared:

```bash
export HF_HOME=/leonardo_work/EUHPC_D29_035/hf_cache
```

## Step 1: Preprocess (graphs + gates)

This produces `processed_data/` + `nodes_edges/` under a run folder `OUT_BASE`.

```bash
METADATA_FILE=/leonardo_work/EUHPC_D29_035/Epi4Ab/input/pdb_info_test3A.csv \
SRC_PROCESSED_DIR=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/processed_data \
sbatch slurm/preprocess_full_autodetect.sbatch
```

Find `OUT_BASE=...` in the SLURM `.out` log.

The gates write filtered metadata:
- `OUT_BASE/gates/preprocess/metadata.ok.preprocess.csv`
- `OUT_BASE/gates/nodes_edges/metadata.ok.nodes_edges.csv`
- `OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv`

## Step 2: Generate Phase 1 targets (ProteinMPNN NLL)

This writes `proteinmpnn_scores.parquet` into each `nodes_edges/<pdb_id>/` and validates them.

```bash
OUT_BASE=<paste OUT_BASE from Step 1>
sbatch slurm/proteinmpnn_scores_full_autodetect.sbatch
```

## Step 3: Phase 1 training (regression head)

This trains against ProteinMPNN targets using the gated PDB list.

```bash
OUT_BASE=<same OUT_BASE>
sbatch slurm/train_phase1_proteinmpnn_regression.sbatch
```

By default this script does a deterministic PDB-level train/test split (to avoid evaluating on the training set). You can override:

```bash
OUT_BASE=<same OUT_BASE> \
SPLIT_SEED=42 \
TEST_FRACTION=0.2 \
sbatch slurm/train_phase1_proteinmpnn_regression.sbatch
```

Training parameters:
- `parameters_phase1_proteinmpnn_regression.txt`

What Phase 1 proves:
- The regression head + MSE plumbing works end-to-end on your graph inputs.
- The pipeline can produce per-residue continuous outputs aligned with `resId`.

What Phase 1 does NOT prove:
- It does not prove epitope accuracy (ProteinMPNN NLL is not an epitope label).

Note: Phase 1 trains on raw ProteinMPNN NLL targets, so `output_activation=identity` is used (raw regression output, not a probability).

## Optional: Visualize Phase 1 regression outputs

Phase 1 training does not automatically emit per-PDB `*_final_result.txt` files.
To visualize, run regression inference using the trained `model.pt`, then build a dashboard.

1) Run inference (produces `test_record/*_final_result.txt` with `pred_score` and `true_score`):

```bash
OUT_BASE=<same OUT_BASE>
MODEL_DIR=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/<RUN_TAG>/<DATE>_GNNResNet_1
sbatch slurm/inference_phase1_proteinmpnn_regression.sbatch
```

2) Create an interactive dashboard:

```bash
python scripts/visualize_results.py \
  --test_record_dir <INFERENCE_OUT_DIR>/<DATE>_<MODEL>/test_record
```

## Step 4: Phase 2 (next): Seqitope fine-tune

Phase 2 switches to real labels per PDB:
- `nodes_edges/<pdb_id>/node_label_seqitope.parquet` with columns `resId, score` where `score` is in `[0,1]`.

Recommended Phase 2 approach:
1) Build/convert Seqitope labels into the per-PDB parquet format above.
2) Validate labels alignment:
   - `python scripts/validate_pipeline.py --metadata <csv> --nodes_edges_dir <...> --step labels --target_type seqitope --target_file node_label_seqitope.parquet --target_column score`
3) Train with:
   - `target_type=seqitope`
   - `target_file=node_label_seqitope.parquet`
   - `target_column=score`
   - `output_activation=sigmoid`

When you have Seqitope labels, do a quick ablation:
- train from scratch vs initialize from the Phase 1 checkpoint

## Notes: pretrained vs non-pretrained

- `use_pretrained` ON: adds ESM2 sequence embeddings as extra node features during training/inference.
- `use_pretrained` OFF: skips ESM2 entirely.

For the detailed explanation (including freeze vs fine-tune / backprop into ESM2), see:
- `docs/PHASE1_REGRESSION_PIPELINE.md`

## Start Phase 2 (Seqitope)

Use the Phase 2 parameters and SLURM wrappers. Same regression head as Phase 1; only labels/activation differ.

```bash
PARAMETERS_FILE=parameters_phase2_seqitope_regression.txt OUT_BASE=<.../upstream_preprocess_autodetect/<RUN_TAG>> METADATA_OK=<your seqitope metadata csv> sbatch slurm/validate_seqitope.sbatch

# Training (after labels available)
sbatch slurm/train_phase1_proteinmpnn_regression.sbatch   # reuse epi_graph with Phase 2 params (mse + sigmoid + target_type=seqitope)
```
