# Pipeline Quickstart


Goal:
1) Preprocess structures into graphs (with gating/quarantine)
2) Generate ProteinMPNN per-residue regression targets
3) Train Phase 1 regression model

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

## Step 2: Generate Phase 1 targets (ProteinMPNN scores)

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

Training parameters:
- `parameters_phase1_proteinmpnn_regression.txt`

## Step 4: Phase 2 placeholder (Seqitope fine-tune)

Phase 2 will switch to real labels per PDB:
- `nodes_edges/<pdb_id>/node_label_seqitope.parquet` with `resId, score`

Training should switch to:
- `target_type=seqitope`
- `target_file=node_label_seqitope.parquet`

## Notes: pretrained vs non-pretrained

- `use_pretrained` ON: adds ESM2 sequence embeddings as extra node features during training/inference.
- `use_pretrained` OFF: skips ESM2 entirely.

For the detailed explanation (including freeze vs fine-tune / backprop into ESM2), see:
- `docs/PHASE1_REGRESSION_PIPELINE.md`
