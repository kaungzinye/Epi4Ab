# Phase 1 Regression Pipeline (ProteinMPNN Calibration)

This file is the Phase 1 deep-dive.

For a single handoff doc that covers the whole pipeline, send:
- `docs/PIPELINE_QUICKSTART.md`

Phase 1 pipeline:

1) Preprocess structures into `processed_data/` + `nodes_edges/` (with antigen-chain autodetect + gating).
2) Generate per-residue regression targets from ProteinMPNN: `proteinmpnn_scores.parquet` per PDB.
3) Train Epi4Ab with a regression head (`out_label=1`) against ProteinMPNN targets.

Important: Phase 1 predicts raw ProteinMPNN NLL values (not epitope probability).
So Phase 1 should use `output_activation=identity`.

## What Phase 1 proves (and what it doesn't)

Phase 1 proves:
- The end-to-end regression pipeline works on your graph inputs (preprocess -> labels -> training -> inference).
- Residue alignment (`resId`) is consistent across features and regression targets.
- Training is numerically stable (loss decreases; predictions are not NaN; outputs can be written per residue).

Phase 1 does NOT prove:
- Epitope prediction accuracy.
- Biological relationship between ProteinMPNN NLL and antibody binding.

Phase 1 is a de-risking step and (optionally) a representation pretraining step; Phase 2 is the real evaluation.

## Evaluation note (train/test split)

To avoid misleading train-set metrics, Phase 1 should be run with a PDB-level train/test split.
`slurm/train_phase1_proteinmpnn_regression.sbatch` generates `pdb_list_train.csv` and `pdb_list_test.csv` by default.

## Environment (Leonardo)

- Module: `python/3.11.7`
- Python env: project `venv/`
- Install packages:
  - Minimal spec: `requirements.txt`
  - Recommended for reproducibility: `requirements.lock.txt` (a `pip freeze` snapshot from a working venv)

### What `use_pretrained` means (important)

`use_pretrained` is a training/inference flag parsed from the parameters file (e.g. `parameters_phase1_proteinmpnn_regression.txt`).

What it does in the actual code:
- In `source_code/data_function/data_function.py`, when `use_pretrained` is enabled, the dataloader loads a HuggingFace Transformers protein language model and tokenizer.
- With `pretrained_model=ESM2_t30`, it calls:
  - `AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")`
  - `EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")`
- Those embeddings are then computed at runtime and concatenated into node features (sequence context features).

So yes: it is the pre-trained ESM2 model (via Transformers). It is not ProteinMPNN.

What it is NOT:
- It does not precompute and store per-residue embeddings as files in this pipeline.
- It does not “cache embeddings”. It caches the model weights/tokenizer files.

### Freeze vs fine-tune (backprop) (important)

There are two modes when `use_pretrained` is enabled:

1) Frozen pretrained (default)
- This is the default behavior unless you explicitly disable freezing.
- The ESM2 embeddings are computed and then detached from the computation graph.
- Result: the GNN trains, but **no gradients flow into ESM2**.

2) Fine-tune pretrained (unfrozen)
- If you disable freezing, ESM2 runs inside the model forward pass.
- Result: **gradients can flow into ESM2** and its weights can update.
- This is slower and uses more GPU memory.

How to control it:
- The CLI flag is `--freeze_pretrained` (note: it uses `store_false`).
- If you do nothing, you get frozen embeddings.
- If you pass `--freeze_pretrained`, you switch to fine-tuning.

### Offline/cache behavior (important on Leonardo)

Leonardo compute nodes are offline. If you enable `use_pretrained` and the ESM2 weights are not already present in a shared cache, training will fail.

Cache here means the local HuggingFace model cache directory that stores downloaded model files.

Recommended offline env vars for compute nodes:
- `TRANSFORMERS_OFFLINE=1`
- `HF_HUB_OFFLINE=1`

Recommended cache location (shared filesystem):
- set `HF_HOME` to a path on `/leonardo_work/...` (or another shared path your team uses)

Example (pick a shared folder):

```bash
export HF_HOME=/leonardo_work/EUHPC_D29_035/hf_cache
```

To pre-warm the cache (run on a login node with internet, if available):

```bash
source venv/bin/activate
export HF_HOME=/leonardo_work/EUHPC_D29_035/hf_cache
python - <<'PY'
from transformers import AutoTokenizer, EsmModel
name = "facebook/esm2_t30_150M_UR50D"
AutoTokenizer.from_pretrained(name)
EsmModel.from_pretrained(name)
print("cached:", name)
PY
```

If you cannot (or do not want to) use pretrained embeddings for Phase 1, remove `use_pretrained` from the parameters file (or set up a separate parameters file without it).

## Step 0: Preprocessing (gated)

Run:

```bash
METADATA_FILE=/leonardo_work/EUHPC_D29_035/Epi4Ab/input/pdb_info_test3A.csv \
SRC_PROCESSED_DIR=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/processed_data \
sbatch slurm/preprocess_full_autodetect.sbatch
```

Outputs:
- `OUT_BASE=.../upstream_preprocess_autodetect/<RUN_TAG>` printed in the job `.out`
- Gating artifacts:
  - `OUT_BASE/gates/preprocess/metadata.ok.preprocess.csv`
  - `OUT_BASE/gates/nodes_edges/metadata.ok.nodes_edges.csv`
  - `OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv`

## Step 1: ProteinMPNN targets (gated)

Run:

```bash
OUT_BASE=<paste OUT_BASE from preprocessing logs>
sbatch slurm/proteinmpnn_scores_full_autodetect.sbatch
```

This:
- autogenerates `OUT_BASE/proteinmpnn_manifest_autogen.csv` using `metadata.ok.fill_edge.csv` when available
- writes `proteinmpnn_scores.parquet` into each `OUT_BASE/nodes_edges/<pdb_id>/`
- runs `scripts/validate_pipeline.py --step labels` for the ProteinMPNN target files

## Step 2: Phase 1 regression training

Run:

```bash
OUT_BASE=<same OUT_BASE>
sbatch slurm/train_phase1_proteinmpnn_regression.sbatch
```

Defaults:
- Parameters file: `parameters_phase1_proteinmpnn_regression.txt`
- Uses only gated PDBs: `OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv`
- Output: `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/<RUN_TAG>/`

Useful overrides:

```bash
OUT_BASE=... \
RUN_TAG=my_run \
OUT_DIR=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/my_run \
PARAMETERS_FILE=/leonardo_work/EUHPC_D29_035/Epi4Ab/parameters_phase1_proteinmpnn_regression.txt \
sbatch slurm/train_phase1_proteinmpnn_regression.sbatch
```

## What other docs are for (do you need them?)

- `docs/PHASE1_REGRESSION_PIPELINE.md` (this file): the Phase 1 deep-dive.
- `docs/PIPELINE_QUICKSTART.md`: the one-file teammate handoff.
- `docs/ENVIRONMENT_DEPENDENCIES.md`: cross-check list of deps against actual imports.
- `requirements.lock.txt`: reproducible install list if a teammate needs to recreate the venv.
- `docs/PIPELINE_DOCUMENTATION.md`: broader pipeline overview (inference, dashboards, legacy flows).
