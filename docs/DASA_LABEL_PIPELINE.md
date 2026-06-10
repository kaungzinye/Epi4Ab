# Delta-ASA Label Pipeline

This document describes how to generate Delta-ASA labels for antigen residues using a
swappable ASA backend (`dssp`, `freesasa`, or `biopython_sr`). Downstream parquet schema
and alignment rules are identical for all engines.

## Output contract

For each PDB ID under `nodes_edges/<pdb_id>/`, the pipeline writes:

- `node_label_dasa.parquet` with columns:
  - `resId` (int, aligned to `node_feature.parquet`)
  - `score` (float in `[0,1]`)
  - `asa_alone` (float)
  - `asa_complex` (float)
  - `delta_asa` (float)
  - `delta_asa_clipped` (float, `max(delta_asa, 0)`)

Label math:

- `delta_asa = asa_alone - asa_complex`
- `delta_asa_clipped = max(delta_asa, 0)`
- `score = minmax(delta_asa_clipped)` per complex

## Inputs

- `--nodes_edges_dir` (contains `nodes_edges/<pdb_id>/node_feature.parquet`)
- `--processed_dir` (contains `processed_data/<pdb_id>/lig.pdb` and full-complex structure)
- `--metadata` CSV with at least:
  - `pdbID`
  - `pdb`
  - `antigen`

## ASA backend (`--asa_backend`)

- **`dssp`** (default): absolute residue ASA from DSSP (`mkdssp`). Same behavior as the original pipeline.
- **`freesasa`**: absolute per-residue SASA from the Python **`freesasa`** package (`pip install freesasa`). No DSSP binary is used.
- **`biopython_sr`**: Shrake–Rupley residue SASA via **`Bio.PDB.SASA.ShrakeRupley`** (already covered by `biopython` in `requirements.lock.txt`). No extra binary.

**Important:** `asa_alone`, `asa_complex`, and raw `delta_asa` values are **not directly comparable**
across backends (different surface models and radii). The script still uses the same definition
`delta_asa = asa_alone - asa_complex` and the same per-complex min–max normalization for `score`,
so training/eval interfaces stay stable.

Use `--output_file` to pick a filename when comparing backends side by side, e.g.
`node_label_dasa_freesasa.parquet`, `node_label_dasa_biopython_sr.parquet`, or the default `node_label_dasa.parquet` for DSSP.

Parameter presets for training: `parameters_phase2_dasa_regression.txt` (DSSP/default name),
`parameters_phase2_dasa_freesasa.txt`, `parameters_phase2_dasa_biopython_sr.txt`.

Study driver (prints `sbatch` lines): `scripts/run_dasa_backend_study.sh`.

**IEDB epitope prior (optional, for dashboards / benchmarking):** `scripts/build_iedb_epitope_prior.py` reads IEDB bulk CSV + PDBe `mappings/uniprot/{pdb}` JSON (cached via `--sifts_cache_dir`), aligns binary `epitope_prior` to `node_feature.resId`. Use `--struct_fallback` and `--infer_uniprot` on unreliable metadata; see script docstring. For IQ-API harvests that include discontinuous B-cell epitopes, run `scripts/epitope_ground_truth/fetch_iedb_epitopes_cohort.py` first. For **PDB-scoped** IEDB rows (`complex__pdb_id`), run `scripts/epitope_ground_truth/fetch_iedb_bcell_pdb_scoped.py` and rebuild with `--pdb_scope` (output e.g. `node_iedb_pdb_epitope_prior.parquet`). **CoV-AbDAb** priors: `scripts/epitope_ground_truth/build_covabdab_epitope_prior.py`. **Augmented cohort PDB list** (CoV ∪ IEDB-PDB codes ∩ your complexes): `scripts/epitope_ground_truth/merge_augmented_epitope_pdb_list.py`. Tier index: `docs/EPITOPE_GROUND_TRUTH_TIERS.md`.

**Literature-subset AUROC (optional):** `scripts/eval_dasa_vs_literature_subset.py` filters a manifest and writes per-PDB and per-run mean±std AUROC. Use `--truth_tier` (`iedb_uniprot`, `iedb_pdb`, `covabdab`) or explicit `--manifest_prior_source_col` / `--manifest_prior_values`. The `runs.csv` `epitope_prior_file` column must name the parquet that matches that manifest (same columns as `visualize_dasa_backend_comparison.py`).

**Comparison dashboard (optional):** after inference, build a small `runs.csv` (see `scripts/visualize_dasa_backend_comparison.py` docstring) and run that script for cohort Pearson and AUROC vs IEDB prior.

### DSSP requirement (only for `--asa_backend dssp`)

You need a DSSP binary, usually `mkdssp`.

Either:

- put it on PATH and use default `--dssp_bin mkdssp`, or
- pass an explicit path, e.g. `--dssp_bin /path/to/mkdssp`

Some **mmCIF** inputs make `mkdssp` fail (parser/CLI quirks). The script then writes a **temporary PDB** via Biopython and runs DSSP on that file. This requires `biopython` in the same environment as the script.

For **FreeSASA**, mmCIF inputs are also converted to a temporary PDB the same way (Biopython required).

## Command line

Single script:

```bash
python scripts/generate_dasa_labels.py \
  --nodes_edges_dir <OUT_BASE>/nodes_edges \
  --processed_dir <OUT_BASE>/processed_data \
  --metadata <metadata.csv> \
  --pdb_list <OUT_BASE>/gates/fill_edge/metadata.ok.fill_edge.csv \
  --asa_backend dssp \
  --dssp_bin mkdssp
```

FreeSASA example (after `pip install freesasa`):

```bash
python scripts/generate_dasa_labels.py \
  --nodes_edges_dir <OUT_BASE>/nodes_edges \
  --processed_dir <OUT_BASE>/processed_data \
  --metadata <metadata.csv> \
  --pdb_list <OUT_BASE>/gates/fill_edge/metadata.ok.fill_edge.csv \
  --asa_backend freesasa \
  --output_file node_label_dasa_freesasa.parquet
```

Optional flags:

- `--strict_residue_mapping` — disable sequence/window fallback; use only explicit `(chain, resseq, insertion)` keys from the ASA maps for **both** antigen-only and complex structures.
- `--fallback_min_aa_fraction` (default `0.95`) — minimum fraction of matching (non-`X`) amino acids when fallback aligns `lig.pdb` sequence to ASA residue order (longer chain slides a window; if the complex chain is **shorter** than the antigen list, the script aligns a contiguous substring and **carries** the nearest ASA key for at most a few terminal positions so every graph row still gets a value).

For batch jobs, set `ASA_BACKEND` and optionally `OUTPUT_FILE` (see SLURM section below). Set `DSSP_BIN` if `mkdssp` is not on the default `PATH` when using `dssp`.

SLURM wrapper:

```bash
OUT_BASE=<.../upstream_preprocess_autodetect/<RUN_TAG>>
METADATA_FILE=<metadata csv used in preprocess>
PDB_LIST=<OUT_BASE>/gates/fill_edge/metadata.ok.fill_edge.csv
# optional: ASA_BACKEND=freesasa OUTPUT_FILE=node_label_dasa_freesasa.parquet
sbatch slurm/generate_labels_dasa.sbatch
```

## Alignment and failure behavior

The generator is strict:

- output row count must equal `node_feature.parquet` row count
- output `resId` order must match `node_feature.resId` exactly
- if ASA residue mapping fails for a residue, that PDB is marked failed and skipped

By default, when **PDB residue numbers** disagree between `lig.pdb` and the full complex (or the ASA engine drops one row vs the PDB), an optional **sequence-based** alignment selects the ASA rows; use `--strict_residue_mapping` to forbid that path for ablations or auditing.

Failures are logged at:

- `processed_data/<pdb_id>/errors/generate_dasa_labels.log`

Batch jobs continue to the next PDB unless `--fail_fast` is provided.
#### Retrying a subset

Use repeated `--pdb_id` or a small CSV with a `pdbID` column:

```bash
python scripts/generate_dasa_labels.py \
  --nodes_edges_dir "$OUT_BASE/nodes_edges" \
  --processed_dir "$OUT_BASE/processed_data" \
  --metadata "$OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv" \
  --asa_backend "${ASA_BACKEND:-dssp}" \
  --dssp_bin "${DSSP_BIN:-mkdssp}" \
  --pdb_id 8vtd_BAC --pdb_id 8zca_DAF
```

## Validation

```bash
python scripts/validate_pipeline.py \
  --metadata <pdb_list.csv> \
  --nodes_edges_dir <OUT_BASE>/nodes_edges \
  --step labels \
  --target_type seqitope \
  --target_file node_label_dasa.parquet \
  --target_column score
```

Use the same `--target_file` if you generated labels under a backend-specific name (e.g. `node_label_dasa_freesasa.parquet`).

## Visual summary (Plotly HTML)

After labels exist, build a dashboard (summary table, histograms, sample per-residue `score` traces):

```bash
python scripts/visualize_dasa_labels.py \
  --nodes_edges_dir <OUT_BASE>/nodes_edges \
  --metadata <OUT_BASE>/gates/fill_edge/metadata.ok.fill_edge.csv \
  --build-index
```

Default output: `epi4ab/plots/cohort/<OUT_BASE_TAG>/html/dasa_label_dashboard.html`. Open `epi4ab/plots/index.html` to browse. Use `--sample_detailed N` to change how many complexes get full sequence plots (default 12).

IEDB/CoV-AbDab bulk CSVs and PDBe cache: `<OUT_BASE>/cache/` (not `reports/`). See [STORAGE_AND_DRIVES.md](STORAGE_AND_DRIVES.md).

## Training and PDB-level train/test split

Training is **by PDB** (whole complexes), not by individual residues. The SLURM wrapper `slurm/train_phase1_proteinmpnn_regression.sbatch` reads `METADATA_OK`, then writes `pdb_list_train.csv` and `pdb_list_test.csv` under `OUT_DIR`.

Reasonable default used in this project: **20% of PDBs for test**, **80% for train**, with a fixed seed so the split reproducible:

- `TEST_FRACTION=0.2`
- `SPLIT_SEED=42`

DASA-specific model args live in `parameters_phase2_dasa_regression.txt` (`target_file=node_label_dasa.parquet`, `output_activation=sigmoid`).

Example:

```bash
cd /leonardo_work/EUHPC_D29_035/Epi4Ab
OUT_BASE=<.../upstream_preprocess_autodetect/<RUN_TAG>>
TEST_FRACTION=0.2 SPLIT_SEED=42 \
PARAMETERS_FILE=/leonardo_work/EUHPC_D29_035/Epi4Ab/parameters_phase2_dasa_regression.txt \
METADATA_OK="$OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv" \
sbatch slurm/train_phase1_proteinmpnn_regression.sbatch
```

Ensure every `pdbID` in `METADATA_OK` has `nodes_edges/<pdbID>/node_label_dasa.parquet` before training.

## Pilot sanity-check checklist

Run on a small pilot subset first (for example 3 complexes):

1. one clean single-chain antigen
2. one case with residue numbering irregularities (insertions / gaps)
3. one case where antigen chain autodetect was needed

For each pilot PDB:

- label file exists and validates
- score range is not degenerate (`score` not all identical)
- top `delta_asa` residues overlap expected contact region
- no silent mapping drift (check `resId` alignment)
