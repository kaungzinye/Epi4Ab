#!/usr/bin/env bash
# Print (and optionally run) commands for the DASA ASA-backend × seed study.
#
# Required env:
#   OUT_BASE         — preprocessing root with nodes_edges/ and processed_data/
#   METADATA_FILE    — CSV with pdbID,pdb,antigen (optional antigen_uniprot for IEDB prior)
#   PDB_LIST         — CSV with pdbID column
#
# Optional:
#   RUN_LABEL_CMD=1   — if set, sbatch generate_labels_dasa for each backend
#   RUN_TRAIN_CMD=1   — if set, sbatch one job per (backend × seed) via train_phase1 (9 jobs)
#   RUN_TRAIN_GRID=1  — if set, sbatch a single array job: slurm/dasa_backend_grid.sbatch (recommended)
#   SEEDS="42 123 456" — only used when RUN_TRAIN_CMD=1 (must match dasa_backend_grid.sbatch)
#   DASA_TRAIN_ROOT   — base dir for training outputs (grid + per-job OUT_DIR)
#
# Label outputs:
#   dssp           -> node_label_dasa.parquet              + parameters_phase2_dasa_regression.txt
#   freesasa       -> node_label_dasa_freesasa.parquet      + parameters_phase2_dasa_freesasa.txt
#   biopython_sr   -> node_label_dasa_biopython_sr.parquet  + parameters_phase2_dasa_biopython_sr.txt
#
# After labels + training + inference, build runs_csv for visualize_dasa_backend_comparison.py

set -euo pipefail

: "${OUT_BASE:?}"
: "${METADATA_FILE:?}"
: "${PDB_LIST:?}"

PROJ="${PROJ:-/leonardo_work/EUHPC_D29_035/Epi4Ab}"
SEEDS="${SEEDS:-42 123 456}"
RUN_LABEL_CMD="${RUN_LABEL_CMD:-0}"
RUN_TRAIN_CMD="${RUN_TRAIN_CMD:-0}"
RUN_TRAIN_GRID="${RUN_TRAIN_GRID:-0}"
DASA_TRAIN_ROOT="${DASA_TRAIN_ROOT:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_dasa_matrix}"
METADATA_OK="${METADATA_OK:-$PDB_LIST}"

echo "# === 1) Generate DASA labels per ASA backend ==="
for spec in \
  "dssp:node_label_dasa.parquet" \
  "freesasa:node_label_dasa_freesasa.parquet" \
  "biopython_sr:node_label_dasa_biopython_sr.parquet"; do
  backend="${spec%%:*}"
  outf="${spec##*:}"
  cmd="OUT_BASE=\"$OUT_BASE\" METADATA_FILE=\"$METADATA_FILE\" PDB_LIST=\"$PDB_LIST\" ASA_BACKEND=\"$backend\" OUTPUT_FILE=\"$outf\" sbatch \"$PROJ/slurm/generate_labels_dasa.sbatch\""
  echo "$cmd"
  if [[ "$RUN_LABEL_CMD" == "1" ]]; then
    eval "$cmd"
  fi
done

echo ""
echo "# === 2) Train Phase-2 DASA regression (3 seeds × 3 backends) ==="
echo "#     METADATA_OK=${METADATA_OK}  DASA_TRAIN_ROOT=${DASA_TRAIN_ROOT}"

grid_cmd="OUT_BASE=\"$OUT_BASE\" METADATA_OK=\"$METADATA_OK\" TEST_FRACTION=\"${TEST_FRACTION:-0.2}\" DASA_TRAIN_ROOT=\"$DASA_TRAIN_ROOT\" sbatch \"$PROJ/slurm/dasa_backend_grid.sbatch\""
echo "# --- One array job (9 tasks), recommended; logs: logs/epi4ab-dasa-grid-<jobid>_<task>.out"
echo "$grid_cmd"
if [[ "$RUN_TRAIN_GRID" == "1" ]]; then
  eval "$grid_cmd"
fi

for spec in \
  "dssp:node_label_dasa.parquet:parameters_phase2_dasa_regression.txt" \
  "freesasa:node_label_dasa_freesasa.parquet:parameters_phase2_dasa_freesasa.txt" \
  "biopython_sr:node_label_dasa_biopython_sr.parquet:parameters_phase2_dasa_biopython_sr.txt"; do
  backend="${spec%%:*}"
  rest="${spec#*:}"
  outf="${rest%%:*}"
  param="${rest##*:}"
  for s in $SEEDS; do
    out_dir="${DASA_TRAIN_ROOT}/${backend}_seed${s}"
    cmd="OUT_BASE=\"$OUT_BASE\" METADATA_OK=\"$METADATA_OK\" PARAMETERS_FILE=\"$PROJ/$param\" SPLIT_SEED=\"$s\" OUT_DIR=\"$out_dir\" sbatch \"$PROJ/slurm/train_phase1_proteinmpnn_regression.sbatch\""
    echo "$cmd"
    if [[ "$RUN_TRAIN_CMD" == "1" ]]; then
      eval "$cmd"
    fi
  done
done

echo ""
echo "# === 3) Inference — set MODEL_DIR to each training output containing model.pt ==="
echo "# Example:"
echo "# OUT_BASE=\"$OUT_BASE\" METADATA_OK=\"$METADATA_OK\" MODEL_DIR=\".../training_dasa_dssp_seed42/<date>_GNNResNet_1\" OUT_DIR=\".../infer_dssp_s42\" sbatch \"$PROJ/slurm/inference_phase1_proteinmpnn_regression.sbatch\""

echo ""
echo "# === 4) IEDB epitope prior (optional) — IQ-API multitype harvest, then build: ==="
PLOTS_ROOT="${EPI4AB_PLOTS_ROOT:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/plots}"
OUT_TAG="${EPI4AB_OUT_BASE_TAG:-$(basename "$OUT_BASE")}"
COHORT_CSV="$PLOTS_ROOT/cohort/$OUT_TAG/csv"

echo "# python \"$PROJ/scripts/epitope_ground_truth/fetch_iedb_epitopes_cohort.py\" \\"
echo "#   --metadata \"$METADATA_FILE\" --out \"$OUT_BASE/cache/iedb/iedb_epitope_multitype.csv\""
echo "# python \"$PROJ/scripts/build_iedb_epitope_prior.py\" \\"
echo "#   --metadata \"$METADATA_FILE\" --iedb_csv \"$OUT_BASE/cache/iedb/iedb_epitope_multitype.csv\" \\"
echo "#   --nodes_edges_dir \"$OUT_BASE/nodes_edges\" --processed_dir \"$OUT_BASE/processed_data\" \\"
echo "#   --pdb_list \"$PDB_LIST\" --sifts_cache_dir \"$OUT_BASE/cache/pdbe\" \\"
echo "#   --struct_fallback --infer_uniprot --manifest_out \"$OUT_BASE/cache/iedb/iedb_prior_manifest.csv\""
echo "# Literature-only AUROC: python \"$PROJ/scripts/eval_dasa_vs_literature_subset.py\" \\"
echo "#   --manifest \"$OUT_BASE/cache/iedb/iedb_prior_manifest.csv\" \\"
echo "#   --runs_csv \"$COHORT_CSV/runs_canonical_cohort.csv\" --score_col true_score --run_id <RUN_ID>"

echo ""
echo "# === 5) Dashboards (plots hub) ==="
echo "# python \"$PROJ/scripts/visualize_dasa_labels.py\" --nodes_edges_dir \"$OUT_BASE/nodes_edges\" \\"
echo "#   --metadata \"$PDB_LIST\" --label_file node_label_dasa_biopython_sr.parquet \\"
echo "#   --epitope_prior_file node_iedb_epitope_prior.parquet --build-index"
echo "# Build runs_csv then:"
echo "# python \"$PROJ/scripts/visualize_dasa_backend_comparison.py\" \\"
echo "#   --runs_csv \"$COHORT_CSV/runs_canonical_cohort.csv\" --build-index"
echo "# Master index: $PLOTS_ROOT/index.html"
