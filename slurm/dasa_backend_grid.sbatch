#!/bin/bash
#SBATCH --job-name=epi4ab-dasa-grid
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=/leonardo_work/EUHPC_D29_035/Epi4Ab/logs/%x-%A_%a.out
#SBATCH --error=/leonardo_work/EUHPC_D29_035/Epi4Ab/logs/%x-%A_%a.err

# DASA experiment matrix: 3 ASA backends × 3 SPLIT_SEED values (9 array tasks).
#
# Required env:
#   OUT_BASE     — preprocessing root (nodes_edges + processed_data)
#
# Optional:
#   METADATA_OK  — gated PDB list (default: $OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv)
#   TEST_FRACTION — default 0.2 (same for all tasks)
#   DASA_TRAIN_ROOT — parent dir for outputs (default: .../epi4ab/training_dasa_matrix)
#   TRANSFORMERS_OFFLINE, HF_HOME, etc. — forwarded into the regression driver
#
# Task layout: SLURM_ARRAY_TASK_ID in 0..8
#   backend_idx = task / 3  -> dssp (0), freesasa (1), biopython_sr (2)
#   seed_idx    = task % 3  -> 42, 123, 456
#
# Each task sets PARAMETERS_FILE to the matching Phase-2 DASA preset (target_file * inside*
# the param file matches the label parquet for that backend). Ensure labels exist first:
#   run_dasa_backend_study.sh with RUN_LABEL_CMD=1, or three generate_labels_dasa jobs.

set -euo pipefail

: "${OUT_BASE:?Set OUT_BASE to preprocessing output (e.g. .../upstream_preprocess_autodetect/<RUN>)}"
: "${SLURM_ARRAY_TASK_ID:?This script must be submitted as a SLURM array job (0-8)}"

PROJ=/leonardo_work/EUHPC_D29_035/Epi4Ab
cd "$PROJ"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
if [ -n "${HF_HOME:-}" ]; then
  export HF_HOME
fi

METADATA_OK="${METADATA_OK:-$OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
SPLIT_MODE="${SPLIT_MODE:-antigen_group}"
export TEST_FRACTION
export SPLIT_MODE

DASA_TRAIN_ROOT="${DASA_TRAIN_ROOT:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_dasa_matrix_loao_antigen}"

TASK=${SLURM_ARRAY_TASK_ID}
BI=$((TASK / 3))
SI=$((TASK % 3))

BACKENDS=(dssp freesasa biopython_sr)
PARAM_FILES=(
  parameters_phase2_dasa_regression.txt
  parameters_phase2_dasa_freesasa.txt
  parameters_phase2_dasa_biopython_sr.txt
)
SEEDS=(42 123 456)

backend="${BACKENDS[$BI]}"
param_rel="${PARAM_FILES[$BI]}"
SPLIT_SEED="${SEEDS[$SI]}"

export OUT_BASE
export METADATA_OK
export SPLIT_SEED
export PARAMETERS_FILE="$PROJ/${param_rel}"
export OUT_DIR="${DASA_TRAIN_ROOT}/${backend}_seed${SPLIT_SEED}"

mkdir -p "$DASA_TRAIN_ROOT"

echo "=========================================="
echo "DASA backend grid  task=${TASK}/${SLURM_ARRAY_TASK_COUNT:-9}"
echo "  backend=${backend}  SPLIT_SEED=${SPLIT_SEED}"
echo "  PARAMETERS_FILE=${PARAMETERS_FILE}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  METADATA_OK=${METADATA_OK}"
echo "=========================================="

# Reuse the Phase-1/2 regression driver; #SBATCH lines are no-ops under bash.
bash "$PROJ/slurm/train_phase1_proteinmpnn_regression.sbatch"

echo "DONE dasa-grid task=${TASK} OUT_DIR=${OUT_DIR}"
