#!/bin/bash
#SBATCH --job-name=epi4ab-dasa-sr-v103
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/leonardo_work/EUHPC_D29_035/Epi4Ab/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_035/Epi4Ab/logs/%x-%j.err

# Train DASA with v1.0.3 features (CDR distances + mse_pearson loss).
#
# Required:
#   OUT_BASE   — same preprocessing run used for refresh_cdr_features_v103
# Optional:
#   SPLIT_SEED — default 42
#   OUT_DIR    — output directory for this training run

set -euo pipefail

: "${OUT_BASE:?Set OUT_BASE to the preprocessing run root}"

PROJ=/leonardo_work/EUHPC_D29_035/Epi4Ab
cd "$PROJ"

export PATH="/leonardo_work/AIFAC_F01_302/Epi4Ab/tools/conda/bin:$PATH"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi

export OUT_BASE
export METADATA_OK="${METADATA_OK:-$OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv}"
export SPLIT_SEED="${SPLIT_SEED:-42}"
export TEST_FRACTION="${TEST_FRACTION:-0.2}"
export SPLIT_MODE="${SPLIT_MODE:-antigen_group}"
export PARAMETERS_FILE="$PROJ/parameters_phase2_dasa_biopython_sr_v103.txt"
export OUT_DIR="${OUT_DIR:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_dasa_v103/biopython_sr_seed${SPLIT_SEED}}"

echo "=========================================="
echo "DASA v1.0.3 training"
echo "Job ID:      $SLURM_JOB_ID"
echo "Started:     $(date)"
echo "OUT_BASE:    $OUT_BASE"
echo "PARAMS:      $PARAMETERS_FILE"
echo "SPLIT_SEED:  $SPLIT_SEED"
echo "OUT_DIR:     $OUT_DIR"
echo "=========================================="

bash "$PROJ/slurm/train_phase1_proteinmpnn_regression.sbatch"

echo "DONE: $(date)"
