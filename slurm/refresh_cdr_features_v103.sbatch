#!/bin/bash
#SBATCH --job-name=epi4ab-cdr-refresh-v103
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/leonardo_work/EUHPC_D29_035/Epi4Ab/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_035/Epi4Ab/logs/%x-%j.err

# Refresh CDR distance features (v1.0.3) on existing preprocessed data.
# Does NOT re-run pdb2pqr, angles, or PyMOL — only CDR distances + node features.
#
# Required:
#   OUT_BASE — root of the existing preprocessing run
#              e.g. /leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess_autodetect/20260305_143201

set -euo pipefail

: "${OUT_BASE:?Set OUT_BASE to the preprocessing run root}"

PROJ=/leonardo_work/EUHPC_D29_035/Epi4Ab
cd "$PROJ"

module purge
module load python/3.11.7

# ANARCI (installed in venv) requires HMMER (hmmscan) from conda
export PATH="/leonardo_work/AIFAC_F01_302/Epi4Ab/tools/conda/bin:$PATH"

if [ -d "venv" ]; then
  source venv/bin/activate
fi

METADATA_OK="${METADATA_OK:-$OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv}"
PROCESSED_DIR="$OUT_BASE/processed_data"
NODES_EDGES_DIR="$OUT_BASE/nodes_edges"

echo "=========================================="
echo "CDR feature refresh  (feature_version v1.0.3)"
echo "Job ID:       $SLURM_JOB_ID"
echo "Started:      $(date)"
echo "OUT_BASE:     $OUT_BASE"
echo "METADATA_OK:  $METADATA_OK"
echo "PROCESSED:    $PROCESSED_DIR"
echo "NODES_EDGES:  $NODES_EDGES_DIR"
echo "=========================================="

python scripts/refresh_cdr_features.py \
    --metadata    "$METADATA_OK" \
    --processed   "$PROCESSED_DIR" \
    --nodes_edges "$NODES_EDGES_DIR"

echo "DONE: $(date)"
