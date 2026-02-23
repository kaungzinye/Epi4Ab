#!/bin/bash
#SBATCH --job-name=proteinmpnn-scores
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/leonardo_work/EUHPC_D29_035/Epi4Ab/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_035/Epi4Ab/logs/%x-%j.err

set -euo pipefail

module purge
module load python/3.11.7

PROJ=/leonardo_work/EUHPC_D29_035/Epi4Ab
PROTEINMPNN_DIR="$PROJ/tools/proteinmpnn/ProteinMPNN"
MANIFEST="$PROJ/input/proteinmpnn_manifest.csv"
OUT_ROOT="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/proteinmpnn_scores"
# Where node_feature.parquet lives for your chosen dataset (edit if needed)
NODES_EDGES_DIR="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/nodes_edges"

cd "$PROJ"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

echo "=========================================="
echo "ProteinMPNN Scoring"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Manifest: $MANIFEST"
echo "ProteinMPNN dir: $PROTEINMPNN_DIR"
echo "Out root: $OUT_ROOT"
echo "Nodes/edges dir: $NODES_EDGES_DIR"
echo "=========================================="

mkdir -p "$OUT_ROOT"

python scripts/run_proteinmpnn_score_only_and_prepare_targets.py \
    --manifest "$MANIFEST" \
    --nodes_edges_dir "$NODES_EDGES_DIR" \
    --proteinmpnn_run "$PROTEINMPNN_DIR/protein_mpnn_run.py" \
    --out_root "$OUT_ROOT" \
    --target_file proteinmpnn_scores.parquet \
    --skip_existing

python scripts/validate_pipeline.py \
    --metadata "$MANIFEST" \
    --nodes_edges_dir "$NODES_EDGES_DIR" \
    --step labels \
    --target_type proteinmpnn \
    --target_file proteinmpnn_scores.parquet \
    --target_column score

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi
