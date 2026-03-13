#!/bin/bash
# Phase 2 only; expects node_label_seqitope.parquet under nodes_edges/<pdb_id>.
# Offline safety: export TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 (optional HF_HOME).
#SBATCH --job-name=seqitope-gen-labels
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
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
cd "$PROJ"

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "Seqitope Ground Truth Label Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Directories
OUTPUT_BASE=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/seqitope_ground_truth/nodes_edges
RAW_TRUTH_CSV=/leonardo_work/EUHPC_D29_035/Epi4Ab/raw_ground_truth.csv

echo "OUTPUT_BASE: $OUTPUT_BASE"
echo "RAW_TRUTH_CSV: $RAW_TRUTH_CSV"
echo ""

# Create base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE"

# Note: node_feature.parquet files must exist in each output directory
# before running label generation. These should be generated from antigen
# structures during preprocessing.

# Process Ty1_RBD
echo "=========================================="
echo "Processing: Ty1_RBD"
echo "Started: $(date)"
echo "=========================================="
output_dir="$OUTPUT_BASE/Ty1_RBD"
mkdir -p "$output_dir"

if python scripts/generate_labels.py \
    --raw_ground_truth "$RAW_TRUTH_CSV" \
    --output_dir "$output_dir" \
    --pdb_id Ty1_RBD; then
    echo "  ✓ Ty1_RBD: Success"
else
    echo "  ✗ Ty1_RBD: Failed"
fi
echo ""

# Process mAb159_RBD
echo "=========================================="
echo "Processing: mAb159_RBD"
echo "Started: $(date)"
echo "=========================================="
output_dir="$OUTPUT_BASE/mAb159_RBD"
mkdir -p "$output_dir"

if python scripts/generate_labels.py \
    --raw_ground_truth "$RAW_TRUTH_CSV" \
    --output_dir "$output_dir" \
    --pdb_id mAb159_RBD; then
    echo "  ✓ mAb159_RBD: Success"
else
    echo "  ✗ mAb159_RBD: Failed"
fi
echo ""

# Process mAb311_RBD
echo "=========================================="
echo "Processing: mAb311_RBD"
echo "Started: $(date)"
echo "=========================================="
output_dir="$OUTPUT_BASE/mAb311_RBD"
mkdir -p "$output_dir"

if python scripts/generate_labels.py \
    --raw_ground_truth "$RAW_TRUTH_CSV" \
    --output_dir "$output_dir" \
    --pdb_id mAb311_RBD; then
    echo "  ✓ mAb311_RBD: Success"
else
    echo "  ✗ mAb311_RBD: Failed"
fi
echo ""

echo "=========================================="
echo "Label generation complete"
echo "Finished: $(date)"
echo "=========================================="

# Verify results
echo ""
echo "Verification: Checking generated labels"
echo "=========================================="
for complex_id in Ty1_RBD mAb159_RBD mAb311_RBD; do
    label_file="$OUTPUT_BASE/$complex_id/node_label_pi.parquet"
    
    if [ -f "$label_file" ]; then
        echo "  ✓ $complex_id: node_label_pi.parquet exists"
    else
        echo "  ✗ $complex_id: MISSING node_label_pi.parquet"
    fi
done

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi

