#!/bin/bash
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=epi4ab-validate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/epi4ab-validate-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/epi4ab-validate-%j.err

# Comprehensive Validation Job for Epi4Ab Processed Data
# 
# This validates ALL processed structures for:
# - Complete column set (50 columns including VH/VK/VLa families)
# - Data integrity (distances, angles, sequences)
# - Model compatibility
# - File completeness
#
# Run this BEFORE inference to catch any issues early!

set -e

echo "=========================================="
echo "Epi4Ab: Comprehensive Data Validation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load python/3.11.7

# Activate virtual environment
source /leonardo_work/AIFAC_F01_302/Epi4Ab/venv/bin/activate

# Directories
WORK_DIR="/leonardo_work/AIFAC_F01_302/Epi4Ab"
PROCESSED_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed"
MODEL_LOG="$WORK_DIR/final_trained_Epi4Ab/log.json"
RESULTS_FILE="$WORK_DIR/validation_results_$(date +%Y%m%d_%H%M%S).json"

cd $WORK_DIR

echo "Validating processed data..."
echo "Processed directory: $PROCESSED_DIR"
echo "Model log: $MODEL_LOG"
echo ""

# Run comprehensive validation
python validate_all.py \
    --processed_dir "$PROCESSED_DIR" \
    --model_log "$MODEL_LOG" \
    --verbose \
    --json_output "$RESULTS_FILE"

VALIDATION_STATUS=$?

echo ""
echo "=========================================="
echo "Validation Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Results saved to: $RESULTS_FILE"

if [ $VALIDATION_STATUS -eq 0 ]; then
    echo ""
    echo "✅✅✅ ALL VALIDATIONS PASSED ✅✅✅"
    echo ""
    echo "Next step: Run inference"
    echo "  sbatch slurm/infer_all.sbatch"
else
    echo ""
    echo "❌❌❌ VALIDATION FAILED ❌❌❌"
    echo ""
    echo "Check the output above for details."
    echo "Common issues:"
    echo "  - Missing VLa columns (Bug #5) → Reprocess with: sbatch slurm/REPROCESS_ALL.sbatch"
    echo "  - Missing resShort column (Bug #4) → Reprocess with: sbatch slurm/REPROCESS_ALL.sbatch"
    echo "  - Zero distances (Bug #2) → Check processing pipeline"
    echo "  - Missing sequence files (Bug #3) → Reprocess with: sbatch slurm/REPROCESS_ALL.sbatch"
    exit 1
fi

