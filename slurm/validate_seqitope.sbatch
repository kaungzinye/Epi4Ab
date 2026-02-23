#!/bin/bash
#SBATCH --job-name=seqitope-validate
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Seqitope-specific paths
NODES_EDGES_DIR="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/seqitope_ground_truth/nodes_edges"
PROCESSED_DIR="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/seqitope_ground_truth/processed_data"
INFERENCE_LIST="$PROJ/inference_list_seqitope.csv"

echo "=========================================="
echo "Seqitope Pipeline Validation"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "NODES_EDGES: $NODES_EDGES_DIR"
echo "PROCESSED_DATA: $PROCESSED_DIR"
echo "=========================================="

# Complexes to validate
COMPLEXES=("Ty1_RBD" "mAb159_RBD" "mAb311_RBD")

echo "Validating ${#COMPLEXES[@]} complexes"
echo ""

# Validate each complex
VALIDATION_PASSED=true
for complex_id in "${COMPLEXES[@]}"; do
    echo "=========================================="
    echo "Validating: $complex_id"
    echo "=========================================="
    
    # Validate all steps
    if ! python scripts/validate_pipeline.py \
        --pdb_id "$complex_id" \
        --nodes_edges_dir "$NODES_EDGES_DIR" \
        --processed_dir "$PROCESSED_DIR" \
        --step all; then
        VALIDATION_PASSED=false
        echo "  ✗ Validation failed for $complex_id"
    else
        echo "  ✓ Validation passed for $complex_id"
    fi
    echo ""
done

# Summary
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
if [ "$VALIDATION_PASSED" = true ]; then
    echo "✓ All validations passed"
    echo ""
    echo "Next step: Run inference"
    echo "  sbatch slurm/inference_seqitope.sbatch"
else
    echo "✗ Some validations failed"
    echo "Please review the validation output above and fix issues before running inference"
    exit 1
fi

echo "=========================================="
echo "Validation complete"
echo "Finished: $(date)"
echo "=========================================="

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi

