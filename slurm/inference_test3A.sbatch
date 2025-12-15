#!/bin/bash
#SBATCH --job-name=epi4ab-inference-test3A
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/%x-%j.err

set -euo pipefail

module purge
module load python/3.11.7

PROJ=/leonardo_work/AIFAC_F01_302/Epi4Ab
cd "$PROJ"

# Use the virtualenv for dependencies
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Use inference list for test3A (create if doesn't exist)
INFERENCE_LIST="$PROJ/inference_list_test3A.csv"

if [ ! -f "$INFERENCE_LIST" ]; then
    echo "Creating inference list from metadata..."
    # Use metadata_complete.csv for 12 complete PDBs
    METADATA_FILE="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/upstream_preprocess/metadata_complete.csv"
    head -1 "$METADATA_FILE" | sed 's/pdbID/pdbId/' > "$INFERENCE_LIST"
    tail -n +2 "$METADATA_FILE" | cut -d',' -f1 >> "$INFERENCE_LIST"
fi

echo "=========================================="
echo "Epi4Ab Inference - Test3A Dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "NODES_EDGES: $DIRECTORY_NODES_EDGES"
echo "PROCESSED_DATA: $DIRECTORY_PROCESSED_DATA"
echo "INFERENCE_LIST: $INFERENCE_LIST"
echo "MODEL: $FINAL_MODEL_FOLDER"
echo "OUTPUT: $DIRECTORY_INFERENCE_OUTPUT"
echo "=========================================="

# Count PDBs
TOTAL_PDBS=$(tail -n +2 "$INFERENCE_LIST" | wc -l)
echo "Total PDBs for inference: $TOTAL_PDBS"
echo ""

# Ensure output directory exists
mkdir -p "$DIRECTORY_INFERENCE_OUTPUT"
mkdir -p logs

# Set OMP threads for performance
export OMP_NUM_THREADS=4

# Run inference
echo "=========================================="
echo "Running inference"
echo "Started: $(date)"
echo "=========================================="
./venv/bin/python epi_prediction.py \
    "$DIRECTORY_NODES_EDGES" \
    "$DIRECTORY_PROCESSED_DATA" \
    "$INFERENCE_LIST" \
    "$FINAL_MODEL_FOLDER" \
    "$DIRECTORY_INFERENCE_OUTPUT"

# Find the output directory (it has a timestamp)
OUTPUT_DIR=$(ls -td "$DIRECTORY_INFERENCE_OUTPUT"/20* 2>/dev/null | head -1)

if [ -n "$OUTPUT_DIR" ]; then
    echo ""
    echo "=========================================="
    echo "Inference complete"
    echo "Finished: $(date)"
    echo "=========================================="
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Verify results:"
    if [ -d "$OUTPUT_DIR/test_record" ]; then
        RESULT_COUNT=$(ls "$OUTPUT_DIR/test_record/"*_final_result.txt 2>/dev/null | wc -l)
        echo "  Results generated: $RESULT_COUNT / $TOTAL_PDBS PDBs"
        ls -1 "$OUTPUT_DIR/test_record/"*_final_result.txt 2>/dev/null | sed 's/.*\//  ✓ /' | sed 's/_final_result.txt//'
    fi
    echo ""
    echo "Next step: Generate dashboard on login node"
    echo "  cd $PROJ"
    echo "  ./scripts/generate_dashboard.sh $OUTPUT_DIR/test_record"
    echo "=========================================="
else
    echo "ERROR: Output directory not found"
    exit 1
fi

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi
