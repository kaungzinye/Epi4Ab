#!/bin/bash
#SBATCH --job-name=epi4ab-process
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/%x-%j.err

set -euo pipefail
module purge
module load python/3.11.7

PROJ=/leonardo_work/AIFAC_F01_302/Epi4Ab
cd $PROJ
source venv/bin/activate

# Load environment variables
set -a
source .env
set +a

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0}}

echo "=========================================="
echo "Processing HER2 PDB files from SAbDab filtered list"
echo "=========================================="

# Use HER2 test set from SAbDab selection (filtered by criteria: resolution ≤4Å, H+L chains, single-chain antigen)
# Priority: corrected list (excludes 6BGT from training) > filtered list > unique list > paper list
HER2_LIST="$PROJ/sabdab_selection_results/her2_test_set_filtered_corrected.csv"
if [ ! -f "$HER2_LIST" ]; then
    HER2_LIST="$PROJ/sabdab_selection_results/her2_test_set_filtered.csv"
fi
if [ ! -f "$HER2_LIST" ]; then
    HER2_LIST="$PROJ/sabdab_selection_results/her2_test_set_unique.csv"
fi
if [ ! -f "$HER2_LIST" ]; then
    HER2_LIST="$PROJ/sabdab_selection_results/her2_test_set_paper.csv"
fi

if [ ! -f "$HER2_LIST" ]; then
    echo "ERROR: No HER2 list found. Expected one of:"
    echo "  - $PROJ/sabdab_selection_results/her2_test_set_filtered.csv"
    echo "  - $PROJ/sabdab_selection_results/her2_test_set_unique.csv"
    echo "  - $PROJ/sabdab_selection_results/her2_test_set_paper.csv"
    exit 1
fi

echo "Using HER2 list: $HER2_LIST"

# Extract PDB IDs from CSV (first column, handle both with/without header)
# Check if first line looks like a header (contains non-PDB characters)
FIRST_LINE=$(head -1 "$HER2_LIST")
if [[ "$FIRST_LINE" =~ ^[A-Z0-9]{4}$ ]]; then
    # No header, all lines are PDB IDs
    PDB_IDS=$(cat "$HER2_LIST" | cut -d',' -f1 | tr '\n' ' ')
else
    # Has header, skip first line
    PDB_IDS=$(tail -n +2 "$HER2_LIST" | cut -d',' -f1 | tr '\n' ' ')
fi
PDB_COUNT=$(echo "$PDB_IDS" | wc -w)

echo "Found $PDB_COUNT HER2 PDB IDs to process: $PDB_IDS"
echo ""

PDB_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files"
OUTPUT_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed"

successful=0
failed=0
skipped=0

for PDB_ID in $PDB_IDS; do
    echo "=========================================="
    echo "Processing: $PDB_ID"
    echo "=========================================="
    
    # Find PDB file
    PDB_FILE=$(find "$PDB_DIR" -name "${PDB_ID}.pdb" -o -name "${PDB_ID}_*.pdb" | head -1)
    
    if [ -z "$PDB_FILE" ]; then
        echo "⚠️  PDB file not found for $PDB_ID, skipping..."
        skipped=$((skipped + 1))
        continue
    fi
    
    echo "PDB file: $PDB_FILE"
    
    # Create output directory for this PDB
    PDB_OUTPUT_DIR="$OUTPUT_DIR/$PDB_ID"
    mkdir -p "$PDB_OUTPUT_DIR"
    
    # Check if already processed (skip if node_feature.parquet exists)
    if [ -f "$PDB_OUTPUT_DIR/node_feature.parquet" ]; then
        echo "✓ $PDB_ID already processed, skipping..."
        skipped=$((skipped + 1))
        continue
    fi
    
    # Process the PDB
    # --sequence_source auto: Tries FASTA → RCSB API → PDB extraction
    echo "Running processing pipeline..."
    if srun -u python -u data_processing/epi4ab_pipeline.py \
        --pdb_file "$PDB_FILE" \
        --output_dir "$PDB_OUTPUT_DIR" \
        --sequence_source auto \
        --antigen_chain A; then
        echo "✅ $PDB_ID processed successfully"
        successful=$((successful + 1))
    else
        echo "❌ $PDB_ID processing failed"
        failed=$((failed + 1))
    fi
    
    echo ""
done

echo "=========================================="
echo "Processing Complete"
echo "=========================================="
echo "Successful: $successful"
echo "Failed: $failed"
echo "Skipped: $skipped"
echo "Total: $PDB_COUNT"
echo ""

if [ $failed -eq 0 ]; then
    echo "✅ All HER2 PDB files processed successfully!"
else
    echo "⚠️  Some PDB files failed. Check logs for details."
fi

deactivate
