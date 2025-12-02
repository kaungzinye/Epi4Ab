#!/bin/bash
#SBATCH --job-name=epi4ab-process-rerun
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
echo "FORCE RERUN: Processing HER2 PDB files with new FreeSASA/PDB2PQR"
echo "=========================================="

# Use HER2 test set from SAbDab selection
HER2_LIST="$PROJ/sabdab_selection_results/her2_test_set_filtered_corrected.csv"
if [ ! -f "$HER2_LIST" ]; then
    HER2_LIST="$PROJ/sabdab_selection_results/her2_test_set_filtered.csv"
fi

if [ ! -f "$HER2_LIST" ]; then
    echo "ERROR: No HER2 list found"
    exit 1
fi

echo "Using HER2 list: $HER2_LIST"

# Extract PDB IDs from CSV
FIRST_LINE=$(head -1 "$HER2_LIST")
if [[ "$FIRST_LINE" =~ ^[A-Z0-9]{4}$ ]]; then
    PDB_IDS=$(cat "$HER2_LIST" | cut -d',' -f1 | tr '\n' ' ')
else
    PDB_IDS=$(tail -n +2 "$HER2_LIST" | cut -d',' -f1 | tr '\n' ' ')
fi
PDB_COUNT=$(echo "$PDB_IDS" | wc -w)

echo "Found $PDB_COUNT HER2 PDB IDs to process: $PDB_IDS"
echo ""

PDB_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files"
OUTPUT_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed"

successful=0
failed=0

for PDB_ID in $PDB_IDS; do
    echo "=========================================="
    echo "Processing: $PDB_ID (FORCE RERUN)"
    echo "=========================================="
    
    # Find PDB file
    PDB_FILE=$(find "$PDB_DIR" -name "${PDB_ID}.pdb" -o -name "${PDB_ID}_*.pdb" | head -1)
    
    if [ -z "$PDB_FILE" ]; then
        echo "⚠️  PDB file not found for $PDB_ID, skipping..."
        failed=$((failed + 1))
        continue
    fi
    
    echo "PDB file: $PDB_FILE"
    
    # Create output directory for this PDB
    PDB_OUTPUT_DIR="$OUTPUT_DIR/$PDB_ID"
    mkdir -p "$PDB_OUTPUT_DIR"
    
    # FORCE RERUN: Remove old cache directories to ensure fresh calculations
    echo "Clearing old cache directories for fresh FreeSASA/PDB2PQR calculations..."
    rm -rf "$PDB_OUTPUT_DIR/rsa_cache" "$PDB_OUTPUT_DIR/naccess_cache" "$PDB_OUTPUT_DIR/pdb2pqr_cache" 2>/dev/null || true
    
    # Process the PDB (will regenerate with new implementations)
    echo "Running processing pipeline with new FreeSASA/PDB2PQR..."
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
echo "Total: $PDB_COUNT"
echo ""

if [ $failed -eq 0 ]; then
    echo "✅ All HER2 PDB files processed successfully!"
else
    echo "⚠️  Some PDB files failed. Check logs for details."
fi
