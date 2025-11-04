#!/bin/bash
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=epi4ab-reprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/epi4ab-reprocess-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/epi4ab-reprocess-%j.err

# Reprocess ALL 12 HER2 structures with VLa columns fix
# This script reprocesses all structures to ensure they have the complete 50 columns

set -e

echo "=========================================="
echo "Epi4Ab: Reprocess All HER2 Structures"
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
DATA_DIR="$WORK_DIR/data"
PROCESSED_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed"

cd $WORK_DIR

# List of 12 HER2 PDB IDs (as currently processed)
PDB_LIST="1N8Z 1S78 3N85 4K5Y 5F8B 6B0J 6B0K 6B0L 6B0N 6B0O 6B0P 6B0Q"

echo "Reprocessing 12 HER2 structures with VLa columns fix..."
echo ""

for PDB_ID in $PDB_LIST; do
    echo "=========================================="
    echo "Processing: $PDB_ID"
    echo "=========================================="
    
    # Find PDB file
    PDB_FILE=$(find $DATA_DIR -name "${PDB_ID}.pdb" -o -name "${PDB_ID}_*.pdb" | head -1)
    
    if [ -z "$PDB_FILE" ]; then
        echo "⚠️  PDB file not found for $PDB_ID, skipping..."
        continue
    fi
    
    echo "PDB file: $PDB_FILE"
    
    # Backup old data if exists
    if [ -d "$PROCESSED_DIR/$PDB_ID" ]; then
        echo "Backing up old processed data..."
        BACKUP_DIR="$PROCESSED_DIR/${PDB_ID}_backup_$(date +%Y%m%d_%H%M%S)"
        mv "$PROCESSED_DIR/$PDB_ID" "$BACKUP_DIR"
        echo "Backed up to: $BACKUP_DIR"
    fi
    
    # Process with fixed pipeline
    echo "Running processing pipeline..."
    python data_processing/epi4ab_pipeline.py \
        --pdb_file "$PDB_FILE" \
        --output_dir "$PROCESSED_DIR/$PDB_ID" \
        --antigen_chain A \
        --sequence_source auto
    
    if [ $? -eq 0 ]; then
        echo "✅ $PDB_ID processed successfully"
        
        # Quick validation: check column count
        COL_COUNT=$(python -c "import pandas as pd; print(len(pd.read_parquet('$PROCESSED_DIR/$PDB_ID/node_feature.parquet').columns))")
        echo "   Node features: $COL_COUNT columns"
        
        if [ "$COL_COUNT" -eq "50" ]; then
            echo "   ✅ Correct number of columns"
        else
            echo "   ⚠️  Expected 50 columns, got $COL_COUNT"
        fi
    else
        echo "❌ $PDB_ID processing failed"
    fi
    
    echo ""
done

echo "=========================================="
echo "Reprocessing Complete"
echo "End time: $(date)"
echo "=========================================="

# Run validation
echo ""
echo "Running validation on all processed data..."
python validate_features.py --processed_dir $PROCESSED_DIR --quiet

if [ $? -eq 0 ]; then
    echo "✅ All structures validated successfully!"
else
    echo "⚠️  Some structures failed validation. Check details above."
fi

