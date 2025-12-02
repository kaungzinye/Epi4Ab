#!/bin/bash
#SBATCH --job-name=epi4ab-reprocess-her2-features
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/%x-%j.err

# Reprocess HER2 PDBs to ensure RSA (FreeSASA) and charge features are included
# This forces reprocessing even if node_feature.parquet already exists

set -e

PROJ="/leonardo_work/AIFAC_F01_302/Epi4Ab"
cd "$PROJ"

# Activate virtual environment
source venv/bin/activate

# HER2 list
HER2_LIST="$PROJ/sabdab_selection_results/her2_test_set_filtered_corrected.csv"
PDB_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files"
OUTPUT_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed"

# Extract PDB IDs from CSV (first column, handle both with/without header)
FIRST_LINE=$(head -1 "$HER2_LIST")
if [[ "$FIRST_LINE" =~ ^[A-Z0-9]{4}$ ]]; then
    PDB_IDS=$(cat "$HER2_LIST" | cut -d',' -f1 | tr '\n' ' ')
else
    PDB_IDS=$(tail -n +2 "$HER2_LIST" | cut -d',' -f1 | tr '\n' ' ')
fi

echo "=========================================="
echo "REPROCESSING HER2 PDBs FOR MISSING FEATURES"
echo "=========================================="
echo "PDB IDs: $PDB_IDS"
echo "Force reprocessing: YES (removing old node_feature.parquet files)"
echo ""

# Process each PDB
for PDB_ID in $PDB_IDS; do
    PDB_ID=$(echo "$PDB_ID" | tr -d '[:space:]')
    if [ -z "$PDB_ID" ]; then continue; fi
    
    echo "----------------------------------------"
    echo "Processing $PDB_ID..."
    echo "----------------------------------------"
    
    # Find PDB file
    PDB_FILE=$(find "$PDB_DIR" -name "${PDB_ID}.pdb" -type f | head -1)
    if [ -z "$PDB_FILE" ]; then
        echo "  ✗ PDB file not found: ${PDB_ID}.pdb"
        continue
    fi
    
    PDB_OUTPUT_DIR="$OUTPUT_DIR/$PDB_ID"
    mkdir -p "$PDB_OUTPUT_DIR"
    
    # Remove old node_feature.parquet to force reprocessing
    if [ -f "$PDB_OUTPUT_DIR/node_feature.parquet" ]; then
        echo "  Removing old node_feature.parquet to force reprocessing..."
        rm -f "$PDB_OUTPUT_DIR/node_feature.parquet"
    fi
    
    # Remove old cache directories to ensure fresh calculations
    echo "  Removing old cache directories..."
    rm -rf "$PDB_OUTPUT_DIR/rsa_cache" "$PDB_OUTPUT_DIR/naccess_cache" "$PDB_OUTPUT_DIR/pdb2pqr_cache" 2>/dev/null || true
    
    # Run preprocessing
    echo "  Running epi4ab_pipeline.py..."
    if srun -u python3 data_processing/epi4ab_pipeline.py \
        --pdb_file "$PDB_FILE" \
        --output_dir "$PDB_OUTPUT_DIR" \
        --antigen_chain A; then
        echo "  ✓ node_feature.parquet created"
    else
        echo "  ✗ epi4ab_pipeline.py failed"
        continue
    fi
    
    # Verify features were added
    if [ -f "$PDB_OUTPUT_DIR/node_feature.parquet" ]; then
        # Check for sasa and charge columns
        python3 << PYTHON_SCRIPT
import pandas as pd
try:
    df = pd.read_parquet("$PDB_OUTPUT_DIR/node_feature.parquet")
    has_sasa = 'sasa' in df.columns
    has_charge = 'charge' in df.columns
    if has_sasa and has_charge:
        sasa_nonzero = (df['sasa'] > 0).sum()
        charge_nonzero = (df['charge'].abs() > 1e-6).sum()
        print(f"  ✓ Features verified: sasa ({sasa_nonzero}/{len(df)} non-zero), charge ({charge_nonzero}/{len(df)} non-zero)")
    else:
        print(f"  ✗ Missing features: sasa={has_sasa}, charge={has_charge}")
except Exception as e:
    print(f"  ✗ Error verifying features: {e}")
PYTHON_SCRIPT
    else
        echo "  ✗ Failed to create node_feature.parquet"
    fi
    
    echo ""
done

echo "=========================================="
echo "REPROCESSING COMPLETE"
echo "=========================================="
echo ""
echo "Verifying all features..."
python3 scripts/verify_features.py "$OUTPUT_DIR" --pdb-list "$(echo $PDB_IDS | tr ' ' ',')"

