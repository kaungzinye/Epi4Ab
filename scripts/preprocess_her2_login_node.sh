#!/bin/bash
# Preprocess HER2 PDBs on login node to generate FreeSASA/PDB2PQR cache files
# This can be run on login node (not via SLURM) where Python dependencies are available

set -e

PROJ="/leonardo_work/AIFAC_F01_302/Epi4Ab"
cd "$PROJ"

# Activate virtual environment
source venv/bin/activate

# HER2 list
HER2_LIST="$PROJ/sabdab_selection_results/her2_test_set_filtered_corrected.csv"
PDB_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files"
OUTPUT_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed"

# Check if tools are available
if ! command -v pdb2pqr &> /dev/null; then
    echo "ERROR: pdb2pqr not found. Please install or add to PATH."
    exit 1
fi

python3 -c "import freesasa" >/dev/null 2>&1 || {
    echo "ERROR: Python package 'freesasa' not found. Install with: pip install freesasa"
    exit 1
}

echo "✓ FreeSASA (Python) and PDB2PQR found"

# Extract PDB IDs from CSV
FIRST_LINE=$(head -1 "$HER2_LIST")
if [[ "$FIRST_LINE" =~ ^[A-Z0-9]{4}$ ]]; then
    PDB_IDS=$(cat "$HER2_LIST" | cut -d',' -f1 | tr '\n' ' ')
else
    PDB_IDS=$(tail -n +2 "$HER2_LIST" | cut -d',' -f1 | tr '\n' ' ')
fi

echo "=========================================="
echo "PREPROCESSING HER2 PDBs ON LOGIN NODE"
echo "=========================================="
echo "PDB IDs: $PDB_IDS"
echo "This will generate cache files for FreeSASA/PDB2PQR"
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
        echo "  Removing old node_feature.parquet..."
        rm -f "$PDB_OUTPUT_DIR/node_feature.parquet"
    fi
    
    # Remove old cache directories to ensure fresh calculations
    echo "  Removing old cache directories..."
    rm -rf "$PDB_OUTPUT_DIR/rsa_cache" "$PDB_OUTPUT_DIR/naccess_cache" "$PDB_OUTPUT_DIR/pdb2pqr_cache" 2>/dev/null || true
    
    # Run preprocessing
    echo "  Running epi4ab_pipeline.py..."
    if python3 data_processing/epi4ab_pipeline.py \
        --pdb_file "$PDB_FILE" \
        --output_dir "$PDB_OUTPUT_DIR" \
        --antigen_chain A; then
        echo "  ✓ Processing completed"
    else
        echo "  ✗ Processing failed"
        continue
    fi
    
    # Verify features were added
    if [ -f "$PDB_OUTPUT_DIR/node_feature.parquet" ]; then
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
echo "PREPROCESSING COMPLETE"
echo "=========================================="
echo ""
echo "Verifying all features..."
python3 scripts/verify_features.py "$OUTPUT_DIR" --pdb-list "$(echo $PDB_IDS | tr ' ' ',')"

