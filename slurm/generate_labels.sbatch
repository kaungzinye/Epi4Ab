#!/bin/bash
#SBATCH --job-name=epi4ab-gen-labels
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
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
echo "Epi4Ab Label Generation (node_label_pi.parquet)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Directories
PDB_DIR=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/pdb_complexes
OUTPUT_DIR=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/nodes_edges
CACHE_DIR=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/bepipred_cache

echo "PDB_DIR: $PDB_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "CACHE_DIR: $CACHE_DIR"
echo ""

# Mapping of base PDB ID to full ID (skip 1n8z_BAC which already has labels)
declare -A PDB_MAP
PDB_MAP["1s78"]="1s78_DCA"
PDB_MAP["3be1"]="3be1_HLA"
PDB_MAP["3n85"]="3n85_HLA"
PDB_MAP["3wlw"]="3wlw_CDA"
PDB_MAP["3wsq"]="3wsq_HLA"
PDB_MAP["4lst"]="4lst_HLG"
PDB_MAP["4mwf"]="4mwf_ABD"
PDB_MAP["4ywg"]="4ywg_HLG"
PDB_MAP["5o4g"]="5o4g_BAC"
PDB_MAP["6att"]="6att_HLA"
PDB_MAP["6j6y"]="6j6y_EFD"
PDB_MAP["6mug"]="6mug_HLG"
PDB_MAP["6nms"]="6nms_HLS"
PDB_MAP["6nmu"]="6nmu_BAC"
PDB_MAP["6urm"]="6urm_DEC"
PDB_MAP["6wo5"]="6wo5_HLE"
PDB_MAP["7l7r"]="7l7r_DCG"
PDB_MAP["7lf7"]="7lf7_ABM"
PDB_MAP["7mn8"]="7mn8_DCB"

echo "Processing 19 PDBs (skipping 1n8z_BAC which has labels)"
echo ""

# Process each PDB individually
for pdb in "${!PDB_MAP[@]}"; do
    full_id="${PDB_MAP[$pdb]}"
    
    echo "=========================================="
    echo "Processing: $pdb -> $full_id"
    echo "Started: $(date)"
    echo "=========================================="
    
    pdb_file="$PDB_DIR/${pdb}.pdb"
    output_subdir="$OUTPUT_DIR/${full_id}"
    
    if [ ! -f "$pdb_file" ]; then
        echo "  ERROR: PDB file not found: $pdb_file"
        continue
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_subdir"
    
    # Extract antigen chain from full ID (last character)
    antigen_chain=$(echo "$full_id" | rev | cut -c1 | rev)
    
    # Run label generation for this PDB
    python scripts/generate_labels.py \
        --pdb_file "$pdb_file" \
        --output_dir "$output_subdir" \
        --cache_dir "$CACHE_DIR" \
        --antigen_chain "$antigen_chain" \
        --pdb_id "$full_id" \
        || echo "  WARNING: Failed to process $pdb"
    
    echo ""
done

echo "=========================================="
echo "Label generation complete"
echo "Finished: $(date)"
echo "=========================================="

# Verify results
echo ""
echo "Verification: Checking generated labels"
echo "=========================================="
for pdb in "${!PDB_MAP[@]}"; do
    full_id="${PDB_MAP[$pdb]}"
    label_file="$OUTPUT_DIR/$full_id/node_label_pi.parquet"
    
    if [ -f "$label_file" ]; then
        echo "  ✓ $full_id: node_label_pi.parquet exists"
    else
        echo "  ✗ $full_id: MISSING node_label_pi.parquet"
    fi
done

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi

