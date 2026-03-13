#!/bin/bash
# Phase 2 only; expects node_label_seqitope.parquet under nodes_edges/<pdb_id>.
# Offline safety: export TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 (optional HF_HOME).
#SBATCH --job-name=seqitope-inference
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
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

# Seqitope-specific paths
INFERENCE_LIST="$PROJ/inference_list_seqitope.csv"
DIRECTORY_NODES_EDGES="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/seqitope_ground_truth/nodes_edges"
DIRECTORY_PROCESSED_DATA="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/seqitope_ground_truth/processed_data"
DIRECTORY_INFERENCE_OUTPUT="/leonardo_work/EUHPC_D29_035/Epi4Ab/output_inference"

# Override environment variables for Seqitope
export DIRECTORY_NODES_EDGES
export DIRECTORY_PROCESSED_DATA

echo "=========================================="
echo "Epi4Ab Inference - Seqitope Dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "NODES_EDGES: $DIRECTORY_NODES_EDGES"
echo "PROCESSED_DATA: $DIRECTORY_PROCESSED_DATA"
echo "INFERENCE_LIST: $INFERENCE_LIST"
echo "MODEL: $FINAL_MODEL_FOLDER"
echo "OUTPUT: $DIRECTORY_INFERENCE_OUTPUT"
echo "=========================================="

# Check inference list exists
if [ ! -f "$INFERENCE_LIST" ]; then
    echo "ERROR: Inference list not found: $INFERENCE_LIST"
    exit 1
fi

# Count PDBs
TOTAL_PDBS=$(tail -n +2 "$INFERENCE_LIST" | wc -l)
echo "Total PDBs for inference: $TOTAL_PDBS"
echo ""

# Ensure output directory exists
mkdir -p "$DIRECTORY_INFERENCE_OUTPUT"
mkdir -p logs

# Set OMP threads for performance
export OMP_NUM_THREADS=4

# Validate inference inputs before running
echo "=========================================="
echo "Validating inference inputs"
echo "Started: $(date)"
echo "=========================================="

# Parse model config
CONFIG_FILE="/tmp/validation_config_${SLURM_JOB_ID}.json"
if [ -d "$FINAL_MODEL_FOLDER" ]; then
    python scripts/parse_model_config.py --model_dir "$FINAL_MODEL_FOLDER" --output "$CONFIG_FILE" 2>/dev/null || true
    # Add max_antigen_len from model parameters if not already present
    if [ -f "$CONFIG_FILE" ]; then
        python -c "
import json
import re
import os

config_file = '$CONFIG_FILE'
model_dir = '$FINAL_MODEL_FOLDER'

# Load existing config
with open(config_file, 'r') as f:
    config = json.load(f)

# Check if max_antigen_len is missing
if 'max_antigen_len' not in config.get('config', {}):
    # Try to find it in parameters_example.txt or log.md
    params_file = os.path.join(model_dir, 'parameters_example.txt')
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            content = f.read()
        match = re.search(r'max_antigen_len[:\s=]+(\d+)', content, re.IGNORECASE)
        if match:
            config['config']['max_antigen_len'] = int(match.group(1))
    
    # Save updated config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
" 2>/dev/null || true
    fi
fi

# Run validation for each PDB
VALIDATION_PASSED=true
while IFS= read -r pdb_id; do
    if [ -z "$pdb_id" ] || [ "$pdb_id" = "pdbId" ]; then
        continue
    fi
    echo ""
    echo "Validating $pdb_id..."
    if ! python scripts/validate_pipeline.py \
        --pdb_id "$pdb_id" \
        --nodes_edges_dir "$DIRECTORY_NODES_EDGES" \
        --processed_dir "$DIRECTORY_PROCESSED_DATA" \
        --step inference_inputs \
        --config "$CONFIG_FILE" 2>&1; then
        VALIDATION_PASSED=false
    fi
done < <(tail -n +2 "$INFERENCE_LIST")

if [ "$VALIDATION_PASSED" = false ]; then
    echo ""
    echo "ERROR: Validation failed. Please fix issues before running inference."
    echo "Check the validation output above for details."
    rm -f "$CONFIG_FILE"
    exit 1
fi

echo ""
echo "✓ All validation checks passed"
rm -f "$CONFIG_FILE"

# Run inference
echo ""
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
    
    # Validate inference outputs
    echo ""
    echo "=========================================="
    echo "Validating inference outputs"
    echo "=========================================="
    
    # Parse model config to determine required files
    CONFIG_FILE="$OUTPUT_DIR/validation_config.json"
    if [ -d "$FINAL_MODEL_FOLDER" ]; then
        python scripts/parse_model_config.py --model_dir "$FINAL_MODEL_FOLDER" --output "$CONFIG_FILE" 2>/dev/null || true
    fi
    
    # Run validation
    if [ -f "$CONFIG_FILE" ]; then
        python scripts/validate_pipeline.py \
            --output_dir "$OUTPUT_DIR" \
            --inference_list "$INFERENCE_LIST" \
            --nodes_edges_dir "$DIRECTORY_NODES_EDGES" \
            --step inference_outputs \
            --config "$CONFIG_FILE" || echo "Validation completed with warnings/errors (see above)"
    else
        # Run without config (will use defaults)
        python scripts/validate_pipeline.py \
            --output_dir "$OUTPUT_DIR" \
            --inference_list "$INFERENCE_LIST" \
            --nodes_edges_dir "$DIRECTORY_NODES_EDGES" \
            --step inference_outputs || echo "Validation completed with warnings/errors (see above)"
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

