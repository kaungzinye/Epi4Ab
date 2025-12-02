#!/bin/bash
#SBATCH --job-name=epi4ab-infer-test-focal
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
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

echo "Running inference on TEST set with focal loss model"
echo "==========================================="
echo "Model: 2025-11-14_GNNResNet_1 (focal loss)"
echo "Dataset: HER2 test set (SAbDab filtered)"
echo ""

# Use HER2 test set from SAbDab selection
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
    echo "Error: HER2 test list not found!"
    exit 1
fi

# Filter test list to only include PDBs with preprocessed data
PROCESSED_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed"
FILTERED_LIST="/tmp/test_pdb_focal.csv"

echo "Filtering HER2 test list to only include preprocessed PDBs..."
> "$FILTERED_LIST"  # Create empty file

# Read HER2 list and check if each PDB has preprocessed data
while IFS= read -r line || [ -n "$line" ]; do
    PDB_ID=$(echo "$line" | cut -d',' -f1 | tr -d '[:space:]')
    
    # Skip empty lines
    if [ -z "$PDB_ID" ]; then
        continue
    fi
    
    # Check if preprocessed data exists
    if [ -f "$PROCESSED_DIR/$PDB_ID/node_feature.parquet" ]; then
        echo "$PDB_ID" >> "$FILTERED_LIST"
        echo "  ✓ $PDB_ID - has preprocessed data"
    else
        echo "  ✗ $PDB_ID - skipping (no preprocessed data)"
    fi
done < "$HER2_LIST"

PDB_COUNT=$(wc -l < "$FILTERED_LIST" | tr -d ' ')
if [ "$PDB_COUNT" -eq 0 ]; then
    echo "Error: No preprocessed PDBs found in HER2 test list!"
    exit 1
fi

echo ""
echo "Processing $PDB_COUNT HER2 test PDBs..."
echo "PDBs: $(cat $FILTERED_LIST | tr '\n' ' ')"
echo ""

# Output directory for focal loss model inference
OUTPUT_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/inference_focal_output/test"
mkdir -p "$OUTPUT_DIR"

# Run inference with focal loss model
# Arguments: directory_data directory_processed_data directory_pdb_list directory_model_folder directory_output
srun -u python -u epi_model.py \
  /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/ \
  /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/ \
  "$FILTERED_LIST" \
  2025-11-14_GNNResNet_1 \
  "$OUTPUT_DIR"

echo "==========================================="
echo "Test set inference complete for $PDB_COUNT structures!"
echo "Results saved to: $OUTPUT_DIR"

deactivate

