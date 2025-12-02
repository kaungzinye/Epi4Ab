#!/bin/bash
#SBATCH --job-name=epi4ab-infer-train-focal
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
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

echo "Running inference on TRAIN set with focal loss model"
echo "==========================================="
echo "Model: 2025-11-14_GNNResNet_1 (focal loss)"
echo "Dataset: HER2 train set"
echo ""

# Train set PDB list
TRAIN_LIST="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_lists/test/test_list.csv"

# Output directory for focal loss model inference
OUTPUT_DIR="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/inference_focal_output/train"
mkdir -p "$OUTPUT_DIR"

echo "Train set PDBs:"
cat "$TRAIN_LIST"
echo ""

# Run inference with focal loss model
# Arguments: directory_data directory_processed_data directory_pdb_list directory_model_folder directory_output
srun -u python -u epi_model.py \
  /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/ \
  /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/ \
  "$TRAIN_LIST" \
  2025-11-14_GNNResNet_1 \
  "$OUTPUT_DIR"

echo "==========================================="
echo "Train set inference complete!"
echo "Results saved to: $OUTPUT_DIR"

deactivate

