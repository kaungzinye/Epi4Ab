#!/bin/bash
#SBATCH --job-name=epi4ab-infer-single
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
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

echo "Running inference on single PDB: 1N8Z"
echo "======================================"

# Create single PDB list file
echo "1N8Z" > /tmp/test_pdb_single.csv

# Run inference with single PDB
# Arguments: directory_data directory_processed_data directory_pdb_list directory_model_folder directory_output
srun -u python -u epi_model.py \
  /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/ \
  /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/ \
  /tmp/test_pdb_single.csv \
  final_trained_Epi4Ab \
  /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/run_model_output/

echo "======================================"
echo "Inference complete for 1N8Z!"

deactivate

