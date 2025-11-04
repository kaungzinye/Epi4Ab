#!/bin/bash
#SBATCH --job-name=epi4ab-infer-all
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

echo "Running inference on all 12 HER2 structures"
echo "==========================================="

# Create PDB list file with all 12 HER2 structures (actually processed)
cat > /tmp/test_pdb_all.csv << EOF
1N8Z
1S78
3N85
4K5Y
5F8B
6B0J
6B0K
6B0L
6B0N
6B0O
6B0P
6B0Q
EOF

echo "Processing 12 HER2 PDB files..."

# Run inference with all PDSs
# Arguments: directory_data directory_processed_data directory_pdb_list directory_model_folder directory_output
srun -u python -u epi_model.py \
  /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/ \
  /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/ \
  /tmp/test_pdb_all.csv \
  final_trained_Epi4Ab \
  /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/run_model_output/

echo "==========================================="
echo "Inference complete for all 12 HER2 structures!"

deactivate

