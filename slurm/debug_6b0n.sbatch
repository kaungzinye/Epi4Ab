#!/bin/bash
#SBATCH --job-name=epi4ab-debug-6b0n
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

echo "=========================================="
echo "DEBUG: Reprocessing 6B0N with PDB sequence"
echo "=========================================="
echo "Creating debug output directory..."
mkdir -p /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed_debug/6B0N

echo "Processing 6B0N with --sequence_source pdb (forces PDB extraction)..."
echo "NOTE: Using antigen_chain G (ENVELOPE GLYCOPROTEIN GP140), not A!"
srun -u python -u data_processing/epi4ab_pipeline.py \
  --pdb_file /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files/6B0N.pdb \
  --output_dir /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed_debug/6B0N \
  --antigen_chain G \
  --sequence_source pdb

echo ""
echo "=========================================="
echo "Validating 6B0N debug output..."
echo "=========================================="
python validate_all.py \
  --processed_dir /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed_debug \
  --pdb_id 6B0N \
  --verbose

echo ""
echo "=========================================="
echo "DEBUG: Processing complete!"
echo "Check logs and validation output above."
echo "=========================================="

deactivate

