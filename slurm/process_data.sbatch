#!/bin/bash
#SBATCH --job-name=epi4ab-process
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
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

echo "Processing all HER2 PDB files..."

# Run the data processing pipeline with modular sequence fetching
# --sequence_source auto: Tries FASTA → RCSB API → PDB extraction
# To use FASTA files: add --fasta_dir /path/to/fasta_files/
# To force RCSB only: use --sequence_source rcsb
srun -u python -u data_processing/epi4ab_pipeline.py \
  --pdb_dir /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files/ \
  --output_dir /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/ \
  --sequence_source auto

echo "All PDB files processed successfully!"

deactivate
