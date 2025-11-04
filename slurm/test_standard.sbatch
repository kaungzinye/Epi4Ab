#!/bin/bash
#SBATCH --job-name=test-standard
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/test-standard-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/test-standard-%j.err

# Standardized SLURM test script for Epi4Ab pipeline
# Tests: Label generation, BepiPred/Ellipro integration, compute node behavior

set -euo pipefail
module purge
module load python/3.11.7

PROJ=/leonardo_work/AIFAC_F01_302/Epi4Ab
cd $PROJ
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=========================================="
echo "STANDARD TEST: Epi4Ab Pipeline (SLURM)"
echo "=========================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Run test script
python3 test_label_generation.py

deactivate

