#!/bin/bash
#SBATCH --job-name=epi4ab-train
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
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

# Load environment variables properly (remove quotes)
set -a
source .env
set +a

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0}}

# Use the actual Epi4Ab training command
PARAMS=""
while IFS='=' read -r key value || [ -n "$key" ]; do
    [[ ! $key =~ ^\ *# && -n $key ]] && PARAMS+="--${key// /} $value "
done < parameters.txt

PARAMETERS="$(<parameters.txt)"
PARAMETERS=$(echo "$PARAMETERS" | tr '\n' ';')
PARAMETERS=$(echo "$PARAMETERS" | sed 's/ /+/g')

srun -u python -u epi_graph.py $PARAMETERS "$DIRECTORY_DATA" "$DIRECTORY_PROCESSED_DATA" "$DIRECTORY_PDB_LIST" "$DIRECTORY_UNSEEN_PDB_LIST" \
                    "$DIRECTORY_AF_PDB_LIST" "$DIRECTORY_UNSEEN_AF_PDB_LIST" "$DIRECTORY_OUTPUT" $PARAMS

deactivate

