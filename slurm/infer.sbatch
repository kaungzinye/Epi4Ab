#!/bin/bash
#SBATCH --job-name=epi4ab-infer
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
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
# Load environment variables properly (remove quotes)
set -a
source .env
set +a

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0}}

# Use the actual Epi4Ab inference command (testing with pre-trained model)
# NOTE: Model configuration is loaded from log.json in the model folder
# Only pass inference-specific flags here
MODEL_NAME="final_trained_Epi4Ab"
USE_REGION=false
USE_RELAXED=false
PLOT_NETWORK=false
INPUT_AB_FEATURE=true

MODEL_FOLDER="$DIRECTORY_OUTPUT/$MODEL_NAME"
PARAMS=""
[ "$USE_REGION" = true ] && PARAMS+="--use_region " || PARAMS+=""
[ "$USE_RELAXED" = true ] && PARAMS+="--use_relaxed " || PARAMS+=""
[ "$PLOT_NETWORK" = true ] && PARAMS+="--plot_network " || PARAMS+=""
[ "$INPUT_AB_FEATURE" = true ] && PARAMS+="--ab_feature_input " || PARAMS+=""



srun -u python -u epi_model.py "$DIRECTORY_DATA" "$DIRECTORY_PROCESSED_DATA" "$DIRECTORY_UNSEEN_PDB_LIST" "$MODEL_FOLDER" "$DIRECTORY_UNSEEN_OUTPUT" $PARAMS

deactivate

