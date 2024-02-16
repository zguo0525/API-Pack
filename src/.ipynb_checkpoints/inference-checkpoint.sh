#!/bin/bash
#SBATCH --output=/gpfs/u/home/SIFA/SIFAzhnu/scratch/logs/%j.log
#SBATCH --error=/gpfs/u/home/SIFA/SIFAzhnu/scratch/logs/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:32g:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mail-type=FAIL

# eval "$(conda shell.bash hook)"
# conda activate llm

export HF_DATASETS_CACHE=/gpfs/u/home/SIFA/SIFAzhnu/scratch/cache
export OMP_NUM_THREADS=10
export LD_LIBRARY_PATH=/gpfs/u/home/SIFA/SIFAzhnu/scratch/cudnn-11.3-linux-ppc64le-v8.2.1.32/targets/ppc64le-linux/lib:$LD_LIBRARY_PATH

set -e
set -x

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# Extract experiment name and command
exp_name=$1
command=$2

# Remove './' from the start of exp_name if present
cleaned_exp_name=${exp_name#./}

# Sanitize exp_name to create a valid log file name
log_file_name=$(echo "$cleaned_exp_name" | sed 's:/:_:g')

# Log directory
log_dir="loggings"

# Ensure log directory exists
mkdir -p "$log_dir"

# Full path for the log file
log_file="${log_dir}/${log_file_name}.log"

echo "Experiment name: $exp_name"
echo "Command: $command"

# Execute the command and redirect output to log file
eval "$command" &>> "$log_file"
# submit <exp_name> <command>