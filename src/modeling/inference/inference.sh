#!/bin/bash
#SBATCH --output=/gpfs/u/home/SIFA/SIFAzhnu/scratch/logs/%j.log
#SBATCH --error=/gpfs/u/home/SIFA/SIFAzhnu/scratch/logs/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1        # total number of tasks across all nodes
#SBATCH --qos=dcs-48hr    # added line for 48 hours QoS
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:32g:2
#SBATCH --cpus-per-gpu=10
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Zhen.Guo@partner.ibm.com

# eval "$(conda shell.bash hook)"
# conda activate llm

export HF_DATASETS_CACHE=/gpfs/u/home/SIFA/SIFAzhnu/scratch/cache
export OMP_NUM_THREADS=10
export LD_LIBRARY_PATH=/gpfs/u/home/SIFA/SIFAzhnu/scratch/cudnn-11.3-linux-ppc64le-v8.2.1.32/targets/ppc64le-linux/lib:$LD_LIBRARY_PATH

set -e
set -x

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

echo "Experiment name: $1"
echo "Command: $2"
eval "$2" &>> loggings/$1.log
# submit <exp_name> <command>