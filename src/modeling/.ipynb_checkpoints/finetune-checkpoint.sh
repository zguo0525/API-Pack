#!/bin/bash -x

# The lines started with SBATCH are directives to sbatch command.  Alternately, they can be specified on the command line.
#SBATCH -a 1
#SBATCH -J SFT
#SBATCH -o logs/SFT_%j_%a.out
#SBATCH -e logs/SFT_%j_%a.err
#SBATCH --gres=gpu:6
#SBATCH --nodes=24
#SBATCH --time=06:00:00

export HF_DATASETS_CACHE=/gpfs/u/home/SIFA/SIFAzhnu/scratch/llm4tools/src/modeling/cache
export OMP_NUM_THREADS=10
export LD_LIBRARY_PATH=/gpfs/u/home/SIFA/SIFAzhnu/scratch/cudnn-11.3-linux-ppc64le-v8.2.1.32/targets/ppc64le-linux/lib:$LD_LIBRARY_PATH

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export GPUS_PER_NODE=6
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export TOTAL_NUM_GPUS=$(( $SLURM_NNODES * $GPUS_PER_NODE ))
export TMPDIR=/gpfs/u/home/SIFA/SIFAzhnu/scratch/llm4tools/src/modeling/finetuning/cache

cd ./finetuning/

model_name=($1)
dataset_name=($2)
resume_from_checkpoint=($3)
fsdp_transformer_layer_cls_to_wrap=($4)

# Run the script
python -m torch.distributed.run --nproc_per_node=$GPUS_PER_NODE \
    --nnode=$SLURM_NNODES --node_rank=$SLURM_PROCID --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    finetune.py \
    --base_model /gpfs/u/home/SIFA/SIFAzhnu/scratch/llm4tools/src/modeling/LLMs/${model_name} \
    --data_path /gpfs/u/home/SIFA/SIFAzhnu/scratch/llm4tools/src/modeling/instr_data/${dataset_name} \
    --output_dir /gpfs/u/home/SIFA/SIFAzhnu/scratch/llm4tools/src/modeling/LLMs/output/${model_name}/${dataset_name} \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap ${fsdp_transformer_layer_cls_to_wrap} \
    --prompt_template_name gorilla \
    --num_epochs 2 \
    --learning_rate 2e-5 \
    --batch_size ${TOTAL_NUM_GPUS} \
    --cutoff_len 4096 \
    --group_by_length \
    --micro_batch_size=1 \
    --resume_from_checkpoint ${resume_from_checkpoint}