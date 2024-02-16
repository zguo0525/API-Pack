export OMP_NUM_THREADS=10
export LD_LIBRARY_PATH=/gpfs/u/home/SIFA/SIFAzhnu/scratch/cudnn-11.3-linux-ppc64le-v8.2.1.32/targets/ppc64le-linux/lib:$LD_LIBRARY_PATH

set -e
set -x

model_names=("CodeLlama-7b-hf")
#model_names=("CodeLlama-13b-hf")
#model_names=("Llama-2-13b-hf")
#model_names=("granite-13b-base-v2")
#model_names=("Mistral-7B-v0.1")

method="combined"

#size="80000"
size="100664"

checkpoint_prefix="/gpfs/u/home/SIFA/SIFAzhnu/scratch/llm4tools/src/modeling/LLMs/output"

# Loop through the models and submit a job for each
for model_name in "${model_names[@]}"; do

    dataset_names=("total_curl/tokenized_${method}_data_${model_name}_${size}")
    
    #"total_curl/tokenized_simple_data_${model_name}_${size}")
    #"total_curl/tokenized_combined_data_${model_name}_${size}")
    
    #resume_from_checkpoint="${checkpoint_prefix}/${model_name}/total_curl/tokenized_${method}_data_${model_name}_${size}/checkpoint-400"

    # Determine the appropriate FSDP class to wrap based on the model name
    if [[ "$model_name" == *"Llama"* ]]; then
        fsdp_transformer_layer_cls_to_wrap="LlamaDecoderLayer"
    elif [[ "$model_name" == *"granite"* ]]; then
        fsdp_transformer_layer_cls_to_wrap="GPTBigCodeBlock"
    elif [[ "$model_name" == *"Mistral"* ]]; then
        fsdp_transformer_layer_cls_to_wrap="MistralDecoderLayer"
    else
        echo "Unknown model: $model_name"
        continue # Skip this iteration
    fi
    
    # Now, loop through the datasets (if more than one)
    for dataset_name in "${dataset_names[@]}"; do
        # Construct the job name or any other parameters dynamically
        job_name="${model_name} with ${dataset_name}"

        # Submit the job
        salloc --nodes 20 --time 6:00:00 --gres=gpu:32g:6 srun bash finetune.sh "${model_name}" "${dataset_name}" "${resume_from_checkpoint}" "${fsdp_transformer_layer_cls_to_wrap}"
        # Empty quotes "" for resume_from_checkpoint if you don't want to resume from a checkpoint

        echo "Submitted job: $job_name"
    done
done

#--qos=dcs-48hr