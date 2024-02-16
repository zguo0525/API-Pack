export HF_DATASETS_CACHE=/gpfs/u/home/SIFA/SIFAzhnu/scratch/cache
export OMP_NUM_THREADS=10
export LD_LIBRARY_PATH=/gpfs/u/home/SIFA/SIFAzhnu/scratch/cudnn-11.3-linux-ppc64le-v8.2.1.32/targets/ppc64le-linux/lib:$LD_LIBRARY_PATH

set -e
set -x

model_names=("CodeLlama-13b-hf")

sizes=("98914")

langs=("cleaned_python" "cleaned_java" "cleaned_go" "cleaned_javascript" "cleaned_libcurl" "cleaned_node" "cleaned_php" "cleaned_ruby" "cleaned_swift")
method="cleaned_python"

for size in "${sizes[@]}"; do

    for lang in "${langs[@]}"; do
    
        test_dataset_names=("total_testing_${lang}_level_1.json" "total_testing_${lang}_level_2.json" "total_testing_${lang}_level_3.json" "total_testing_${lang}_level_1_retrieval_IC_3.json" "total_testing_${lang}_level_2_retrieval_IC_3.json" "total_testing_${lang}_level_3_retrieval_IC_3.json")
        
        for model_name in "${model_names[@]}"; do
            
            train_dataset_names=("${method}/tokenized_combined_data_${model_name}_${size}")
            
            for train_dataset in "${train_dataset_names[@]}"; do
                for test_dataset in "${test_dataset_names[@]}"; do
                    export EXP_NAME="${model_name}_${method}_${lang}_${size}_${test_dataset}"
                    echo $EXP_NAME
                    
                    COMMAND="python model_eval_multi_cross.py \
                    --model-path ../LLMs/output_aws/output/${model_name}/${train_dataset} \
                    --question-file ../instr_data/${lang}/${test_dataset} \
                    --answer-file ${model_name}/${train_dataset}/${test_dataset}.json"
        
                    # python model_eval_multi_batch.py \
                    # --model-path ../LLMs/output_aws/output/${model_name}/${train_dataset} \
                    # --question-file ../instr_data/${lang}/${test_dataset} \
                    # --answer-file ${model_name}/${train_dataset}/${test_dataset}.json
            
                    sbatch inference.sh "${EXP_NAME}" "$COMMAND"
            
                    echo "Sleeping for 2 seconds..."
                    sleep 2
                done
            
            done
        
        done
    
    done

done