#EXAMPLE: ./step3-3-run-process.sh -m ./data -p ibm
#EXAMPLE: ./step3-3-run-process.sh -m ./data -p api_gurus

while getopts m:p:a: opt
do
   case "$opt" in
      m ) main_dir="$OPTARG" ;;
      p ) datasource="$OPTARG";; # This is the dir name containing he input files (e.g., ibm, api_gurus)
   esac
done

model_path="/gpfs/u/home/SIFA/SIFAzhnu/scratch/LLM/Mistral-7B-Instruct-v0.2"
examples_path="${main_dir}/temporal_files/inst_examples/${datasource}/*.json"

# Initialize the counter
processed_count=0
total_files_count=0

# Loop over each JSON file in the directory
for file in $examples_path; do
    # Increment the total files counter
    total_files_count=$((total_files_count + 1))
    if [ -f "$file" ]; then
        # Check if any entry in 'list' lacks 'output to refine' or it is empty
        if jq -e '.list[] | select(.["output to refine"] | not or .["output to refine"] == "")' "$file" > /dev/null; then
            echo "----------------------------------------"
            echo "Processing instruction examples for $file ..."

            # Construct the command to be executed
            COMMAND="python step3_3_inst_examples_rewrite.py \
                        --ins_ex_path ${file} \
                        --model_path ${model_path} \
                        --num_gpus 1"

            # Submit the job using sbatch
            if sbatch inference.sh "${file}" "$COMMAND"; then
            #if $COMMAND; then
                processed_count=$((processed_count + 1))
                echo "Job successfully submitted for file: $file"
            else
                echo "Error submitting job for file: $file"
            fi

            sleep 0.1
        fi
    fi
done

# After the loop, display the count of processed files and total files
echo "Total number of files processed: $processed_count"
echo "Total number of files examined: $total_files_count"