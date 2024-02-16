#EXAMPLE: ./step3-4-run-process.sh -m ./data/ -p ibm
#EXAMPLE: ./step3-4-run-process.sh -m ./data/ -p api_gurus

while getopts m:p:a: opt
do
   case "$opt" in
      m ) main_dir="$OPTARG" ;;
      p ) datasource="$OPTARG";; # This is the dir name containing he input files (e.g., ibm, api_gurus)
   esac
done

# Arguments
inputs_path="${main_dir}output/api_dbs/${datasource}/*_api_db.json"
examples_dir="${main_dir}temporal_files/inst_examples/${datasource}/"
instruction_files_dir="${main_dir}temporal_files/instruction_files/${datasource}/"
logs_path="${main_dir}logs/"

model_path="/gpfs/u/home/SIFA/SIFAzhnu/scratch/LLM/Mistral-7B-Instruct-v0.2"

# vars
examples_pos_str="inst_exa"
instructions_pos_str="instructions"

processed_files_log="./processed_files.log"
touch "$processed_files_log"  # Create the log file if it doesn't exist

for file in $inputs_path; do
    if [ -f "$file" ]; then
        base_name=$(basename "$file" .json)

        # Skip the file if it's already processed
        if grep -q "$base_name" "$processed_files_log"; then
            echo "Skipping already processed file: $file"
            continue
        fi

        echo "----------------------------------------"
        echo "Processing instructions for $file ..."

        check_examples_file="${examples_dir}/${base_name}_${examples_pos_str}.json"
        
        examples_file="./${datasource}/${base_name}_${examples_pos_str}.json"
        instructions_file="${datasource}/${base_name}_${instructions_pos_str}.json"

        # Check if any entry in 'list' lacks 'output to refine' or it is empty for examples_file
        if jq -e '.list[] | select(.["output to refine"] and .["output to refine"] != "")' "$check_examples_file" > /dev/null; then
            echo "----------------------------------------"
            echo "processing"
            echo "----------------------------------------"
            # Construct the command to be executed
            COMMAND="python step3_4_instructions_gen_local.py --input_dir '${main_dir}output/api_dbs/${datasource}' \
            --api_db_file '${base_name}.json' \
            --prompt_examples '${examples_file}' \
            --model_path ${model_path} \
            --instructions_file '${instructions_file}'"
    
            # Submit the job using sbatch
            if sbatch inference.sh "${file}" "$COMMAND"; then
                echo "$base_name" >> "$processed_files_log"  # Log the processed file
                echo "Job successfully submitted for file: $file"
            else
                echo "Error submitting job for file: $file"
            fi
    
            sleep .1
        fi
    fi
done