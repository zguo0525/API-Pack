#!/bin/bash

# EXAMPLE: ./step4-run-process.sh -m ./data/ -p ibm -a custom
# EXAMPLE: ./step4-run-process.sh -m ./data/ -p api_gurus -a custom
# EXAMPLE: ./step4-run-process.sh -m ./data/ -p ibm -a simple
# EXAMPLE: ./step4-run-process.sh -m ./data/ -p api_gurus -a simple

while getopts m:p:a: opt
do
   case "$opt" in
      m ) main_dir="$OPTARG";;
      p ) source="$OPTARG";; # This is the dir name containing he input files (e.g., ibm, api_gurus)
      a ) processing_type="$OPTARG";; # simple: for jsonl file with input,out pairs; custom: for gorilla output
   esac
done

# Arguments
inputs_path="${main_dir}temporal_files/instruction_files_cleaned/${source}/*_api_db_instructions.json"
output_dir="./data/output/final_dataset_${processing_type}/${source}"
script_name="step4_apply_${processing_type}_format.py"

if [ $processing_type == 'custom' ]; then
    format="json"
    echo "$format"
fi

if [ $processing_type == 'simple' ]; then
    format="jsonl"
    echo "$format"
fi


echo "$inputs_path"
echo "$output_dir"
echo "$script_name"

# vars
output_pos_str="final"

echo "Applying final format for each API DB ......"

for file in $inputs_path; do
    if [ -f "$file" ]; then
        echo "$file"
        
        base_name=$(basename "$file" .json)
        output_file="${base_name}_${output_pos_str}.${format}"
        echo "$output_file"

        python ${script_name} --instruction_files_dir "" \
        --instructions_temp_file $file \
        --output_file ${output_file} \
        --output_dir ${output_dir}
    fi
done
echo "Done!"