#!/bin/bash

#EXAMPLE: ./step3-1-run-process.sh -m ./data/ -p ibm
#EXAMPLE: ./step3-1-run-process.sh -m ./data/ -p api_gurus

while getopts m:p:a: opt
do
   case "$opt" in
      m ) main_dir="$OPTARG" ;;
      p ) datasource="$OPTARG";; # This is the dir name containing he input files (e.g., ibm, api_gurus)
   esac
done

# Arguments -> update the url for each 
inputs_path="${main_dir}output/api_dbs/${datasource}/*_api_db.json"
outputs_path="${main_dir}temporal_files/inst_examples/${datasource}"

echo "$inputs_path"
echo "$outputs_path"

# vars
pos_str="inst_exa"

echo "Processing in-context examples for each API DB ..."

max_jobs=10

for file in $inputs_path; do
    if [ -f "$file" ]; then
        while [ $(jobs -p | wc -l) -ge $max_jobs ]; do
            sleep 0.5  # Wait before checking the number of running jobs
        done

        base_name=$(basename "$file" .json)
        output_file="${base_name}_${pos_str}.json"

        # Run in the background
        python step3_1_inst_examples_gen.py --api_db_dir "" \
        --api_db_file $file \
        --inst_exa_file ${output_file} \
        --inst_exa_dir ${outputs_path} &
        
        echo "Started processing ${base_name}"
    fi
done

# Wait for all background jobs to finish
wait
echo "Done!"