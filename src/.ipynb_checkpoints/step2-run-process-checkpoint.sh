#!/bin/bash

#EXAMPLE with 'extracted' parser: ./step2-run-process.sh -m ./data/ -p extracted -a IBM 
#EXAMPLE with 'generated' parser: ./step2-run-process.sh -m ./data/ -p generated -a api_gurus

while getopts m:p:a: opt
do
   case "$opt" in
      m ) main_dir="$OPTARG" ;;
      p ) parser_id="$OPTARG" ;;
      a ) api_provider="$OPTARG" ;;
   esac
done

# Arguments
parser="${parser_id}"

if [[ "$parser" == "generated" ]]; then
    inputs_path="${main_dir}input/${api_provider}/generated/*.json"
    outputs_path="${main_dir}output/api_dbs/${api_provider}"
    echo "$outputs_path"
fi

if [[ "$parser" == "extracted" ]]; then
    inputs_path="${main_dir}input/${api_provider}/extracted/*.json"
    outputs_path="${main_dir}output/api_dbs/${api_provider}"
fi

provider="${api_provider}"
pos_str="api_db"

echo "Processing API DBs ..."
total_api_calls=0
for file in $inputs_path; do
    if [ -f "$file" ]; then
        
        base_name=$(basename "$file" .json)
        output_file="${base_name}_${pos_str}.json"

        python step2_1_build_api_db.py --input_dir "" \
        --input_file_name $file \
        --output_dir ${outputs_path} \
        --parser_id ${parser} \
        --api_db_output_file ${output_file}
        
        echo "$result"
        echo "${base_name} was processed!"
    fi
done
echo "Done!"