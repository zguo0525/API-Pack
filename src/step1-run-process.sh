#!/bin/bash

main_dir="./openapi_spec/"
folder_name="api_gurus"

echo "Processing folder: $folder_name"

# Fixed input_dir to point directly to the files
input_dir="${main_dir}${folder_name}/*.json"

# Use a glob expansion directly in the loop to handle multiple files
for file in $input_dir; do
    if [ -f "$file" ]; then
        base_name_input_file=$(basename "$file" .json)
        echo "Processing file: $base_name_input_file"

        # Make sure to use 'node' (check the case sensitivity)
        # Also ensure your script path is correctly specified relative to where you're running this script
        result=$(node step1_generate_api_calls.js "${main_dir}" "${base_name_input_file}" "${folder_name}")
        echo "Result: $result"
    else
        echo "No JSON files found in $input_dir"
    fi
done