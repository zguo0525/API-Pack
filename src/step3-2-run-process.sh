#!/bin/bash

#EXAMPLE: ./step3-2-run-process.sh -m ./data -p ibm
#EXAMPLE: ./step3-2-run-process.sh -m ./data -p api_gurus

while getopts m:p:a: opt
do
   case "$opt" in
      m ) main_dir="$OPTARG" ;;
      p ) datasource="$OPTARG";; # This is the dir name containing he input files (e.g., ibm, api_gurus)
   esac
done

python step3_2_inst_examples_filter.py \
        --base_path ${main_dir} \
        --data_source ${datasource}