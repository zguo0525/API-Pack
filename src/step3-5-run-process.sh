#EXAMPLE: ./step3-5-run-process.sh -p ibm
#EXAMPLE: ./step3-5-run-process.sh -p api_gurus

while getopts m:p:a: opt
do
   case "$opt" in
      p ) datasource="$OPTARG";; # This is the dir name containing he input files (e.g., ibm, api_gurus)
   esac
done

python step3_5_validate_api_calls.py \
    --instructions_dir "./data/temporal_files/instruction_files/${datasource}/" \
    --instructions_cleaned_dir "./data/temporal_files/instruction_files_cleaned/${datasource}/"