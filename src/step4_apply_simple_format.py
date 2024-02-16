import os
import argparse
# import json

import tools.data_manger as dm
from tools.output_formatting import apply_default_format

"""
EXAMPLE: 
python step4_apply_simple_format.py --instruction_files_dir ./data/temporal_files/instruction_files --instructions_temp_file activity-tracker_api_db_instructions.json --output_dir ./data/output/final_dataset_simple/ibm --output_file activity-tracker_api_db_instructions_final.jsonl
"""

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction_files_dir",
                    type=str,
                    required=False,
                    default="./data/temporal_files/instruction_files", 
                    help="Directory to load and/or save all temporal files.")
    parser.add_argument("--instructions_temp_file",
                    type=str,
                    required=True,
                    help="The name of the instruction file. It cannot be empty.")
    parser.add_argument("--output_dir",
                    type=str,
                    required=False,
                    default="./data/output/final_dataset_simple", 
                    help="Directory to save all output files.") 
    parser.add_argument("--output_file",
                            type=str,
                            required=True,
                            help="The name of the file with format applied. It cannot be empty.")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()

    # INPUT
    INSTRUCTIONS_FILE_DIR = args.instruction_files_dir
    INSTRUCTIONS_FILE = args.instructions_temp_file
   
    # OUTPUT FILES
    OUTPUT_DIR = args.output_dir
    OUTPUT_FILE = args.output_file

    print("--- Arguments ---")
    print(f"\t Instructions file: {INSTRUCTIONS_FILE}")
    print(f"\t Temporal directory: {INSTRUCTIONS_FILE_DIR}")
    print(f"\t Output file (final): {OUTPUT_FILE}")

    print(f"--- Loading temporal file with instructions {os.path.join(INSTRUCTIONS_FILE_DIR,INSTRUCTIONS_FILE)} ---")
    data = dm.load_local_file_as_json(os.path.join(INSTRUCTIONS_FILE_DIR,INSTRUCTIONS_FILE))

    print(f"--- Applay default format ---")
    output = apply_default_format(data=data)

    print(f"--- Saving API dataset with training format ---")
    dm.save_as_jsonl(output,os.path.join(OUTPUT_DIR,OUTPUT_FILE))
    print(f"--- API dataset with the default training format was saved as {os.path.join(OUTPUT_DIR,OUTPUT_FILE)} ---")

    #TEST
    # with open(os.path.join(OUTPUT_DIR,OUTPUT_FILE)) as f:
    #     data = [json.loads(line) for line in f]

    # for datapoint in data:
    #     print(datapoint['output'])
    #     break