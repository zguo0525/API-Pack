import os
import argparse

import tools.data_manger as dm
from tools.output_formatting import apply_format

"""
EXAMPLE: 
python step4_apply_custom_format.py --instructions_temp_file activity-tracker_api_db_instructions.json --output_file activity-tracker_api_db_instructions_final.json
python step4_apply_custom_format.py --instructions_temp_file activity-tracker_api_db_instructions.json --output_file activity-tracker_api_db_instructions_final.json --output_dir ./data/output/final_dataset/ibm
"""

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates_dir",
                    type=str,
                    required=False,
                    default="./data/templates", 
                    help="Directory to save all template files.") 
    parser.add_argument("--instruction_files_dir",
                    type=str,
                    required=False,
                    default="./data/temporal_files/instruction_files", 
                    help="Directory to load and/or save all temporal files.")
    parser.add_argument("--code_template",
                    type=str,
                    required=False,
                    default="code_template.txt",
                    help="The template to apply output format, it cannot be empty.")
    parser.add_argument("--instructions_temp_file",
                        type=str,
                        required=True,
                        help="The name of temporal file containing the API info and instructions in json format. It cannot be empty.")
    parser.add_argument("--output_dir",
                    type=str,
                    required=False,
                    default="./data/output/final_dataset", 
                    help="Directory to save all output files.") 
    parser.add_argument("--output_file",
                            type=str,
                            required=True,
                            help="The name of the file with format applied. It cannot be empty.")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()

    # INPUT
    INSTRUCTIONS_FILE = args.instructions_temp_file

    # TEMPLATES   
    TEMPLATES_DIR = args.templates_dir
    CODE_TEMPLATE = args.code_template

    # TEMPORAL FILES
    INSTRUCTIONS_FILE_DIR = args.instruction_files_dir
   

    # OUTPUT FILES
    OUTPUT_DIR = args.output_dir
    OUTPUT_FILE = args.output_file

    print("--- Arguments ---")
    print(f"\t Instructions file: {INSTRUCTIONS_FILE}")
    print(f"\t Templates directory: {TEMPLATES_DIR}")
    print(f"\t Code template file: {CODE_TEMPLATE}")
    print(f"\t Temporal directory: {INSTRUCTIONS_FILE_DIR}")
    print(f"\t Output file (final): {OUTPUT_FILE}")

    print(f"--- Loading temporal file with instructions {os.path.join(INSTRUCTIONS_FILE_DIR,INSTRUCTIONS_FILE)} ---")
    data = dm.load_local_file_as_json(os.path.join(INSTRUCTIONS_FILE_DIR,INSTRUCTIONS_FILE))

    print("--- Applying training format to API Dataset ---")
    output = apply_format(data=data, template_path = os.path.join(TEMPLATES_DIR,CODE_TEMPLATE))

    print(f"--- Saving API dataset with training format ---")
    dm.save_data_as_json_array(output,os.path.join(OUTPUT_DIR,OUTPUT_FILE))
    print(f"--- API dataset with training format applied saved as {os.path.join(OUTPUT_DIR,OUTPUT_FILE)} ---")