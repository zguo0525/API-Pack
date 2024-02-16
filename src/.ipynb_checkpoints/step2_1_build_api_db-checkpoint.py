import os
import argparse

import tools.data_manger as dm
import tools.output_formatting as of
import tools.spec_file_parser as fp

"""
EXAMPLES:
python step2_1_build_api_db.py --input_dir /Users/amezasor/Projects/llm4tools/src/data/input/api_gurus/generated --input_file_name api_gurus_1password.local_connect_1.5.7_openapi_api_calls.json --parser_id generated --output_dir ./data/output/api_dbs/api_gurus --api_db_output_file api_gurus_1password.local_connect_1.5.7_openapi_api_calls_api_db.json
"""

API_CALLS_GENERATED = "generated"
API_CALLS_EXTRACTED = "extracted"

def create_parser(source,parser,provider):
    my_parser = None
    if parser.lower() == API_CALLS_GENERATED:
        my_parser = fp.GenParser()
        my_parser.parse_data(source, provider)
    elif parser.lower() == API_CALLS_EXTRACTED.lower():
        my_parser = fp.ExtParser() 
        my_parser.parse_data(source, provider)
    return my_parser

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", # INPUT_PATH
                        type=str,
                        required=True,
                        help="Directory where input files are located.")
    parser.add_argument("--input_file_name",
                        type=str,
                        required=True,
                        help="The input file name, it cannot be empty.")
    parser.add_argument("--parser_id",
                        type=str,
                        required=True,
                        help="A str to identify the Parser class to be used.")
    parser.add_argument("--api_provider",
                    type=str,
                    required=False,
                    default="",
                    help="A str to identify the API provider (e.g., IBM, AWS, Microsoft).")
    parser.add_argument("--output_dir",
                        type=str,
                        required=False,
                        default="./data/output/api_dbs", 
                        help="Directory to save all output files.") 
    parser.add_argument("--api_db_output_file",
                        type=str,
                        required=True,
                        help="The name of output file containing the API info in json format. It cannot be empty.")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()
    # INPUT
    INPUT_DIR = args.input_dir
    INPUT_FILE = args.input_file_name

    # OUTPUT
    OUTPUT_DIR = args.output_dir
    API_DB_OUTPUT_FILE = args.api_db_output_file
    
    # API
    PARSER = args.parser_id
    PROVIDER = args.api_provider

    print("--- Arguments ---")
    print(f"\t Input directory: {INPUT_DIR}")
    print(f"\t Input file: {INPUT_FILE}")
    print(f"\t Output directory: {OUTPUT_DIR}")
    print(f"\t Output file (API DB): {API_DB_OUTPUT_FILE}")
    print(f"\t Parser id: {PARSER}")
    print(f"\t API Provider: {PROVIDER}")

    print(f"--- Loading input file from {os.path.join(INPUT_DIR,INPUT_FILE)} ---")
    source = dm.load_local_file_as_json(os.path.join(INPUT_DIR,INPUT_FILE))

    print(f"--- Extracting data from {INPUT_FILE} ---")
    parser = create_parser(source=source,parser=PARSER, provider=PROVIDER)

    if type(parser) is None:
        print(f"ERROR: Unable to process OpenAPI file: {INPUT_FILE}")
    else:
        print(f"--- Saving API DB ---")
        dm.save_data_as_json_array(parser.get_data(),os.path.join(OUTPUT_DIR,API_DB_OUTPUT_FILE))

        print(f"--- API DB saved as {API_DB_OUTPUT_FILE} ---")
        print(f"--- \tTota API calls: {len(parser.get_data())}")


