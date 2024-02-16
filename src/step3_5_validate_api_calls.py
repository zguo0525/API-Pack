import os
import argparse

from tools import data_manger as dm
from tools import api_call_validator as cv

"""
EXAMPLES:
python validate_api_calls.py
python validate_api_calls.py --instructions_dir './data/temporal_files/instruction_files/ibm/' --instructions_cleaned_dir './data/temporal_files/instruction_files_cleaned/ibm/' > ./data/logs/api_calls_validation_ibm.txt
python validate_api_calls.py --instructions_dir './data/temporal_files/instruction_files/api_gurus/' --instructions_cleaned_dir './data/temporal_files/instruction_files_cleaned/api_gurus/' > ./data/logs/api_calls_validation_api_gurus.txt
"""

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructions_dir",
                        type=str,
                        required=False,
                        default="./data/temporal_files/instruction_files/", 
                        help="Directory with the input files for the experiment.")
    parser.add_argument("--instructions_cleaned_dir",
                    type=str,
                    required=False,
                    default="./data/temporal_files/instruction_files_cleaned/", 
                    help="Directory with the input files for the experiment.")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()

    # OUTPUT
    INSTRUCTIONS_DIR = args.instructions_dir
    INSTRUCTIONS_CLEANED_DIR = args.instructions_cleaned_dir


    c_wrong_lang = 0
    c_no_method = 0
    c_invalid_url = 0
    c_original_dp = 0
    c_valid = 0

    print("= API CALL VALIDATION =")
    for filename in os.listdir(INSTRUCTIONS_DIR):
        f = os.path.join(INSTRUCTIONS_DIR, filename)

        valid_data = []
        invalid_data = []

        if os.path.isfile(f) and filename.endswith(".json"):
            data = dm.load_local_file_as_json(os.path.join(INSTRUCTIONS_DIR,filename))

            print(f"API File: {filename}") # TEST
            print(f"{len(data)} datapoints") # TEST

            for datapoint in data:
                is_valid_dp = True

                if datapoint["lang"] not in ["curl", "python", "java", "node", "go", "ruby", "php", "swift", "javascript xhr", "libcurl"]:
                    continue

                # Check api_call matches lang
                if (datapoint["lang"] == "curl" and "curl" not in datapoint["api_call"].lower()) or \
                    (datapoint["lang"] == "python" and "http.client.HTTPSConnection".lower() not in datapoint["api_call"].lower()) or \
                    (datapoint["lang"] == "java" and "HttpResponse<String>".lower() not in datapoint["api_call"].lower()) or \
                    (datapoint["lang"] == "node" and "const http = require".lower() not in datapoint["api_call"].lower()) or \
                    (datapoint["lang"] == "go" and "package main" not in datapoint["api_call"].lower()) or \
                    (datapoint["lang"] == "ruby" and "require 'uri'" not in datapoint["api_call"].lower()) or \
                    (datapoint["lang"] == "php" and "php".lower() not in datapoint["api_call"].lower()) or \
                    (datapoint["lang"] == "swift" and "import Foundation".lower() not in datapoint["api_call"].lower()) or \
                    (datapoint["lang"] == "javascript xhr" and "const data =".lower() not in datapoint["api_call"].lower()) or \
                    (datapoint["lang"] == "libcurl" and "CURL *hnd = curl_easy_init()".lower() not in datapoint["api_call"].lower()):
                    is_valid_dp = False
                    c_wrong_lang += 1
                    
                    print(f"= Lang issue =") # TEST
                    print(f"lang: {datapoint['lang']}") # TEST
                    print(f"api call: {datapoint['api_call']}") # TEST

                # Check a valid method (GET, POST, PUT, DELETE, PATCH, HEAD) is in API call
                # if datapoint["lang"] = "curl":
                if (datapoint["method"].lower() not in datapoint["api_call"].lower()):
                    is_valid_dp = False
                    c_no_method += 1
                    
                    print(f"= Method issue =") # TEST
                    print(f"api call: {datapoint['api_call']}") # TEST

                # Check URL is valid; 
                    # With protocol: curl_shell, node_request, go_native, ruby_native, php_curl, swift_nsurlsession
                    # Without protocol: python_python3
                urls = cv.extract_url_from_str(datapoint["api_call"])
                if len(urls) > 0 :
                
                    if datapoint["lang"] in ["python"]:
                        url_validation = cv.validate_url(url = urls[0], include_protocol = False)
                    else:
                        url_validation = cv.validate_url(url = urls[0])

                    if url_validation is None:
                        is_valid_dp = False
                        c_invalid_url += 1

                        print(f"= URL issue =") # TEST
                        print(f"api call: {datapoint['api_call']}") # TEST

                if is_valid_dp:
                    c_valid += 1
                    valid_data.append(datapoint)
                else:
                    invalid_data.append(datapoint)

                c_original_dp += 1

            # Save a new cleaned data file only if valid dps exist
            if len(valid_data) >0 : 
                print(f"--- Saving API dataset cleaned API dataset ---")
                if not os.path.exists(INSTRUCTIONS_CLEANED_DIR):
                    # Create the folder
                    os.makedirs(INSTRUCTIONS_CLEANED_DIR)
                dm.save_data_as_json_array(valid_data,os.path.join(INSTRUCTIONS_CLEANED_DIR,filename))
                print(f"\tAPI dataset with instructions after api call validation saved as {os.path.join(INSTRUCTIONS_CLEANED_DIR,filename)}")

    # Instruction files with valid data
    print("= FILES WITH VALID DATA =")
    c_valid_files = 0
    for filename in os.listdir(INSTRUCTIONS_CLEANED_DIR):
        f = os.path.join(INSTRUCTIONS_CLEANED_DIR, filename)
        if os.path.isfile(f) and filename.endswith(".json"):
            data = dm.load_local_file_as_json(os.path.join(INSTRUCTIONS_CLEANED_DIR,filename))
            if len(data) > 0:
                c_valid_files += 1

    print("=====================================================================")
    print("= API call validation process =")

    print("= Issues =")
    print(f"\t{c_wrong_lang} datapoints with api call not matching the language")           
    print(f"\t{c_no_method} datapoints without method")
    print(f"\t{c_invalid_url} datapoints with an invalid url example")

    print("= Totals =")
    print(f"\t{c_original_dp} original datapoints")
    print(f"\t{c_valid} valid datapoints")
    print(f"\t{c_valid_files} files with valid data after api call validation")