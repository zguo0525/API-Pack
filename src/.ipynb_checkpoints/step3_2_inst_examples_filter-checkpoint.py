import os
import json
import argparse
import glob
from tqdm import tqdm
from tools.data_manger import load_local_file_as_json
from langdetect import detect, LangDetectException

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def check_api_dict(api_dict):
    # Check if the dictionary is empty
    if len(api_dict) == 0:
        print("The dictionary has no items.")
        return False

    # Check if the API name is "Simple Inventory API"
    if "Simple " in api_dict.get('API name'):
        print("The API name is 'Simple Inventory API'.")
        return False

    non_english = []
    empty_count = 0

    for key, value in api_dict.items():
        if not value or len(value) == 0:  # Check if the value is empty
            empty_count += 1
        elif not is_english(value):  # Check if the value is non-English
            non_english.append(key)

    if empty_count >= 2:
        print("Two or more entries are empty.")
        return False
    if len(non_english) >= 3:
        print("Non-English text found in the following fields:", ', '.join(non_english))
        return False
    return True

def delete_files(base_path, example_file, api_version):
    example_path = os.path.join(base_path, 'temporal_files', 'inst_examples', api_version, example_file)
    db_file = example_file.replace('_inst_exa.json', '.json')  # Modify this based on actual naming convention
    db_path = os.path.join(base_path, 'output', 'api_dbs', api_version, db_file)

    #print(example_path, db_path)

    if os.path.exists(example_path):
        os.remove(example_path)
        print(f"Deleted file: {example_path}")

    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted file: {db_path}")

def delete_non_overlap_files(base_path, api_version):
    # Gather all filenames in the inst_examples directory and transform them to match the api_dbs naming convention
    inst_example_files = set(os.path.basename(file).replace('_inst_exa.json', '.json') for file in glob.glob(os.path.join(base_path, f'temporal_files/inst_examples/{api_version}/*.json')))

    # Iterate over all JSON files in the api_dbs directory
    db_json_files = glob.glob(os.path.join(base_path, f'output/api_dbs/{api_version}/*.json'))

    for db_file in tqdm(db_json_files):
        db_file_name = os.path.basename(db_file)

        if db_file_name not in inst_example_files:
            if os.path.exists(db_file):
                os.remove(db_file)
                print(f"Deleted file: {db_file}")

if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description="Process the ins_ex_path")
    parser.add_argument("--data_source", type=str, help="Path to the instruction examples JSON file")
    parser.add_argument("--base_path", type=str, help="Base path")
    args = parser.parse_args()
    
    api_version = args.data_source
    base_path = args.base_path
    
    json_files = glob.glob(os.path.join(base_path, f'temporal_files/inst_examples/{api_version}/*.json'))
    origin_files = len(json_files)
    
    for json_file in tqdm(json_files):
        
        instruction_examples_data = load_local_file_as_json(file_path=json_file)['list']
        
        if len(instruction_examples_data) == 0:
            os.remove(json_file)
            print(f"Deleted file: {json_file}")

        for instruction_example_data in instruction_examples_data:
            if not check_api_dict(instruction_example_data):
                delete_files(base_path, os.path.basename(json_file), api_version)
                break  # Stop the loop after deleting the files

    delete_non_overlap_files(base_path, api_version)

    json_files = glob.glob(os.path.join(base_path, f'temporal_files/inst_examples/{api_version}/*.json'))
    print("origin files:", origin_files)
    print("remaining files:", len(json_files))