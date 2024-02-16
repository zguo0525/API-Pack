import json
import os
import re
import random
import time
from tqdm import tqdm

def get_questions(question_file):
    """
    Loads a JSON file and returns its content as a data structure.

    :param question_file: The path to the JSON file.
    :return: The content of the JSON file.
    """
    # Attempt to load the questions file
    try:
        with open(question_file, "r", encoding='utf-8') as ques_file:
            data = json.load(ques_file)
        return data
    except FileNotFoundError:
        print(f"The file {question_file} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"The file {question_file} is not a valid JSON file.")
        return None

def list_json_files(directory_path):
    """
    Returns a list of full paths to .json files in the given directory.

    :param directory_path: The path to the directory in which to search for .json files.
    :return: A list of strings, where each string is a full path to a .json file.
    """
    # Get a list of files in the directory
    files_in_directory = os.listdir(directory_path)
    
    # Filter out all non-.json files
    json_files = [file for file in files_in_directory if file.endswith('.json')]
    
    # Get full paths
    json_files_full_path = [os.path.join(directory_path, file) for file in json_files]

    return json_files, json_files_full_path

def find_string_after_pattern(text, pattern="###Output:"):
    """
    Finds and returns the substring that comes after a given pattern.

    :param text: The text to search within.
    :param pattern: The pattern to search for.
    :return: The substring found after the pattern. None if the pattern is not found.
    """
    # Search for the pattern and capture all characters after it
    match = re.search(pattern + r'(.*)', text, re.DOTALL)
    
    # If a match is found, return the capturing group 1
    # which contains everything after the pattern
    if match:
        return match.group(1).strip()  # .strip() removes leading/trailing whitespace

    # If the pattern is not found, return None
    return None

def save_json(data, file_path, filter_length=0):
    """
    Saves the given data to a JSON file at the specified file path.

    :param data: The data structure to save (usually a dict or a list).
    :param file_path: The path, including the filename, where the JSON file should be saved.
    :return: None
    """
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Attempt to save the data to the JSON file
    if len(data) <= filter_length:
        print("no data for this API")
    else:
        try:
            with open(file_path, "w", encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
            print(f"Data successfully saved to {file_path} with length {len(data)}")
        except IOError as e:
            print(f"An error occurred while writing the file: {e}")
        except TypeError as e:
            print(f"An error occurred with the data type being saved: {e}")

def process_string(s):
    # Convert the string to lower case and find the position of "query:"
    query_index = s.lower().find("query:")
    
    # If "query:" is found in the string
    if query_index != -1:
        # Extract the part of the string after "query:"
        # The length of "query:" is 6, so add 6 to the index to start after it
        return s[query_index + 6:].strip()
    
    # If "query:" is not found, return the original string or an empty string,
    # depending on your requirement
    return s

def process_files(file_names, full_files, data_source, additional_filter=True):
    
    # Process each file
    for file_name, file_path in zip(file_names, full_files):
        
        data = get_questions(file_path)
        total_data = []
        total_data_second_best_instruction = []
        
        for entry in data:
            api_name = entry["api_data"]["api_name"]
            language = entry["api_call_data"]["lang"].lower()
            
            # # filter the programming language for curl only
            # if language != "curl":
            #     continue

            # in instruction is None, skip the current datapoint
            if entry["instruction"]['candidate'] == "None":
                continue

            entry["instruction"] = process_string(entry["instruction"]['candidate'])
            
            # Initialize instruction_test to be empty initially
            entry["instruction_test"] = ""
            
            # Iterate over the range from 0 to 4 inclusive
            for i in range(0, 5):
                try:
                    # Process the candidate instruction at index i
                    processed_candidate = process_string(entry["instruction_candidates"][i]['candidate'])
                
                    # Check if the processed candidate is different from the given instruction
                    if entry["instruction"] in processed_candidate or processed_candidate in entry["instruction"]:
                        continue  # If they are too similar, skip the rest of the loop and continue with the next number
                    else:
                        if entry["instruction_candidates"][i]['mistral-7b label'] == "#GOOD INST#":
                            # If they are different, assign the processed candidate to instruction_test and break out of the loop
                            entry["instruction_test"] = processed_candidate
                            break
                except:
                    continue

            # # after looking at all the candidates, if not for instruction_test, skip this data point
            # if entry["instruction_test"] == "":
            #     continue
            
            entry["input"] = ""
            entry["output"] = find_string_after_pattern(entry["code"], pattern="###Output:")
            
            # String to check against
            code_marker = "\n<<<code>>>:"
            # Check if the end of 'entry["output"]' matches 'code_marker'
            if entry["output"].endswith(code_marker):
                # Remove 'code_marker' from the end of 'entry["output"]'
                entry["output"] = entry["output"][:-len(code_marker)]

            # lang_marker = "\n<<<lang>>>:curl"
            # if entry["output"].lower().endswith(lang_marker):
            #     # Remove 'code_marker' from the end of 'entry["output"]'
            #     entry["output"] = entry["output"][:-len(lang_marker)]

            if additional_filter:
                # Define the pattern to search for the api_call
                pattern = r'<<<api_call>>>:\s*(.*?)(?=\n|<<<)'
                # Search for the pattern in the entry_output
                match = re.search(pattern, entry["output"])
                api_call_text = match.group(1).strip()
                only_special_characters = bool(re.match(r'^[^a-zA-Z0-9]+$', api_call_text))
                if only_special_characters or len(pattern)<=10:
                    continue

            if entry["instruction_test"] != "" and entry["api_data"]["api_description"] != "" and entry["api_data"]["api_description"] != " ":
                # total_data_second_best_instruction.append(filtered_data_test)
                filtered_data = {"api_name": entry["api_data"]["api_name"],
                            "api_description": entry["api_data"]["api_description"],
                            "api_call_data": entry["api_call_data"],
                             "instruction": entry["instruction"],
                             "instruction_test": entry["instruction_test"],
                             "input": entry["input"], 
                             "output": entry["output"].replace("<<<", "**").replace(">>>", "**")}
                total_data.append(filtered_data)  # Add to total data list

        # Save the individual processed file
        if entry["api_data"]["api_name"] != "IBM Maximo Health, Predict and HP Utilities API" and entry["api_data"]["api_name"] != "Maximo RESTful API":
            data_source = data_source.replace('_converted', '')
            if not os.path.exists(f"./instr_data/{data_source}"):
                os.makedirs(f"./instr_data/{data_source}")
            save_json(total_data, f"./instr_data/{data_source}/{file_name}", 1) 

if __name__ == "__main__":

    langs = ["curl", "go", "java", "javascript", "libcurl", "node", "php", "python", "ruby", "swift"]
    data_srouces = [f"cleaned_{lang}_converted" for lang in langs]
    
    for data_source in tqdm(data_srouces):
        # Path to the directory you want to search
        directory_path = f"/gpfs/u/home/SIFA/SIFAzhnu/scratch/llm4tools_data/llm4tools/src/data/output/final_dataset_custom/{data_source}"
        file_names, full_files = list_json_files(directory_path)
        
        process_files(file_names, full_files, data_source, additional_filter=True)

        data_source = data_source.replace('_converted', '')
        directory_path = f"./instr_data/{data_source}"
        file_names, full_files = list_json_files(directory_path)
        # this data is duplicated somehow
        try:
            file_names.remove("factsheets_generated_api_db_instructions_final.json")
        except:
            print("not found")
    
        total_data = []
        for file_name in file_names:
            data = get_questions(f"{directory_path}/{file_name}")
            # Add a unique identifier to each data point
            for i, item in enumerate(data):
                # Creating a unique identifier. Here, we use the combination of file_name and index.
                item['unique_id'] = f"{file_name}_{i}"
            total_data.extend(data)

        # Total Training Data
        total_training = total_data.copy()
        
        # Calculate API counts across the entire dataset
        api_counts = {}
        for data in total_training:
            api = data.get('api_name')  # Replace 'api' with the actual key for API name in your data
            if api:
                api_counts[api] = api_counts.get(api, 0) + 1
        
        # Filter APIs with more than 4 counts
        apis_with_count_more_than_4 = {api for api, count in api_counts.items() if count > 4}

        ##################################################
        # First Level Testing
        ##################################################
        total_testing_level_1 = [data for i, data in enumerate(total_training[:20000]) 
                                 if data.get('instruction_test') != "None" and 
                                    data.get('api_name') in apis_with_count_more_than_4][:1000]
        
        ##################################################
        # Second Level Testing
        ##################################################
        total_testing_level_2 = []
        
        # Create a set to keep track of selected APIs
        selected_apis_for_level_2 = set()
        
        # Iterate through the first 20000 entries of the training data.
        for data in total_training[:20000]:
            api = data.get('api_name')
            # Check if the API meets the count criteria and hasn't been selected yet
            if api_counts.get(api, 0) > 3 and api not in selected_apis_for_level_2:
                # Add this data point to the Level 2 testing dataset.
                total_testing_level_2.append(data)
                # Add the API to the set of selected APIs
                selected_apis_for_level_2.add(api)
                # Stop if we have collected 1000 unique APIs
                if len(selected_apis_for_level_2) == 1000:
                    break
        
        # Remove ONLY the selected data points from the training dataset.
        selected_ids = set([data['unique_id'] for data in total_testing_level_2])  # Assuming each data point has a unique 'id'
        total_training = [data for data in total_training if data['unique_id'] not in selected_ids]

        # Get the set of APIs used in Level 2 testing data
        apis_used_in_level_2 = set([data['api_name'] for data in total_testing_level_2])

        ##################################################
        # third Level Testing
        ##################################################
        total_testing_level_3 = []
        apis_used_in_level_3 = set()
        
        # Iterate through the training data starting from the 80000th entry to collect Level 3 testing data
        for data in total_training[80000:]:
            # Check if the API is not one of those used in Level 2
            if data['api_name'] not in apis_used_in_level_2:
                # Add this data point to the Level 3 testing dataset
                total_testing_level_3.append(data)
                # Add the API to the set of APIs used in Level 3
                apis_used_in_level_3.add(data['api_name'])
                # If we have collected 1000 data points, break the loop
                if len(total_testing_level_3) >= 1000:
                    break
        
        # Now filter the training data to remove data points with APIs used in Level 3
        total_training = [data for data in total_training if data['api_name'] not in apis_used_in_level_3]

        # Save the individual processed file
        save_json(total_data, f"{directory_path}/total_data_{data_source}.json", 0)
        save_json(total_training, f"{directory_path}/total_training_{data_source}.json", 0)
        save_json(total_testing_level_1, f"{directory_path}/total_testing_{data_source}_level_1.json", 0)
        save_json(total_testing_level_2, f"{directory_path}/total_testing_{data_source}_level_2.json", 0)
        save_json(total_testing_level_3, f"{directory_path}/total_testing_{data_source}_level_3.json", 0)