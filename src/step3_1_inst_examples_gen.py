import os
import json
import argparse
import random

import tools.data_manger as dm
#import tools.instruction_generator as ig

"""
EXAMPLE: python step3_1_inst_examples_gen.py --api_db_file activity-tracker_api_db.json --inst_exa_file  activity-tracker_api_db_inst_exa.json
EXAMPLE: python step3_1_inst_examples_gen.py --api_db_file api_gurus_1forge.com_0.0.1_swagger_api_calls_api_db.json --inst_exa_file  api_gurus_1forge.com_0.0.1_swagger_api_calls_api_db_inst_exa.json --api_db_dir ./data/output/api_dbs/api_gurus --inst_exa_dir ./data/temporal_files/inst_examples/api_gurus
"""

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_db_dir",
                        type=str,
                        required=False,
                        default="./data/output/api_dbs", 
                        help="Directory in which the API DB file is located.")
    parser.add_argument("--api_db_file",
                        type=str,
                        required=True,
                        help="The name of API DB file in json format, it cannot be empty.")
    parser.add_argument("--inst_exa_dir",
                    type=str,
                    required=False,
                    default="./data/temporal_files/inst_examples", 
                    help="Directory to save instruction exmaple files.")
    parser.add_argument("--inst_exa_file",
                type=str,
                required=True,
                help="The name of the in-context example file in json format, it cannot be empty.") 
    return parser.parse_args()

def generate_example_candidates(api_name:str, functionality:str, description:str, endpoint:str):
        examples = []
        inst_candidate_functionality = ""
        inst_candidate_description = ""
        inst_candidate_endpoint = ""
        if len(functionality)>0 and "example request" not in functionality.lower():
            requests = ["Please tell me how to","Please show me how to","Hey, tell me how to","I need to know how to", "Give me an example of how to"]
            request = random.sample(requests, 1)[0]
            inst_candidate_functionality = f"{request} {functionality.lower()} with the {api_name}."
        if len(description)>len(endpoint):
            inst_candidate_description = f"I need to {description.lower()} with the {api_name}. Please help me with that."
        
        if len(endpoint)>0:
            inst_candidate_endpoint = f"Please give me an example of how to use the endpoint {endpoint} from {api_name}."  

        if len(inst_candidate_functionality)>0:
            examples.append({"API name": f"{api_name}",
                        "functionality": f"{functionality}", 
                        "description": f"{description}", 
                        "endpoint":f"{endpoint}", 
                        "output": f"{inst_candidate_functionality}"})    
        if len(inst_candidate_description)>0:
            examples.append({"API name": f"{api_name}",
                        "functionality": f"{functionality}", 
                        "description": f"{description}", 
                        "endpoint":f"{endpoint}", 
                        "output": f"{inst_candidate_description}"})
        if len(inst_candidate_endpoint)>0:
            examples.append({"API name": f"{api_name}",
                        "functionality": f"{functionality}", 
                        "description": f"{description}", 
                        "endpoint":f"{endpoint}", 
                        "output": f"{inst_candidate_endpoint}"})  
        return examples

def generate_examples(data:[]):
    all_examples = []
    examples = []
    print(f"Datapoints: {len(data)}")
    for datapoint in data:
        examples.clear()
        examples = generate_example_candidates(datapoint["api_name"],
                                                    datapoint["functionality"],
                                                    datapoint["description"],
                                                    datapoint["endpoint"])
        print(f"Examples generated per datapoint: {len(examples)}") # TEST
        all_examples.extend(examples)
        # examples.append({"API name": f"{datapoint['api_name']}",
        #                  "functionality": f"{datapoint['functionality']}", 
        #                  "description": f"{datapoint['description']}", 
        #                  "endpoint":f"{datapoint['endpoint']}", 
        #                  "output": f"{inst_candidate}"})
    print(f"All examples: {len(all_examples)}") # TEST
    output = {"list": all_examples}
    return output
    
def remove_duplicates(data:[]):
    
    if len(data)>0:
        if len(data[0]["endpoint"])>0: key ='endpoint'
        else: key ='path'
    unique = list({ each[key] : each for each in data }.values())
    return unique

if __name__=="__main__":
    args = parse_arguments()

    # INPUT
    API_DB_DIR = args.api_db_dir
    API_DB_FILE = args.api_db_file
    EXAMPLES_DIR = args.inst_exa_dir
    INST_FILE_NAME = args.inst_exa_file

    print(f"--- Loading API DB file from {os.path.join(API_DB_DIR,API_DB_FILE)} ---")
    data = dm.load_local_file_as_json(os.path.join(API_DB_DIR,API_DB_FILE))
    print(f"Total datapoints: {len(data)}")

    data_no_duplicates = remove_duplicates(data)
    # print(data_no_duplicates)
    # print(type(data_no_duplicates))
    print(f"No duplicated datapoints: {len(data_no_duplicates)}")
    
    if len(data_no_duplicates)>=3:
        # Use the sample() method to select 3 examples without duplicates from the api_db file
        selected_data = random.sample(data_no_duplicates, 3)
        print(f"{len(selected_data)} datapoints selected.")
        output = generate_examples(data=selected_data)
    else:
        output = generate_examples(data=data_no_duplicates)
        
    # print(output["list"]) # TEST

    print(f"--- Saving API in context examples file at {os.path.join(EXAMPLES_DIR,INST_FILE_NAME)} ---")
    dm.save_data_as_json_array(output,os.path.join(EXAMPLES_DIR,INST_FILE_NAME))

         
    

