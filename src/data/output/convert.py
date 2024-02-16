import glob
import json
import os
from tqdm import tqdm

#lang = "curl"

for lang in ["curl", "go", "java", "javascript", "libcurl", "node", "php", "python", "ruby", "swift"]:
    
    # Directory where the grouped files will be saved
    output_dir = f'./final_dataset_custom/cleaned_{lang}_converted'
    os.makedirs(output_dir, exist_ok=True)
    
    # Read all files matching the pattern
    files = glob.glob(f'./final_dataset_custom/cleaned_{lang}/{lang}_batch_*_final.json')
    
    # List to hold all combined data
    combined_data = []
    
    # Process each file and combine the data
    for file_name in tqdm(files):
        with open(file_name, 'r') as file:
            data = json.load(file)
            combined_data.extend(data)
    
    print("length of data1:", len(combined_data))
    
    # Read all files matching the pattern
    files = glob.glob(f'{output_dir}/*.json')
    
    # List to hold all combined data
    combined_data = []
    
    # Process each file and combine the data
    for file_name in tqdm(files):
        with open(file_name, 'r') as file:
            data = json.load(file)
            combined_data.extend(data)
    
    print("length of data2:", len(combined_data))
    
    
    # # Dictionary to hold grouped data
    # grouped_data = {}
    
    # # Group the combined data by api_name
    # for entry in tqdm(combined_data):
    #     # Extract api_name
    #     api_name = entry['api_data']['api_name']
    #     # Add entry to the corresponding group
    #     if api_name not in grouped_data:
    #         grouped_data[api_name] = []
    #     grouped_data[api_name].append(entry)
    
    # # Save grouped data into separate files
    # for api_name, entries in tqdm(grouped_data.items()):
    #     with open(os.path.join(output_dir, f'{api_name.replace("/", "-").replace(" ", "-")}.json'), 'w') as file:
    #         json.dump(entries, file, indent=4)
    
    # print(f'Grouped data saved in {output_dir}')