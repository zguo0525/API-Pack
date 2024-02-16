import os
import json
from data_manger import load_local_file_as_json

INPUT_PATH = "/Users/amezasor/Projects/llm4tools/src/data/input"
INPUT_FILE = "Watch Folders API-4.2.0" # DO NOT INCLUDE EXTENSION!
OUTPUT_FILE = f'{INPUT_FILE.replace(" ","")}_validated.json'

data = load_local_file_as_json(os.path.join(INPUT_PATH,INPUT_FILE+".json"))

# Identify deprecated paths
paths_to_delete = set()
for path in data["paths"]:
    for key in data["paths"][path]:
        if "deprecated" in data["paths"][path][key] or "servers" in data["paths"][path][key] :
            paths_to_delete.add(path)
            print(f"{path} will be deleted")
            
# Delete deprecated paths
for path in paths_to_delete:
    del data["paths"][path]

# Save json to file
with open(os.path.join(INPUT_PATH,OUTPUT_FILE), 'w') as f:
    json.dump(data, f)


