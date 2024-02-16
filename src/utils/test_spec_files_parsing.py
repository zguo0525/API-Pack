import os
from tools.spec_file_parser import GDPSParser
import tools.data_manger as dm

INPUT_DIR = "./data/input" # Note: place the original OpenAPI spec file to be parsed into this directory
INPUT_FILE = "GDPS_REST_API_V4R6GM-4.6.0_api_calls.json" 

print(f"---Loading API specification {INPUT_FILE}---")
source = dm.load_local_file_as_json(os.path.join(INPUT_DIR,INPUT_FILE)) # Note: replace 'INPUT_FILE' with the file name that you want to test

print(f"---Creating parser for {INPUT_FILE}---")
my_parser = GDPSParser() # Note: replace 'COS_Parser' with the paser class that you want to test
my_parser.parse_data(source)

print(f"---Print parser output---")
print(my_parser.get_data())
