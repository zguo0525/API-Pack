import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def load_local_file_as_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(load_str_as_json(line))
    return data 

def load_str_as_json(json_line:str):
    return json.loads(json_line)

# Load local json array as a dict
def load_local_file_as_json(file_path:str):
    path = Path(file_path)
    if path.is_file():
        # file = open(file_path,"r", encoding='utf-8')
        file = open(file_path,"r")
        try:
            return json.load(file)
        except ValueError as e:
            error_msg = "".join(["{", f'"error":"Invalid json format - {e}"', "}"])
            return json.loads(error_msg)
    # Invalid file path
    error_msg = "".join(["{", f'"error":"The path {file_path} is invalid"', "}"])
    return json.loads(error_msg)

# Load local json as json object
def load_web_file_as_json(url:str):
    req = Request(url)
    try:
        response = urlopen(req)
        result = json.loads(response.read())
    except HTTPError as e:
        # print('Error code: ', e.code)
        result = json.loads("".join(["{", f'"error":"{e.code}"', "}"]))
    except URLError as e:
        # print('Reason: ', e.reason)
        result = json.loads("".join(["{", f'"reason":"{e.reason}"', "}"]))
    except ValueError as e:
        result = json.loads("".join(["{", f'"error":"Invalid json format - {e}"', "}"]))
    return result

# TO DO: What should be returned as error?
def load_txt_file(file_path:str):
    path = Path(file_path)
    if path.is_file():
        with open(file_path, 'r') as reader:
            file = reader.read()
        return file
    # Invalid file path
    error_msg = ""
    return error_msg

# TO DO: Add invalid pad and json form validations
def save_data_as_json_array(data:list,file_path:str):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4, ensure_ascii=False)

def save_text_file(data:list,file_path:str):
    for element in data:
        with open(file_path, 'a') as f:
            f.write(str(element))

def save_as_jsonl(data:list, output_file_path:str):
    with open(output_file_path, 'w') as jsonl_output:
        for datapoint in data:
            jsonl_output.write(json.dumps(datapoint) + "\n")