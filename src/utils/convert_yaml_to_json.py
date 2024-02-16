import os
import yaml
import json

def yaml_to_json(input, output):
    try:
        with open(input, 'r') as file:
            configuration = yaml.safe_load(file)
    except:
        print(f"ERROR: Cannot load {input}")

    try:
        with open(output, 'w') as json_file:
            json.dump(configuration, json_file, indent=4, ensure_ascii=False)
    except:
        print(f"ERROR: Could not save {output}")


def split_name_ext(path):
    temp = path.rsplit(".")
    name = temp[0:-1]
    name = '.'.join(str(e) for e in name)
    extension = temp[len(temp)-1]
    return name,extension

def clean_filename(name, forbidden_chars = ' %:/,\\[]<>*?=#{}()"'): #let hyphens and underscores in the urls
    """Ensures each file name does not contain forbidden characters and is within the character limit"""
    # The file name limit should be around 240 for linux eviroments.
    filename = ''.join([x if x not in forbidden_chars else '_' for x in name])
    if len(filename) >= 200:
        filename = filename[:200]
    return filename

if __name__=="__main__":
    main = "/Users/amezasor/Projects/openapi-snippet/test/input_raw/"
    input_path = "/Users/amezasor/Projects/openapi-snippet/test/input_raw/api_gurus/"
    output_path = "/Users/amezasor/Projects/openapi-snippet/test/input_raw/api_gurus_json"

    full_paths = []
    for root, d_names, f_names in os.walk(input_path):
        # print(root, d_names, f_names)
        for f in f_names:
            full_paths.append(os.path.join(root, f))
    # print(f"Total files: {len(full_paths)}")

    for path in full_paths:
        print(f"Processing: {path}")
        full_url,extension = split_name_ext(path)
        str_list = full_url.split(main)
        output_string = "".join(str_list)
        name = clean_filename(output_string)
        output = os.path.join(output_path,name + ".json")
        # print("The output string is:", output)
        yaml_to_json(path,output)


