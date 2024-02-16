# Requirements: pip install pystache
import pystache
import os
import json

def format_list_from_template(template_path, data):    
    output = ""
    try:
        lines = ""
        with open (template_path, "r") as source:
            lines = source.readlines()
        template = ''.join(lines)
    except OSError:
        print ("Could not open/read template:", template_path)
    try:
        my_render = pystache.Renderer(string_encoding='utf8')
        output = my_render.render(template,data)
    except:
        print ("The rendering process failed")
    return output

def format_from_template(template_path, data_point):    
    output = ""
    try:
        lines = ""
        with open (template_path, "r") as source:
            lines = source.readlines()
        template = ''.join(lines)
    except OSError:
        print ("Could not open/read template:", template_path)
    try:
        my_render = pystache.Renderer(string_encoding='utf8')
        output = my_render.render(template,data_point)
    except:
        print ("The rendering process failed")
    return output

def apply_format(data:[], template_path:str = ""):
    output = []
    for datapoint in data:
        datapoint["instruction"] = datapoint["best_instruction"]["candidate"]
        code = format_from_template(template_path = template_path, data_point = datapoint)
        new_datapoint = {
            "code": f"{code}",
            "instruction": datapoint['best_instruction'],
            "instruction_candidates": datapoint['instruction_candidates'],
            "api_data": {
                "api_name": f"{datapoint['api_name']}",
                "framework": f"{datapoint['framework']}",
                "api_description": f"{datapoint['api_description']}",
            },
            "api_call_data":{
                "api_call": f"{datapoint['api_call']}",
                "lang": f"{datapoint['lang']}",
                "functionality": f"{datapoint['functionality']}",
                "api_arguments": datapoint['api_arguments'],
                "description": datapoint['description'],
                "domain": datapoint['domain'],
                "path": datapoint['path']
            }
        }
        output.append(new_datapoint)
    return output

def apply_default_format(data:list):
    custom_data = []
    for datapoint in data:
        custom_datapoint = {
            "input": datapoint['best_instruction']['candidate'],
            "output": datapoint['api_call']
        }
        # print(custom_datapoint) # TEST
        custom_data.append(custom_datapoint)
    return custom_data

   
