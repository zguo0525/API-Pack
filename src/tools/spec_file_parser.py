import os
import json

from abc import ABC, abstractmethod
from schemas.dicts import prog_lang
from schemas.types import DataPoint

"""
Note: Use 'OpenAPIParser' class as a blueprint to build a custom OpenAPI parser
"""
class OpenAPIParser(ABC):
    def __init__(self):
        self.data = None
    
    @abstractmethod
    def set_data(self,data):
        self.data = data

    @abstractmethod
    def get_data(self):
        return self.data
    
    @abstractmethod
    def parse_data(self, source, provider):
        pass

    def add_datapoint(self, dt:DataPoint):
        new_dict = {"api_call":dt.api_call, # *
                    "api_name":dt.api_name,
                    "api_provider":dt.api_provider,
                    "endpoint":dt.endpoint,
                    # "explanation":dt.explanation, 
                    # "code":dt.code, # Repeated
                    "framework":dt.framework,
                    "functionality":dt.functionality, # *
                    "api_arguments":dt.api_arguments,
                    # "python_environment_requirements":dt.python_environment_requirements, 
                    # "example_code":dt.example_code, # Repeated
                    # "dataset":dt.dataset,
                    # "accuracy":dt.accuracy,
                    "description":dt.description,
                    "path": dt.path, 
                    "method": dt.method, 
                    "lang": dt.lang, # *
                    "domain": dt.domain, # *
                    "api_description": dt.api_description, 
                    "api_license": dt.api_license} # *
        return new_dict
    
"""
'GenParser': Use this parser for APIs with generated calls
"""
class GenParser(OpenAPIParser):
    def __init__(self):
        super().__init__()

    def set_data(self,data):
        super().set_data(data)

    def get_data(self):
        return super().get_data()
    
    def parse_data(self, source, provider):
        if len(provider) <= 0: 
            _provider = source['info']['x-providerName'] if "x-providerName" in source['info'] else ""
        else:
            _provider = provider
        _api_name = source['info']['title'] if  "title" in source['info'] else ""
        _api_description = source['info']['description'] if  "description" in source['info'] else ""
        _api_license = source['info']['license'] if  "license" in source['info'] else ""

        data = []
        for path in source["paths"]:
            for method in source["paths"][path]:
                if method not in ["x-gdps-restrict-flavors", "x-gdps-exclude-topologies"]:
                    _path = path
                    _method = method
                    _operation_id = ""
                    _summary = ""
                    _description = ""
                    _api_call = ""
                    _lang = ""
                    _domain = []

                    if "operationId" in source["paths"][path][method]: _operation_id = source['paths'][path][method]['operationId']
                    if "summary" in source["paths"][path][method]: _summary = source['paths'][path][method]['summary']
                    if "description" in source["paths"][path][method]: _description = source['paths'][path][method]['description']
                    if "tags" in source["paths"][path][method]: _domain = source['paths'][path][method]['tags']
                    if "api_calls" in source["paths"][path][method]: 
                        for snippet in source['paths'][path][method]['api_calls']:
                            _lang = prog_lang[snippet['id']]
                            _api_call = snippet['content']

                            # As all the snippets generated are a list,
                            # we add a record per language, same api call might be in different languages
                            datapoint = DataPoint( 
                                                api_call = _api_call, 
                                                api_name = _api_name,
                                                api_provider = _provider, 
                                                endpoint = _operation_id,
                                                # explanation = [],
                                                # code = _api_call,
                                                framework = _provider,
                                                functionality = _summary,
                                                api_arguments = {},
                                                # python_environment_requirements = [],
                                                # example_code = _api_call,
                                                description = _description,
                                                path = _path,
                                                method = _method,
                                                lang = _lang,
                                                domain = _domain, 
                                                api_description =_api_description,
                                                api_license = _api_license)
                                                # dataset = None,
                                                # accuracy = None)

                            data.append(self.add_datapoint(datapoint))
        self.data = data

"""
'ExtParser': Use this parser for APIs that contain API calls
"""
class ExtParser(OpenAPIParser):
    def __init__(self):
        super().__init__()

    def set_data(self,data):
        super().set_data(data)

    def get_data(self):
        return super().get_data()
    
    def extract_api_call(self,example_dict, datapoint, name):
        if example_dict['type'] == 'code' and example_dict['source'] is not None:
            for element in range(0,len(example_dict['source']),1):
                if len(example_dict['source'][element])>0:
                    datapoint.api_call = example_dict['source'][element]  
                    # Update functionanlity only if the api call name has a longer text than the summary
                    datapoint.functionality = name if len(name)>len(datapoint.functionality) else datapoint.functionality                                                                                                          
        return datapoint

    def init_datapoint(self,provider,path):
        return DataPoint( 
                    api_call = "", 
                    api_name = "",
                    api_provider = provider, 
                    endpoint = "",
                    # explanation = [],
                    # code = _api_call,
                    framework = "",
                    functionality = "",
                    api_arguments = {},
                    # python_environment_requirements = [],
                    # example_code = _api_call,
                    description = "",
                    path = path,
                    method = "",
                    lang = "",
                    domain = "",
                    api_description ="",
                    api_license = "")
                    # dataset = None,
                    # accuracy = None)

    def is_valid(self, parent):
        for element in parent:
            if "$ref" in element:
                return False
        return True

    def parse_data(self, source, provider):
        data = []

        print("---API Totals---")
        print(f"\t Total number of paths: {len(source['paths'])}") #TEST

        for path in source["paths"]:
            # Create datapoint
            datapoint = self.init_datapoint(provider,path)

            # Temp var
            name = ""

            # Save API info
            datapoint.api_provider = provider
            datapoint.api_name = source['info']['title'] if  "title" in source['info'] else ""
            datapoint.api_description = source['info']['description'] if  "description" in source['info'] else ""
            datapoint.api_license = source['info']['license'] if  "license" in source['info'] else ""

            for method in source["paths"][path]:
                datapoint.method = method
                if isinstance(source['paths'][path][method], dict):
                    if "operationId" in source["paths"][path][method]: datapoint.endpoint = source['paths'][path][method]['operationId']
                    if "summary" in source["paths"][path][method]: datapoint.functionality = source['paths'][path][method]['summary']
                    if "description" in source["paths"][path][method]: datapoint.description = source['paths'][path][method]['description']
                    if "tags" in source["paths"][path][method]: datapoint.domain = source['paths'][path][method]['tags']

                    # print(f"\t TEST - Endpoint: {datapoint.endpoint}") #TEST

                    if "x-sdk-operations" in source['paths'][path][method]: # Some elements do not have a code example
                        sdk_ops = source['paths'][path][method]['x-sdk-operations'] 
                        if "request-examples" in sdk_ops and self.is_valid(sdk_ops['request-examples']):
                            for lang in sdk_ops['request-examples']:
                                datapoint.lang = lang
                                examples = sdk_ops['request-examples'][lang]
                                if  examples is not None: 
                                    for example in examples:
                                        if "name" in example: name = example['name']
                                        if isinstance(example['example'], list):
                                            for example_dict in example['example']:     
                                                datapoint = self.extract_api_call(example_dict, datapoint, name)                                                                                                                        
                                        elif isinstance(example['example'], dict):
                                            datapoint = self.extract_api_call(example['example'], datapoint, name)
                                        else:
                                            pass # Add elif for other types
                                        # Only add datapoint to the list if the api_call text is valid
                                        if len(datapoint.api_call)>0: data.append(self.add_datapoint(datapoint)) 
                                else:
                                    print("skipping ...")
                                    print(f"\t Endpoint: {datapoint.endpoint}")                           
                        else:
                            print("skipping ...")
                            print(f"\t Endpoint: {datapoint.endpoint}")

                    else:
                        print("skipping ...")
                        print(f"\t Endpoint: {datapoint.endpoint}")
                else:
                    print("skipping ...")
                    print(f"\t Path: {path}")
                    print(f"\t Method: {method}")
            self.data = data