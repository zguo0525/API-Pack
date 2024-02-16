"""
 Schema
"""
class DataPoint():
    def __init__(self, 
                 api_call:str, 
                 api_name:str,
                 api_provider:str, 
                 endpoint:str,
                #  explanation:str,
                #  code:str, # repeated
                 framework:str,
                 functionality:str,
                 api_arguments:dict,
                #  python_environment_requirements: list, # Do we need it?
                #  example_code: str, # repeated
                 description: str,
                 path: str,
                 method: str,
                 lang: str,
                 domain: str, 
                 api_description:str,
                 api_license: str,
                #  dataset = None, # Do we need it?
                #  accuracy = None, # Do we need it?
                 ):
        self.api_call = api_call
        self.api_name = api_name
        self.api_provider = api_provider
        self.endpoint = endpoint
        # self.explanation = explanation
        # self.code = code
        self.framework = framework
        self.functionality = functionality
        self.api_arguments = api_arguments
        # self.python_environment_requirements = python_environment_requirements
        # self.example_code = example_code
        # self.dataset = dataset
        # self.accuracy = accuracy
        self.description = description
        self.path = path
        self.method = method
        self.lang = lang
        self.domain = domain
        self.api_description = api_description
        self.api_license = api_license

    def get_api_call(self):
        return self.api_call
    def set_api_call(self, value):
        self.api_call = value

    def get_api_name(self):
        return self.api_name
    def set_api_name(self,value):
        self.api_name = value

    def get_api_provider(self):
        return self.api_provider
    def set_api_provider(self,value):
        self.api_provider = value
    
    def get_endpoint(self):
        return self.endpoint
    def set_endpoint(self,value):
        self.endpoint = value

    def get_framework(self):
        return self.framework
    def set_framework(self,value):
        self.framework = value

    def get_functionality(self):
        return self.functionality
    def set_functionality(self,value):
        self.functionality = value

    def get_api_arguments(self):
        return self.api_arguments
    def set_api_arguments(self,value):
        self.api_arguments = value

    def get_description(self):
        return self.description
    def set_description(self,value):
        self.description = value

    def get_path(self):
        return self.path
    def set_path(self,value):
        self.path = value


    def get_method(self):
        return self.method
    def set_method(self,value):
        self.method = value

    def get_lang(self):
        return self.lang
    def set_lang(self,value):
        self.lang = value

    def get_domain(self):
        return self.domain
    def set_domain(self,value):
        self.domain = value

    def get_api_description(self):
        return self.api_description
    def set_api_description(self,value):
        self.api_description = value

    def get_api_license(self):
        return self.api_license
    def set_api_license(self,value):
        self.api_license = value