# Processing Pipeline
We follow a pipeline of six steps to build the *instruction-dataset* used for fine-tuning. At the core, this processing pipeline extracts information from [OpenAPI](https://www.openapis.org/) specification files. Using [OpenAPI](https://www.openapis.org/) files as datasource brings two important advantages over other methods to create instruction datasets (e.g., manual creation, mining code repositories, using Large Language Models):

- The relationship between instruction and api call can be automatically distilled. No manual work to create it is needed.
- The API call (code) is reliable enough as a probabilistic method (i.e., Large Language Model) is not used to generate it.

The figures below introduce the processing pipeline at a high level.

![image info](dataset-pipe-small.pdf)
<embed src="dataset-pipe-small.pdf" type="application/pdf" width="100%" height="600px" />

As a result of the process, two output files are generated per API:

1. API DB. A file that contains the API calls and their associated metadata.
2. API Dataset. A file that in addition to API calls and metadata, contains a natural language instruction in the format required for tuning.

The code snippet below shows the key components of a datapoint in the API dataset, *instruction* and *api call*:

```
###Instruction: Please tell me how to check the high level GDPS GM statuses.

###Output:
<<<api_call>>>: 
OkHttpClient client = new OkHttpClient();

Request request = new Request.Builder().url(\"https:///%3Cgdpsdomain%3E:%3Cport%3E/org.ibm.gdps/rest/v2/status/gm_global_status\").get().addHeader(\"domain\", \"SOME_STRING_VALUE\").addHeader(\"Authorization\", \"Basic REPLACE_BASIC_AUTH\").build();

Response response = client.newCall(request).execute();
```
In the next sections, we explain each step of the pipeline in detail.


## Step 1: Data Collection
It is up to API owners to make publicly available [OpenAPI](https://www.openapis.org/) specifications. If API owners want users to have access to their OpenAPI specification files, they most likely refernce them on their API documentation.

To build our dataset, we used [OpenAPI](https://www.openapis.org/) specification files publicly available from two sources: 

- [IBM API Hub](https://developer.ibm.com/apis/)
- [APIs.gurus](https://apis.guru/).

### Collection of OpenAPI specification files
There are two options to collect [OpenAPI](https://www.openapis.org/) specification files:
1. Manual. The OpenAPI specification file is downloaded from each API documentation website.
2. Automatic. A crawler is used to search for links to specification files (links to files with JSON or YAML extension). An API Hub is used as the seed URL from which the crawling process begins (i.e., [APIs.gurus](https://apis.guru/)). The file extension, website domain, and other URL characteristics can be used as scope delimiters for web crawling.

## Step 2: Build API DB

### Generate API calls
While some specification files contain the code to call API endpoints, others do not. For the second scenario, generate the api calls for each edpoint with the ```openapi-snippet``` library. 

If the API file that you are going to parse contains the API calls code, skip this step. Otherwise, visit https://github.com/ErikWittern/openapi-snippet, and follow the instructions to use ```openapi-snippet``` library or sourcecode to generete API calls for an OpenAPI specification file.

### How to run the scripts to build an API DB file?
Run the script ```python step2_build_api_db.py``` to build the API DB file. This script takes as input the OpenAPI specification file of your preference and parses its content. The output file obtained out of this step contains the information of all API endpoints in the format:

```json
    {
        "api_call": "",
        "api_provider": "",
        "endpoint": "",
        "explanation": [],
        "framework": "",
        "functionality": "",
        "api_arguments": {},
        "python_environment_requirements": [],
        "dataset": null,
        "accuracy": null,
        "description": "",
        "path": "",
        "method": "",
        "lang": "",
        "domain": "",
        "api_description": "",
        "api_license": ""
    }
```
Use the command below to run the script. Note this example only shows required arguments (```--input_file_name```, ``` --api_id```, and ```--api_db_output_file```):

```bash
python step2_build_api_db.py --input_file_name <name_of_api_file_including_api_calls>.json --api_id <api_id_str> --api_db_output_file <name_assigned_to_api_db>.json
``` 
You can also explicitly define input and output directories of your preference by including the arguments ```--input_dir``` and ```--output_dir``` in your command (see the example below).

```bash
python step2_build_api_db.py --input_file_name <name_of_api_file_including_api_calls>.json --api_id <api_id_str> --api_db_output_file <name_assigned_to_api_db>.json --input_dir ./data/input --output_dir ./data/output
```

The respective default values for ```--input_dir``` and ```--output_dir``` arguments are `./data/input` and `./data/output`.

### How to add a new parser for an API?
To add a new API parser you must complete two steps: 

1. Create a new parser class in ```spec_file_parser.py```.
2. Modify ```step2_build_api_db.py``` to add a new API string to the method ```create_parser```.

#### Create a new parser class for your API
The abstract class ```OpenAPIParser``` declared in ```spec_file_parser.py``` serves as blueprint to create new parsers. You must create a parser class that inherits from ```OpenAPIParser```, and that implements the abstract method ```parse_data```. The implementation of this method is custom for each API or family of APIs. Follow the example below.

```python
def parse_data(self, source):
    pass # Replace 'pass' with your code
```

You must also implement the abstract properties ```set_data``` and ```get_data``` as part of your class. As these properties contain boilerplate code you can simply call the super class. Copy and paste the code below inside your class:

```python
    def set_data(self,data):
        super().set_data(data)
```

```python
    def get_data(self):
        return super().get_data()
```
#### Add a new string to identify your API

First, add a string variable to identify your API (or API family).

```python
COS = "cos" # COS identifies the parser for COS APIs (cos-compatibility, cos-configuration). 
```

Second, take the example below as reference to modify ```create_parser``` in ```step2_build_api_db.py```.

```python
def create_parser(source,api):
    # ...
    # the code for other APIs goes here
    # ...
    elif api.lower() == COS:
        parser = COSParser() # COSParser is the class that you created and that inherits from OpenAPIParser
        parser.parse_data(source)
    return parser
```
## Step 3: Generate Instructions
To create synthetic instructions for each endpoint, we task a LLM with generating an instruciton prompt based on the API metadata. We use three metadata fields:

1. Functionality: a short description that summarizes the functionality of an endpoind.
2. Description: a longer text that describes how each endpoint works. This description is more verbose that *functionality*, and sometimes it is not present (it is empty) in the API specification file.
3. Endpoint: the name of the endpoint (function) in the API.

We use the [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai) to programmatically generate text (instructions) with pre-trained models hosted at IBM models hub. For this task, we have specifically used FLAN-UL2 (20B) pre-trained model. Once the instruction text has been generated, we concatenante the programming language for each datapoint to the instruction. Language information is also part of the API metadata.

### How to run the script to generate instructions?
Run the script ```python step3_instructions_gen.py``` to generate instructions for an API.

Use the command below to run the script. Note this example only shows required arguments (```--prompt_examples```, ``` --api_db_file```, and ```--instructions_output_file```):

```bash
python step3_instructions_gen.py --api_db_file <name_assigned_to_api_db>.json --prompt_examples <api_instruction_and_examples>.json --instructions_temp_file <name_assigned_to_instructions>.json
```
#### Key parameters
- **api_db_file** this is the *API DB* file generated as a result of completing [Step 2](#step-2-build-api-db).
- **prompt_examples** is a file that you must create manually by following the schema bellow. Note that you can copy-paste the ```instruction``` field (this will be the prompt) into your file as this is the same for all APIs. Then, select at least three datapoints to use as in-context examples from the *API DB* file. Copy ```functionality```, ```description```, and ```endpoint``` fields directly from this file. Finally, add a *human-generated* instruction that shows what each endpoint to be used as in-context example can do.
```json
{
    "instruction": "Given the functionality, description, and endpoint of an API, generate a task for which this API would be useful.\n\nYou should make the task concrete by replacing general concepts with more specific concepts.\nYou should try your best not to make the task become verbose, the task can only be 20 to 40 words long.\nThe API's functionality should not be exactly equal to the task generated.\nOnly include the API endpoint in the output to ask how to use it.",
    "list": [
        {"functionality": "<example functionality text>", "description": "<example description text>", "endpoint":"<example endpoint name>", "output": "<human-generated example instruciton for the endpoint>"},
        // <Add more examples here>
    ]
}
```
- **instructions_temp_file** is the name of the temporal file that will be generated as output of completing this processing step.

#### Other parameters
You can also explicitly define the path of *template* and *output* directories as well as file names for standard templates required along the process by explicitly including these arguments in your command. The command below shows how to do it.

```bash
python step3_instructions_gen.py --api_db_file <name_assigned_to_api_db>.json --prompt_examples <api_instruction_and_examples>.json --instructions_temp_file <name_assigned_to_instructions>.json --templates_dir ./data/templates --input_template input_template.txt --prompt_template prompt_template.txt --refined_prompt_template refined_prompt_template.txt --inpud_dir ./data/output
```
By default all templates and temporal files generated are respectively placed at ```./data/templates``` and ```./data/output```

Running this script will generate a temporal file with a similar structure to the *API DB* file, but with two new attributes, **instruction** and **refined_instruction**.

```json
    {
        // Attributes generated during Step 2 go here.
        "instruction": "",
        "refined_instruction": ""
    }
```
## Step 4: Generate API Call Domain
TBD

## Step 5: Apply Format
Once the dataset has been integrated by following the previous steps, we re-format the dataset as the fine tuning pipeline requiers. This format is inspired on [the Gorilla project](https://gorilla.cs.berkeley.edu/).

### How to run the script?
Run the script ```python step5_formatting.py``` to apply the format required by the training pipeline to the dataset. Note this example only shows  arguments that are required (```--instructions_temp_file``` and ``` --output_file```):

```bash
python step5_formatting.py --instructions_temp_file <name_assigned_to_instructions>.json --output_file <name_assigned_to_final>.json
```
