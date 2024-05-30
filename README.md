# API Pack A Massive Multi-Programming Language Dataset for API Call Generation

This is the repository with dataset, code, and models for the paper [API Pack A Massive Multi-Programming Language Dataset for API Call Generation](https://github.com/zguo0525/API-Pack)

## Dataset Summary

API Pack is a large-scale, multi-programming language dataset containing over 1 million instances across 10 programming languages for API call generation and intent detection. Its key features include multilinguality, scale, and a wide range of real-world APIs and use cases, enabling the assessment of cross-lingual skill transfer. Evaluation experiments demonstrated that CodeLlama-13B, fine-tuned with only 20,000 Python instances from API Pack, outperformed GPT-3.5 and GPT-4 in generating API calls for entirely new APIs, highlighting the dataset's effectiveness in improving the API call generation capabilities of large language models.

## Dataset Access

The dataset in JSON format is hosted on [Huggingface](https://huggingface.co/datasets/zguo0525/API-Pack), with each programming language as an individual file.

## Dataset Structure

Each instance in the API Pack dataset follows the example structure below:

```json
{
    "api_name": "Food-Cooking Recipe-API",
    "api_description": "Food-Cooking Recipe-API",
    "api_call_data": {
        "api_call": "curl --request GET \\\n  --url 'https//cooking-recipe2.p.rapidapi.com/getbycat/%7Bcategory%7D?category=SOME_STRING_VALUE' \\\n  --header 'X-RapidAPI-Host: SOME_STRING_VALUE' \\\n  --header 'X-RapidAPI-Key: SOME_STRING_VALUE'",
        "lang": "cURL",
        "functionality": "getrecipebycat",
        "api_arguments": {},
        "description": "Return specific list of recipes by category which will be pass",
        "domain": [],
        "path": "/getbycat/{category}"
    },
    "instruction": "I'd be happy to help you find recipes based on a specific category using the Food-Cooking Recipe-API. Could you please provide the name or ID of the desired category to get a list of applicable recipes?",
    "instruction_test": "Could you kindly guide me on how to fetch a list of recipes within a specific category using the Food-Cooking Recipe-API?",
    "input": "",
    "output": "**domain**:[]\n**api_call**:curl --request GET \\\n  --url 'https//cooking-recipe2.p.rapidapi.com/getbycat/%7Bcategory%7D?category=SOME_STRING_VALUE' \\\n  --header 'X-RapidAPI-Host: SOME_STRING_VALUE' \\\n  --header 'X-RapidAPI-Key: SOME_STRING_VALUE'\n**api_provider**:\n**lang**:cURL",
    "unique_id": "Food-Cooking-Recipe-API.json_0"
},
```

- `api_name` (str): Name of the API
- `api_description` (str): Description of the API
- `api_call_data` (dict): Data related to the API call
  - `api_call` (str): API call code snippet
  - `lang` (str): Programming language of the API call
  - `functionality` (str): Functionality of the API call
  - `api_arguments` (dict): Arguments for the API call (empty in this example)
  - `description` (str): Description of the API call functionality
  - `domain` (list): List of domains (empty in this example)
  - `path` (str): API endpoint path
- `instruction` (str): Instruction or query related to the API call
- `instruction_test` (str): Rephrased or alternative instruction
- `input` (str): Input data for the API call (empty in this example)
- `output` (str): Expected output or response from the API call, including the following fields:
  - `domain` (list): List of domains (empty in this example)
  - `api_call` (str): API call code snippet
  - `api_provider` (str): API provider (empty in this example)
  - `lang` (str): Programming language of the API call
- `unique_id` (str): Unique identifier for the API data entry

## License

API Pack dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License. You are free to share and adapt the material under the terms that you must give appropriate credit, provide a link to the license, and indicate if changes were made.
The code under this repo is licensed under an MIT License.

## Disclaimer

This dataset was collected and released solely for research purposes to improve open-source large language models' API call generation capabilities. The authors are strongly against any potentially harmful use of the data or technology by any party.

## Citation

If you find our dataset useful, please consider citing the paper:

```
@misc{guo2024api,
      title={API Pack: A Massive Multilingual Dataset for API Call Generation}, 
      author={Zhen Guo and Adriana Meza Soria and Wei Sun and Yikang Shen and Rameswar Panda},
      year={2024},
      eprint={2402.09615},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
