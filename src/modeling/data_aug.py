import json
import os
import re
import random
import time
from tqdm import tqdm
import pyarrow
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch.nn.functional as F
import argparse

def get_questions(question_file):
    """
    Loads a JSON file and returns its content as a data structure.

    :param question_file: The path to the JSON file.
    :return: The content of the JSON file.
    """
    # Attempt to load the questions file
    try:
        with open(question_file, "r", encoding='utf-8') as ques_file:
            data = json.load(ques_file)
        return data
    except FileNotFoundError:
        print(f"The file {question_file} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"The file {question_file} is not a valid JSON file.")
        return None

def save_json(data, file_path, filter_length=0):
    """
    Saves the given data to a JSON file at the specified file path.

    :param data: The data structure to save (usually a dict or a list).
    :param file_path: The path, including the filename, where the JSON file should be saved.
    :return: None
    """
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Attempt to save the data to the JSON file
    if len(data) <= filter_length:
        print("no data for this API")
    else:
        try:
            with open(file_path, "w", encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
            print(f"Data successfully saved to {file_path} with length {len(data)}")
        except IOError as e:
            print(f"An error occurred while writing the file: {e}")
        except TypeError as e:
            print(f"An error occurred with the data type being saved: {e}")

def truncate_and_tokenize(api_description, truncate_length=1024):
    # Tokenize the input text
    tokens = LLM_tokenizer.tokenize(api_description)

    # Check if the number of tokens exceeds 1024
    if len(tokens) > truncate_length:
        # Truncate the tokens
        truncated_tokens = tokens[:truncate_length]

        # Convert tokens back to string
        truncated_text = LLM_tokenizer.convert_tokens_to_string(truncated_tokens)
        return truncated_text
    else:
        # If the text does not exceed the limit, return it as it is
        return api_description

def parse_language():
    # Define the lists of languages
    langs1 = ["curl", "go", "java", "javascript", "libcurl"]
    langs2 = ["node", "php", "python", "ruby", "swift"]

    # Combine both lists
    combined_langs = langs1 + langs2

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process the language.')

    # Add the language argument
    parser.add_argument('lang', choices=combined_langs, help='Language to process')

    # Parse the arguments
    args = parser.parse_args()

    # Generate the API group name
    api_group = f"cleaned_{args.lang}"
    print(f"The API group is: {api_group}")
    return api_group

if __name__ == "__main__":

    api_description_truncate = 512
    new_instruct_truncate = 1024 * 3

    # Set the seed for the random number generator
    random.seed(42)  # You can use any number as the seed
    
    api_group = parse_language()

    my_device = 'cuda'

    print("loading LLM tokenizer")
    LLM_tokenizer = AutoTokenizer.from_pretrained('./LLMs/Mistral-7B-v0.1', use_fast=True)

    # Load embedding model from HuggingFace Hub
    print("loading embedding model")
    tokenizer = AutoTokenizer.from_pretrained('./LLMs/bge-large-en-v1.5')
    model = AutoModel.from_pretrained('./LLMs/bge-large-en-v1.5', torch_dtype=torch.float32).to(my_device)
    model.eval()

    # Load reranker model from HuggingFace Hub
    print("loading reranker model")
    reranker_tokenizer = AutoTokenizer.from_pretrained('./LLMs/bge-reranker-large')
    reranker_model = AutoModelForSequenceClassification.from_pretrained('./LLMs/bge-reranker-large', torch_dtype=torch.float32).to(my_device)
    reranker_model.eval()

    
    ##############################################
    # generating simple training data
    ##############################################
    
    training_data_path = f"./instr_data/{api_group}/total_training_{api_group}.json"
    
    training_data = get_questions(training_data_path)
    simple_training_data = []

    for data in training_data:
        data['api_description'] = truncate_and_tokenize(data['api_description'], truncate_length=api_description_truncate)
        data['instruction'] = f"**api_description**:{data['api_description']}\n**lang**:{data['api_call_data']['lang']}\n**instruction**:{data['instruction']}\n**output**\n"
        del data['api_description']
        del data['api_call_data']
        del data['instruction_test']
        simple_training_data.append(data)

    save_json(simple_training_data, f"./instr_data/{api_group}/total_simple_training_data_{api_group}.json", 0)

    ##############################################
    # generating few shot training data
    ##############################################

    training_data = get_questions(training_data_path)
    in_context_training_data = []

    num_of_context = 3

    for current_index, data in tqdm(enumerate(training_data)):
        data['api_description'] = truncate_and_tokenize(data['api_description'], truncate_length=api_description_truncate)
        new_instruction = f"**api_description**:{data['api_description']}\n**lang**:{data['api_call_data']['lang']}\n\nGiven the following examples:\n\n"
            
        matching_indexes = [i for i, item in enumerate(training_data) if item['api_description'] == data['api_description'] and i != current_index]
        #print(matching_indexes)
        selected_indexes = random.sample(matching_indexes, num_of_context) if len(matching_indexes) >= num_of_context else matching_indexes
        
        #print(selected_indexes)
        for selected_index in selected_indexes:
            new_instruction += f"**instruction**\n{training_data[selected_index]['instruction']}\n**output**\n{training_data[selected_index]['output']}\n\n"

        new_instruction = truncate_and_tokenize(new_instruction, truncate_length=new_instruct_truncate)
            
        new_instruction += f"Your actual task:\n**instruction**\n{data['instruction']}\n**output**\n"
        output = data['output']
        in_context_data = {"instruction": new_instruction,
                             "input": data["input"], 
                             "output": output}
        in_context_training_data.append(in_context_data)
    
    save_json(in_context_training_data, f"./instr_data/{api_group}/total_in_context_training_data_{api_group}.json", 0)

    ##############################################
    # combined simple and few shot training data
    ##############################################

    combined_training_data = []
    combined_training_data.extend(simple_training_data)
    combined_training_data.extend(in_context_training_data)

    save_json(combined_training_data, f"./instr_data/{api_group}/total_combined_training_data_{api_group}.json", 0)

    ##############################################
    # generating few shot testing data
    ##############################################

    test_level = 3
    num_of_context = 3
    
    for i in range(test_level):
        
        test_data_path = f"./instr_data/{api_group}/total_testing_{api_group}_level_{i+1}.json"
        test_data = get_questions(test_data_path)

        total_data_path = f"./instr_data/{api_group}/total_data_{api_group}.json"
        total_data = get_questions(total_data_path)

        in_context_test_datas = []
        
        for test in tqdm(test_data):
            
            test_api_name = test['api_name']
            test_api_output = test['output']
            
            matching_api_in_total_data = [i for i, data in enumerate(total_data) if data['api_name'] == test_api_name and data['output'] != test_api_output]
            
            selected_indexes = random.sample(matching_api_in_total_data, num_of_context) if len(matching_api_in_total_data) >= num_of_context else random.sample(matching_api_in_total_data, len(matching_api_in_total_data))

            test['api_description'] = truncate_and_tokenize(test['api_description'], truncate_length=api_description_truncate)
            new_api_description = f"**api_description**:{test['api_description']}\n**lang**:{test['api_call_data']['lang']}\n\nGiven the following examples:\n\n"

            for selected_index in selected_indexes:
                new_api_description += f"**instruction**\n{total_data[selected_index]['instruction']}\n**output**\n{total_data[selected_index]['output']}\n\n"

            new_api_description = truncate_and_tokenize(new_api_description, truncate_length=new_instruct_truncate)
            
            in_context_test_data = {"api_name": test['api_name'],
                            "api_description": new_api_description,
                            "api_call_data": test['api_call_data'],
                             "instruction": test['instruction'],
                             "instruction_test": test['instruction_test'],
                             "input": test['input'],
                             "output": test['output']}

            in_context_test_datas.append(in_context_test_data)

        save_json(in_context_test_datas, f"./instr_data/{api_group}/total_testing_{api_group}_level_{i+1}_IC_{num_of_context}.json", 0)

    ##############################################
    # generating retrieval few shot testing data
    ##############################################

    test_level = 3
    num_of_context = 3
    reranker_length = 5

    total_data_path = f"./instr_data/{api_group}/total_data_{api_group}.json"
    total_data = get_questions(total_data_path)
    
    total_data_with_embed = []
    for data in tqdm(total_data):
        input_instruction = data['instruction']
        # Tokenize sentences
        encoded_input = tokenizer([input_instruction], padding=True, truncation=True, return_tensors='pt').to(my_device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][0, 0].to('cpu')
        data['instruction_emb'] = sentence_embeddings
        total_data_with_embed.append(data)
    
    for i in range(test_level):
        
        test_data_path = f"./instr_data/{api_group}/total_testing_{api_group}_level_{i+1}.json"
        test_data = get_questions(test_data_path)

        in_context_test_datas = []
        in_context_rerank_test_datas = []
        
        for test in tqdm(test_data):
            
            test_api_name = test['api_name']
            test_api_output = test['output']

            encoded_input = tokenizer([test['instruction_test']], padding=True, truncation=True, return_tensors='pt').to(my_device)
            with torch.no_grad():
                model_output = model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                new_instruction_emb = model_output[0][0, 0].to('cpu')

            embeddings = torch.stack([data['instruction_emb'] for data in total_data_with_embed if data['api_name'] == test_api_name and data['output'] != test_api_output])
            embeddings_idx = [i for i, data in enumerate(total_data_with_embed) if data['api_name'] == test_api_name and data['output'] != test_api_output]

            cos_similarities = F.cosine_similarity(new_instruction_emb.unsqueeze(0), embeddings, dim=1)

            ########################################################
            # retrieval part
            ########################################################

            # Find the indices of the top k similarities
            if len(cos_similarities) >= num_of_context:
                top_indices = torch.topk(cos_similarities, num_of_context).indices
            else:
                top_indices = torch.topk(cos_similarities, len(cos_similarities)).indices

            test['api_description'] = truncate_and_tokenize(test['api_description'], truncate_length=api_description_truncate)
            new_api_description = f"**api_description**:{test['api_description']}\n**lang**:{test['api_call_data']['lang']}\n\nGiven the following examples:\n\n"
            
            # Convert top_indices to a Python list
            top_indices = top_indices.tolist()
            
            # reverse the order such that the top-1 is closest to the response for the final answer
            for selected_index in top_indices[::-1]:
                true_idx = embeddings_idx[selected_index]
                #new_api_description += f"{total_data_with_embed[true_idx]['instruction']}\n{total_data_with_embed[true_idx]['output']}\n\n"
                new_api_description += f"**instruction**\n{total_data_with_embed[true_idx]['instruction']}\n**output**\n{total_data_with_embed[true_idx]['output']}\n\n"

            new_api_description = truncate_and_tokenize(new_api_description, truncate_length=new_instruct_truncate)

            in_context_test_data = {"api_name": test['api_name'],
                            "api_description": new_api_description,
                            "api_call_data": test['api_call_data'],
                             "instruction": test['instruction'],
                             "instruction_test": test['instruction_test'],
                             "input": test['input'],
                             "output": test['output']}

            in_context_test_datas.append(in_context_test_data)

            ########################################################
            # reranker part
            ########################################################
            
            # Find the indices of the top k similarities
            if len(cos_similarities) >= reranker_length:
                top_indices = torch.topk(cos_similarities, reranker_length).indices
            else:
                top_indices = torch.topk(cos_similarities, len(cos_similarities)).indices

            pairs = [[test['instruction_test'], total_data_with_embed[embeddings_idx[indice]]['instruction']] for indice in top_indices]

            with torch.no_grad():
                inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(my_device)
                scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float().to('cpu')

            if len(top_indices) >= num_of_context:
                reranker_indices = torch.topk(scores, num_of_context).indices
            else:
                reranker_indices = torch.topk(scores, len(top_indices)).indices

            reranker_indices = reranker_indices.tolist()

            test['api_description'] = truncate_and_tokenize(test['api_description'], truncate_length=api_description_truncate)
            new_api_description = f"**api_description**:{test['api_description']}\n**lang**:{test['api_call_data']['lang']}\n\nGiven the following examples:\n\n"
            # reverse the order such that the top-1 is closest to the response for the final answer
            for selected_index in reranker_indices[::-1]:
                original_index = top_indices[selected_index]
                true_idx = embeddings_idx[original_index]
                new_api_description += f"**instruction**\n{pairs[selected_index][1]}\n**output**\n{total_data_with_embed[true_idx]['output']}\n\n"

            new_api_description = truncate_and_tokenize(new_api_description, truncate_length=new_instruct_truncate)
            
            in_context_rerank_test_data = {"api_name": test['api_name'],
                            "api_description": new_api_description,
                            "api_call_data": test['api_call_data'],
                             "instruction": test['instruction'],
                             "instruction_test": test['instruction_test'],
                             "input": test['input'],
                             "output": test['output']}

            in_context_rerank_test_datas.append(in_context_rerank_test_data)

        save_json(in_context_test_datas, f"./instr_data/{api_group}/total_testing_{api_group}_level_{i+1}_retrieval_IC_{num_of_context}.json", 0)
        save_json(in_context_rerank_test_datas, f"./instr_data/{api_group}/total_testing_{api_group}_level_{i+1}_retrieval_IC_{num_of_context}_reranker_{reranker_length}.json", 0)