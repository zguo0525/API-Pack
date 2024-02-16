import json
import os
import re
import random
import time
from tqdm import tqdm
import pyarrow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from utils.prompter import Prompter
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from datasets import disable_caching
disable_caching()

HF_DATASETS_CACHE = '/gpfs/u/home/SIFA/SIFAzhnu/scratch/llm4tools/src/modeling/finetuning/cache'

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(HF_DATASETS_CACHE):
    os.makedirs(HF_DATASETS_CACHE)

# Set the environment variable for the Hugging Face datasets cache
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    tokenized_output = tokenizer.encode(prompt, truncation=False)
    if len(tokenized_output) >= cutoff_len:
        print(f"Warning: Input length is {len(tokenized_output)}, which is longer than the maximum length ({cutoff_len}). The input will be truncated.")
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

def save_data(data, save_directory):
    # Ensure 'data' is a Dataset object
    if not isinstance(data, Dataset):
        raise ValueError("The 'data' argument must be a 'datasets.Dataset' object.")

    # Save the dataset to the specified directory
    data.save_to_disk(save_directory)

if __name__ == "__main__":

    prefix = "/gpfs/u/home/SIFA/SIFAzhnu/scratch/llm4tools/src/modeling"

    #base_models = [f"{prefix}/LLMs/CodeLlama-13b-hf", f"{prefix}/LLMs/Llama-2-13b-hf", f"{prefix}/LLMs/Mistral-7B-v0.1", f"{prefix}/LLMs/granite-13b-base-v2"]
    base_models = [f"{prefix}/LLMs/CodeLlama-13b-hf"]
    data_prefixs = ["combined"]
    #["simple"]
    
    for base_model in base_models:
        for data_prefix in data_prefixs:
    
            data_folder = "total_curl"
            data_path = f"{prefix}/instr_data/{data_folder}/total_{data_prefix}_training_data_total_curl.json"
        
            cutoff_len = 4096
            train_on_inputs = False
            add_eos_token = False
        
            prompter = Prompter("alpaca")
        
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
            
            tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            tokenizer.padding_side = "left"  # Allow batched inference
        
            data = load_dataset("json", data_files=data_path, cache_dir=HF_DATASETS_CACHE)["train"]
            
            # Define the sizes for data subsets
            if data_prefix == "simple":
                #data_sizes = [20000]
                data_sizes = [40000, 60000, 80000, len(data)]
            elif data_prefix == "combined":
                #data_sizes = [20000]
                data_sizes = [40000, 60000, 80000, len(data)//2]
            
            for size in data_sizes:
                
                if data_prefix == "simple":
                
                    subset = data.select(range(size))
                
                elif data_prefix == "combined":
                    
                    total_data_size = len(data)
                    middle_start_index = total_data_size // 2
                    middle_end_index = middle_start_index + size
                    
                    print(total_data_size, middle_start_index)
                    
                    # Selecting the first size entries
                    first_subset = data.select(range(size))
                    # Selecting the first size entries from the middle
                    middle_subset = data.select(range(middle_start_index, middle_end_index))
                    # Combining both subsets
                    subset = concatenate_datasets([first_subset, middle_subset])
                    
                tokenized_data = subset.shuffle(seed=42).map(generate_and_tokenize_prompt)
        
                # Construct the filename
                folder_name = f"../instr_data/{data_folder}/tokenized_{data_prefix}_data_{base_model.split('/')[-1]}_{size}"
                save_data(tokenized_data, folder_name)