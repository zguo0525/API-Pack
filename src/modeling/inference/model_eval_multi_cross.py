import json
import argparse
import os
from tqdm import tqdm
import pyarrow
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
import re
import tensor_parallel as tp
from torch import nn

# You need to adjust the 'max_split_size_mb' to a value that fits your specific scenario
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

# Load Model from HF
def load_model(
        model_path: str,
        device: str,
        num_gpus: int,
    ):
    pattern = r".*output\/(.*?)\/"
    match = re.search(pattern, model_path)
    if match is not None:
        match_model_name = match.group(1)
        tokenizer = AutoTokenizer.from_pretrained(f"../LLMs/{match_model_name}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage = True, 
        torch_dtype=torch.float32
    )

    model = model.to(dtype=torch.float16)
    
    print("num_gpus:", num_gpus)
    model = tp.tensor_parallel(model, [i for i in range(num_gpus)])
    model.eval()
    return model, tokenizer

def get_questions(question_file):
    # Load dataset
    with open(question_file) as file:
        question_jsons = json.load(file)
    return question_jsons

def truncate_and_tokenize(api_description, LLM_tokenizer, truncate_length=1024):
    
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

def run_eval(args, question_jsons, with_in_context):
    # Evaluate the model for answers
    model, tokenizer = load_model(
        args.model_path, args.device, args.num_gpus,
    )

    if not with_in_context:
        LLM_tokenizer = AutoTokenizer.from_pretrained('/gpfs/u/home/SIFA/SIFAzhnu/scratch/llm4tools/src/modeling/LLMs/Mistral-7B-v0.1', use_fast=True)
    
    ans_jsons = []
    for i, ques_json in enumerate(tqdm(question_jsons)):
        if with_in_context:
            prompt = f"{ques_json['api_description']}Your actual task:\n**instruction**\n{ques_json['instruction_test']}\n**output**\n"
        else:
            ques_json['api_description'] = truncate_and_tokenize(ques_json['api_description'],
                                                                 LLM_tokenizer,
                                                                 truncate_length=512)
            prompt = f"**api_description**:{ques_json['api_description']}\n**lang**:{ques_json['api_call_data']['lang']}\n\nYour actual task:\n**instruction**\n{ques_json['instruction_test']}\n**output**\n"
        prompt = f"{prompt}\n### ASSISTANT:\n"
        input_ids = tokenizer([prompt], padding=False, truncation=True, max_length=4000).input_ids
        
        output_ids = model.generate(
            torch.as_tensor(input_ids).to('cuda'),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=512,
        )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        ans_jsons.append(
            {
                "instruction_test": prompt,
                "output": outputs,
                "api_name": ques_json["api_name"],
                "api_description": ques_json["api_description"],
                "ground_truth": ques_json["output"],
            }
        )
    
    # Create folder if it doesn't exist
    folder_path = os.path.dirname(args.answer_file)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # Write output to file
    with open(args.answer_file, "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")
    return ans_jsons

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True)
    parser.add_argument(
        "--question-file", 
        type=str, 
        required=True)
    parser.add_argument(
        "--device", 
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--answer-file", 
        type=str, 
        default="answer.jsonl"
    )
    parser.add_argument(
        "--num-gpus", 
        type=int, 
        default=2
    )
    args = parser.parse_args()

    questions_json = get_questions(args.question_file)

    with_in_context = True if "_IC_3" in args.question_file else False
    print("with_in_context:", with_in_context)
    
    run_eval(
        args,
        questions_json,
        with_in_context
    )