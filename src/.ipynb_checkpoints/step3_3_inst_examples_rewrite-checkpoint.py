import json
import os
import time
import pandas as pd
import numpy as np
import argparse
import torch
import pyarrow
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
import tensor_parallel as tp
import concurrent.futures

from tools.data_manger import load_local_file_as_json, load_txt_file

# You need to adjust the 'max_split_size_mb' to a value that fits your specific scenario
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description="Process the ins_ex_path")
    parser.add_argument("--ins_ex_path", type=str, help="Path to the instruction examples JSON file")
    parser.add_argument("--model_path", type=str, help="Path to LLM model to rewrite")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs to use")
    args = parser.parse_args()

    # Using the argument
    ins_ex_path = args.ins_ex_path

    # load your local model
    model_path = args.model_path
    num_gpus = args.num_gpus
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16
    )

    model = model.to(dtype=torch.float16)
    
    print("num_gpus:", num_gpus)
    model = tp.tensor_parallel(model, [i for i in range(num_gpus)])
    model.eval()

    # load your instruction_examples_data
    data = load_local_file_as_json(file_path=ins_ex_path)

    instruction_examples_data = data['list']

    my_rewrite_prompt = load_txt_file("./prompts/rewrite_prompt.txt")

    for instruction_example_data in instruction_examples_data:
        
        template = "###Input:\nFunctionality: {functionality}\nDescription: {description}\nEndpoint: {endpoint}\nAPI: {api_name}\nUser query to refine: {output}\n###Output (refined user query):\n"

        filled_template = template.format(
                functionality=instruction_example_data["functionality"],
                description=instruction_example_data["description"],
                endpoint=instruction_example_data["endpoint"],
                api_name=instruction_example_data["API name"],
                output=instruction_example_data["output"]
            )

        rewrite_prompt = my_rewrite_prompt + filled_template

        messages=[{ 'role': 'user', 'content': f"{rewrite_prompt}"}]
        # Apply chat template and tokenize inputs
        inputs = tokenizer.apply_chat_template(
            messages,
            padding=False,
            return_tensors="pt",
        ).to("cuda")

        # Manually create an attention mask if it's not returned
        attention_mask = torch.ones(inputs.shape, dtype=torch.long).to("cuda")

        try:
            # Use concurrent.futures to apply a timeout to model.generate
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    model.generate,
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=True,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,  # Using eos token as padding token
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                outputs = future.result(timeout=360)  # Timeout set to 360 seconds
    
        except concurrent.futures.TimeoutError:
            print(f"Timeout occurred. Skipping...")
            continue
        except RuntimeError as e:
            print(f"Runtime error: {e}")
            continue

        # Process generated tokens
        generated_tokens = outputs.sequences[0][inputs.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_text = generated_text.replace('"', '')

        instruction_example_data["output to refine"] = instruction_example_data["output"]
        instruction_example_data["output"] = generated_text
        
        print("----------------------------------------------------------------")
        print("output to refine:", instruction_example_data["output to refine"])
        print("----------------------------------------------------------------")
        print("refined:", instruction_example_data["output"])

    # Save the modified data back into the original file
    with open(ins_ex_path, 'w') as file:
        json.dump(data, file, indent=4)