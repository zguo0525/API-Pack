import os
import json
import argparse
import time
import tools.data_manger as dm
import tools.instruction_generator_local as ig
import torch
import pyarrow
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
import tensor_parallel as tp

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        type=str,
                        required=False,
                        help="Directory in which the API DB file is located.") 
    parser.add_argument("--api_db_file",
                        type=str,
                        required=True,
                        help="The name of API DB file in json format, it cannot be empty.")
    parser.add_argument("--templates_dir",
                        type=str,
                        required=False,
                        default="./data/templates", 
                        help="Directory to save all template files.")   
    parser.add_argument("--input_template",
                        type=str,
                        required=False,
                        default="input_template_instruction.txt",
                        help="The input template file name.")
    parser.add_argument("--prompt_template",
                        type=str,
                        required=False,
                        default="prompt_template_instruction.txt",
                        help="The prompt template file name.")
    parser.add_argument("--bt_prompt_template",
                    type=str,
                    required=False,
                    default="back_translation_prompt_template.txt",
                    help="The prompt template file name for back translation.")
    parser.add_argument("--bt_input_template",
                type=str,
                required=False,
                default="back_translation_input_template.txt",
                help="The input template file name for back translation.")
    parser.add_argument("--prompt_examples",
                        type=str,
                        required=True,
                        help="The file name with examples to add to the prompt, it cannot be empty.")
    parser.add_argument("--temporal_dir",
                        type=str,
                        required=False,
                        default="./data/temporal_files", 
                        help="Directory to load and/or save all temporal files.") 
    parser.add_argument("--instructions_file",
                        type=str,
                        required=True,
                        help="The name of temporal file containing the API info and instructions in json format. It cannot be empty.")
    parser.add_argument("--dotenv_path",
                            type=str,
                            required=False,
                            default="../.env",
                            help="Name of dotenv file.")
    parser.add_argument("--checkpoint_number",
                            type=str,
                            required=False,
                            default="1",
                            help="Set the stage to start from 1:'instructions generation', 2:'generate bt', 3:'selecte best candidate')")
    parser.add_argument("--num_gpus",
                            type=int,
                            required=False,
                            default=1,
                            help="Number of GPUs to use for inference")
    parser.add_argument("--model_path",
                            type=str,
                            required=True,
                            help="Number of GPUs to use for inference")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()
    start_time = time.time()

    # CREDS FILE
    DOTENV_PATH = args.dotenv_path
    print(f"The environment file loaded is {DOTENV_PATH}")

    # INPUT
    INPUT_DIR = args.input_dir
    API_DB_FILE = args.api_db_file

    # TEMPLATES
    TEMPLATES_DIR = args.templates_dir
    INPUT_TEMPLATE = args.input_template
    PROMPT_TEMPLATE = args.prompt_template
    PROMPT_EXAMPLES = args.prompt_examples
    BACK_TRANS_PROMPT_TEMPLATE = args.bt_prompt_template
    BACK_TRANS_INPUT_TEMPLATE = args.bt_input_template

    # TEMPORAL FILES
    TEMPORAL_DIR = args.temporal_dir
    INSTRUCTION_FILES_DIR = os.path.join(TEMPORAL_DIR,"instruction_files")
    CHECKPOINT_FILES_DIR = os.path.join(TEMPORAL_DIR,"checkpoints")
    EXAMPLES_DIR = os.path.join(TEMPORAL_DIR,"inst_examples")
    INSTRUCTIONS_OUTPUT_FILE = args.instructions_file

    # VARS
    CHECKPOINT = args.checkpoint_number

    # LOCAL MODELS
    num_gpus = args.num_gpus
    model_path = args.model_path

    print("---Arguments---")
    print(f"\t API DB file: {API_DB_FILE}")
    print(f"\t Templates directory: {TEMPLATES_DIR}")
    print(f"\t Input template file: {INPUT_TEMPLATE}")
    print(f"\t Prompt template file: {PROMPT_TEMPLATE}")
    print(f"\t Prompt examples directory: {EXAMPLES_DIR}")
    print(f"\t Prompt examples file: {PROMPT_EXAMPLES}")
    print(f"\t Instruction files directory: {INSTRUCTION_FILES_DIR}")
    print(f"\t Output file (instructions): {INSTRUCTIONS_OUTPUT_FILE}")
    print(f"\t Using model from path: {model_path}")


    # load your local model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16
    )

    model = model.to(dtype=torch.float16)
    model = tp.tensor_parallel(model, [i for i in range(num_gpus)])
    model.eval()

    if(CHECKPOINT == "1"):
        print(f"--- Loading API DB file from {os.path.join(INPUT_DIR,API_DB_FILE)} ---")
        data = dm.load_local_file_as_json(os.path.join(INPUT_DIR,API_DB_FILE))

        print(f"--- Generating instructions ---")
        print(f"\t {len(data)} datapoints to process ...")

        data_instruct = ig.generate_instructions(data = data, model=model, tokenizer=tokenizer,
                                                inputs_template_path = os.path.join(TEMPLATES_DIR,INPUT_TEMPLATE), 
                                                prompt_template_path = os.path.join(TEMPLATES_DIR,PROMPT_TEMPLATE), 
                                                ins_ex_path = os.path.join(EXAMPLES_DIR,PROMPT_EXAMPLES),
                                                dotenv_path=DOTENV_PATH) 
        
        # Save instructions checkpoint
        print(f"--- Checkpoint: Saving API dataset with instructions ---")
        dm.save_data_as_json_array(data_instruct,os.path.join(CHECKPOINT_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE))
        print(f"\tAPI dataset with instructions saved as {os.path.join(CHECKPOINT_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE)}")

        print(f"--- Back translation ---")
        data_instruct = ig.back_translation(data = data_instruct, model=model, tokenizer=tokenizer,
                                            back_trans_prompt_template_path = os.path.join(TEMPLATES_DIR,BACK_TRANS_PROMPT_TEMPLATE),
                                            back_trans_input_template_path = os.path.join(TEMPLATES_DIR,BACK_TRANS_INPUT_TEMPLATE),
                                            ins_ex_path = os.path.join(EXAMPLES_DIR,PROMPT_EXAMPLES),
                                            dotenv_path=DOTENV_PATH)
        
        # Save back translation checkpoint
        print(f"--- Checkpoint: Saving API dataset with instructions and back translation ---")
        dm.save_data_as_json_array(data_instruct,os.path.join(CHECKPOINT_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE))
        print(f"\tAPI dataset with instructions and back translation saved as {os.path.join(CHECKPOINT_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE)}")
    
    elif(CHECKPOINT == "2"): # Load data to run BT and select best candidate
        print(f"--- Loading checkpoint file from {os.path.join(CHECKPOINT_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE)} ---")
        data_instruct = dm.load_local_file_as_json(os.path.join(CHECKPOINT_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE))

        print(f"\t {len(data_instruct)} datapoints to process ...")
  
        print(f"--- Back translation ---")
        data_instruct = ig.back_translation(data = data_instruct,
                                            back_trans_prompt_template_path = os.path.join(TEMPLATES_DIR,BACK_TRANS_PROMPT_TEMPLATE),
                                            back_trans_input_template_path = os.path.join(TEMPLATES_DIR,BACK_TRANS_INPUT_TEMPLATE),
                                            params = params_back_trans,
                                            dotenv_path=DOTENV_PATH)
    
        # Save back translation checkpoint
        print(f"--- Checkpoint: Saving API dataset with instructions and back translation ---")
        dm.save_data_as_json_array(data_instruct,os.path.join(CHECKPOINT_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE))
        print(f"\tAPI dataset with instructions and back translation saved as {os.path.join(CHECKPOINT_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE)}")

    elif(CHECKPOINT == "3"): # Load data to select best candidate
        print(f"--- Loading checkpoint file from {os.path.join(CHECKPOINT_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE)} ---")
        data_instruct = dm.load_local_file_as_json(os.path.join(CHECKPOINT_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE))

        print(f"\t {len(data_instruct)} datapoints to process ...")

    print(f"--- Selecting best candidate ---")
    data_instruct = ig.select_best_candidate(data = data_instruct) 

    print(f"--- Saving API dataset with instructions ---")
    dm.save_data_as_json_array(data_instruct,os.path.join(INSTRUCTION_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE))
    print(f"\tAPI dataset with instructions saved as {os.path.join(INSTRUCTION_FILES_DIR,INSTRUCTIONS_OUTPUT_FILE)}")

    print(f"--- Calculating processing time ---")
    end_time = time.time()
    execution_time = start_time - end_time
    print(f"\tInstructions for {API_DB_FILE} were created in {execution_time}.")