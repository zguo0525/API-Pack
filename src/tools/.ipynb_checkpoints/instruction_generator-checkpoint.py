import pystache
import json
import os
import time

import pandas as pd
import numpy as np

from dotenv import load_dotenv

from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams
from genai.prompt_pattern import PromptPattern
from genai.options import Options

from tools.output_formatting import format_from_template
from tools.data_manger import load_local_file_as_json, load_txt_file

GEN_ERROR = "<<ERROR: INSTRUCTION NOT GENERATED>>"

def refine_instructions(data:[], 
                          prompt_template:str, 
                          params: GenerateParams, 
                          dotenv_path:str,
                          model_id:str = "google/flan-ul2"):
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("GENAI_KEY", None)
    api_endpoint = os.getenv("GENAI_API", None)

    creds = Credentials(api_key, api_endpoint)
    model = Model(model_id, params=params, credentials=creds)

    prompts = []
    for data_point in data: 
        prompt = format_from_template(template_path = prompt_template, data_point = data_point)
        prompts.append(prompt)

    responses = model.generate_as_completed(prompts)
    for (data_point, resp) in zip(data,responses):
        if resp is None: data_point["refined_instruction"] = GEN_ERROR
        else: data_point["refined_instruction"] = resp.generated_text
    return data

def parse_candidates_info(responses):
    idx = 1
    idx_lst = []
    inputs = []
    instructions = []
    means = []

    for resp in responses:
        idx_lst.append(idx)
        # print(f"--------------- \
        #         \nINPUT_TEXT:\n\n{resp.input_text}\
        #         \n\nGENERATED_TEXT:{resp.generated_text}")
        if resp is not None:
            logprob_total = 0
            for token in resp.generated_tokens:
                logprob_total += token.logprob
                # print(f"--------------- \
                #  \nINPUT_TEXT:\n\n{token.logprob}")
            mean = (logprob_total/resp.generated_token_count)

            instructions.append(resp.generated_text)
            inputs.append(resp.input_text)
            means.append(mean)
        else:
            instructions.append(GEN_ERROR)
            inputs.append(GEN_ERROR)
            means.append(-0.999999) # TO DO: Check min value for negative numbers
        idx += 1
    
    # TEST
    # print(f"Lists - idx_lst:{len(idx_lst)}, candidates:{len(instructions)}, input_text:{len(inputs)}, means:{len(means)}")

    df_candidates = pd.DataFrame(list(zip(idx_lst,instructions,inputs,means)),
                      columns=['idx','candidate','input_text','gen_tokens_mean'])

    return df_candidates

def select_best_candidate(data:[]):

    # Convert All the candidates for al datapoints into a pandas DF
    lst_idx = []
    lst_candidates = []
    lst_input_text = []
    lst_gen_tokens_mean = []
    lst_bt_input_tokens_mean = []
    lst_sum = []
    lst_dp_id = []

    dp_counter = 1
    for datapoint in data:
        for candidate_info in datapoint["instruction_candidates"]:
            idx = candidate_info["idx"]
            candidate = candidate_info["candidate"]
            input_text = candidate_info["input_text"]
            gen_tokens_mean = candidate_info["gen_tokens_mean"]
            back_trans_input_tokens_mean =  candidate_info["back_trans_input_tokens_mean"]
            sum = float(gen_tokens_mean) + float(back_trans_input_tokens_mean)

            # TEST
            # print(f"gen_tokens_mean:{gen_tokens_mean} back_trans_input_tokens_mean:{back_trans_input_tokens_mean} sum:{sum}")

            lst_idx.append(idx)
            lst_candidates.append(candidate)
            lst_input_text.append(input_text)
            lst_gen_tokens_mean.append(gen_tokens_mean)
            lst_bt_input_tokens_mean.append(back_trans_input_tokens_mean)
            lst_sum.append(sum)
            lst_dp_id.append(str(dp_counter))
            
        dp_counter += 1

        # Create pd dataframe with all candidates
        df_candidates = pd.DataFrame({'idx_candidate': lst_idx,
                            'group_id' : lst_dp_id,
                            'candidate': lst_candidates,
                            'input_text': lst_input_text,
                            'gen_tokens_mean': lst_gen_tokens_mean,
                            'input_tokens_mean': lst_bt_input_tokens_mean,
                            'sum': lst_sum})

    # For each group (dp position in the list), find the best candidate
    group = 1
    lst_best_candidates = []
    while (group < dp_counter):
        mask = df_candidates['group_id'].isin([str(group)])
        df_group = df_candidates[mask]
        candidates_by_gen_tokens_mean = df_group.sort_values(by=['gen_tokens_mean'], ascending=False)
        candidates_by_input_tokens_mean = df_group.sort_values(by=['input_tokens_mean'], ascending=False)
        candidates_by_sum = df_group.sort_values(by=['sum'], ascending=False)

        # TEST
        # print(f"group Id: {group}")
        # print(candidates_by_gen_tokens_mean)
        # print(candidates_by_input_tokens_mean)
        # print(candidates_by_sum)
        
        metrics_count = []
        candidates_idx = []
        metrics_sum = []
        for i,(by_gen_tokens_mean, by_input_tokens_mean, by_sum) in enumerate(zip(candidates_by_gen_tokens_mean.values, candidates_by_input_tokens_mean.values, candidates_by_sum.values)):
            # TEST
            # print(by_gen_tokens_mean[0],by_input_tokens_mean[0],by_sum[0])

            # Same candidate is the top for three metrics
            if(by_sum[0] == by_gen_tokens_mean[0]) and (by_sum[0] == by_input_tokens_mean[0]): 
                metrics_count.append(3)
                candidates_idx.append(by_sum[0])
                metrics_sum.append(by_sum[6]) 
            # Same candidate is the top for two metrics
            elif(by_sum[0] == by_gen_tokens_mean[0]) or (by_sum[0] == by_input_tokens_mean[0]):
                metrics_count.append(2)
                candidates_idx.append(by_sum[0])
                metrics_sum.append(by_sum[6])
            elif (by_gen_tokens_mean[0]== by_input_tokens_mean[0]):
                metrics_count.append(2)
                candidates_idx.append(by_gen_tokens_mean[0])
                metrics_sum.append(by_sum[6])
            else:
                metrics_count.append(1)
                candidates_idx.append(by_sum[0])
                metrics_sum.append(by_sum[6])
        
        # Create Pandas DF with metrics count per group of candidates
        df_metrics = pd.DataFrame({'metrics_count': metrics_count,
                            'candidate_idx' : candidates_idx,
                            'sum': metrics_sum})
        df_metrics = df_metrics.sort_values(by=['metrics_count'], ascending=False)

        # TEST
        # print("== DF Metrics ==")
        # print(df_metrics)
        # print(type(df_metrics.loc[0]['metrics_count']))
        # print(f"max metric count: {df_metrics.loc[0]['metrics_count']}")

        max_metric_count = df_metrics.loc[0]['metrics_count']
        mask = df_metrics['metrics_count'].isin([max_metric_count])
        df_ma_metrics = df_metrics[mask]

        # If for than one candidate has the same metric count, sort by sum metric
        if isinstance(df_ma_metrics,pd.DataFrame): df_ma_metrics = df_ma_metrics.sort_values(by=['sum'], ascending= False)

        # TEST
        # print("== DF MAX Metrics Sorted ==")
        # print(df_ma_metrics)
        print(f"\tBest candidate idx: {df_ma_metrics.loc[0]['candidate_idx'].astype(np.int64)}")

        lst_best_candidates.append(df_ma_metrics.loc[0]['candidate_idx'].astype(np.int64))
        group += 1
    
    # TEST
    # print("== Best candidates list ==")
    # print(lst_best_candidates)
    print(f"\tTotal datapoints: {len(data)}, Total best candidates: {len(lst_best_candidates)}")

    for datapoint, best_candidate in zip(data,lst_best_candidates):
        for candiate_info in datapoint["instruction_candidates"]:
            if (candiate_info["idx"] == best_candidate):
                datapoint["best_instruction"] = {
                    "idx" : candiate_info["idx"],
                    "candidate": candiate_info["candidate"]
                }

    return data

def print_candidates_info(responses):
    for resp in responses:
        if resp is not None:
            print(f"--------------- \
                \nINPUT_TEXT:\n\n{resp.input_text}\
                \n\nGENERATED_TEXT:{resp.generated_text}")
            
            logprob_total = 0
            for token in resp.generated_tokens:
                logprob_total += token.logprob
                print(f"\ttoken:{token.text} -- logprob:{token.logprob}")
            mean = (logprob_total/resp.generated_token_count)
            print(f"\n\nMEAN:{mean}\
                    \n---------------")
        else:
            print(f"--------------- \
                  \n RESPONSE IS NONE \
                    ---------------")


def create_credentials(dotenv_path:str):
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("GENAI_KEY", None)
    api_endpoint = os.getenv("GENAI_API", None)
    return Credentials(api_key, api_endpoint)

def create_model(params: GenerateParams, 
                 creds: Credentials,
                 model_id:str = "google/flan-ul2"):
    return Model(model_id, params=params, credentials=creds)

def generate_instructions(data:[], 
                          inputs_template_path:str, 
                          prompt_template_path:str, 
                          ins_ex_path:str,
                          params: GenerateParams, 
                          dotenv_path:str,
                          model_id:str = "google/flan-ul2",
                          candidates_max:int = 5):

    creds = create_credentials(dotenv_path)
    model = create_model(params=params, model_id=model_id, creds=creds)

    # Load general templates, instruction, and data examples
    prompt_template = load_txt_file(prompt_template_path)
    # print(prompt_template)
    instruction_examples_data = load_local_file_as_json(file_path = ins_ex_path)
    # print(instruction_examples_data) #TEST

    # Prompt config
    pt = PromptPattern.from_watsonx(credentials=creds, name="prompt_builder", template = prompt_template)
    options = Options(
        # watsonx_template=pt # TEST
        watsonx_template=pt,
        watsonx_data = instruction_examples_data
    )

    # Candidates generation
    for data_point in data:
        # Validate inputs length
        if (len(data_point['description']) + 200) >= 4096: 
                previous_len = len(data_point['description'])
                data_point['description'] = data_point['description'][0:200]
                print(f"Truncated 'description' length for endpoint'{data_point['endpoint']}' from {previous_len} to {len(data_point['description'])}.")

        # Create input
        if len(inputs_template_path)>0:
            input = format_from_template(template_path = inputs_template_path, data_point = data_point)
            if (len(input)) >= 4096: 
                print(f"ERROR: Input length for endpoint '{data_point['endpoint']}' is over {len(input)}.")
        else:
            input = data_point["api_description"]

        # Create the list of inputs (same input repeated N times)
        inputs = []
        iterations = range(candidates_max)
        for n in iterations:
            inputs.append(input)
        # print(inputs) # TEST

        # Generate instruction candidates
        responses = model.generate_async(inputs, options=options, hide_progressbar=True, throw_on_error=True)

        # Calculate mean log probs
        # print_candidates_info(responses=responses) #TEST
        df_candidates = parse_candidates_info(responses=responses)
        print("=df_candidates=")
        print(df_candidates) # TEST

        # Save instruction candidates
        data_point["instruction_candidates"] = df_candidates.to_dict(orient="records")

        # Sleep routine 1 s per datapoint
        # seconds = 1
        # print(f"Sleeping {seconds} ...")
        # time.sleep(seconds)

    return data

# Back translation -> Regenerate the metadata based on the candidate and respective ground truth (real metadata).
def back_translation(data:[],
                     back_trans_prompt_template_path:str,
                     back_trans_input_template_path:str,
                     params: GenerateParams,
                     model_id:str,
                     dotenv_path:str):

    creds = create_credentials(dotenv_path)
    model = create_model(params=params, model_id=model_id, creds=creds)

    # ['idx','candidate','input_text','mean']
    for data_point in data:
        # Create inputs and tokenizers for bt
        inputs = []
        tokenized_metadata_lst = []
        for candidate in data_point["instruction_candidates"]:  
            bt_data_point = {"candidate": candidate['candidate'],
                                "metadata": candidate['input_text']}
        
            # Load general templates
            bt_prompt_template = load_txt_file(back_trans_prompt_template_path)
    
            # Prompt config
            pt = PromptPattern.from_watsonx(credentials=creds, name="prompt_builder", template = bt_prompt_template)
            options = Options(
                watsonx_template=pt
            )
            # Create inputs list
            input = format_from_template(template_path = back_trans_input_template_path, data_point =bt_data_point)
            inputs.append(input)

            tokenized_metadata = model.tokenize([bt_data_point["metadata"]], return_tokens=True)
            tokenized_metadata_lst.append(tokenized_metadata)

        # Generate back translation
        bt_responses = model.generate_async(prompts = inputs, options=options, ordered=True, hide_progressbar=True)
        print(f"inputs:{len(inputs)}, tokenized_metadata_lst:{len(tokenized_metadata_lst)} ") # TEST

        # Parse BAM results
        back_trans_input_lst = []
        back_trans_gen_text_lst = []
        mean_input_tokens_lst = []
        for input, tokenized_metadata, bt_response in zip(inputs,tokenized_metadata_lst,bt_responses):
            if bt_response is not None:
                back_trans_input_lst.append(bt_response.input_text)
                back_trans_gen_text_lst.append(bt_response.generated_text)

                init_pos = bt_response.input_token_count - tokenized_metadata[0].token_count
                sub_bt_response = bt_response.input_tokens[init_pos:]+bt_response.input_tokens[:init_pos]
                
                logprob_total = 0
                for input_token in (sub_bt_response):
                    if input_token.logprob is not None: logprob_total += input_token.logprob
                mean_input_tokens = (logprob_total/len(sub_bt_response))
            else:
                back_trans_input_lst.append(GEN_ERROR)
                back_trans_gen_text_lst.append(GEN_ERROR)

                init_pos = -1
                sub_bt_response = -1
                mean_input_tokens = -0.999999

            print(f"\tbt mean: {mean_input_tokens}") #TEST
            mean_input_tokens_lst.append(mean_input_tokens)

        # # TEST
        # print(f"back_trans_input_lst:{len(back_trans_input_lst)}, \
        #       back_trans_gen_text_lst:{len(back_trans_gen_text_lst)}, \
        #       mean_input_tokens_lst: {len(mean_input_tokens_lst)}, \
        #       candidates: {len(data_point['instruction_candidates'])}")
        
        # Assign bt results to the instruction candidates
        for candidate, back_trans_input,back_trans_gen_text, mean_input_tokens in zip(data_point['instruction_candidates'],back_trans_input_lst,back_trans_gen_text_lst,mean_input_tokens_lst):
            candidate["back_trans_input"] = back_trans_input
            candidate["back_trans_gen_text"] = back_trans_gen_text
            candidate["back_trans_input_tokens_mean"] = mean_input_tokens
            
        # Sleep routine 1 s per datapoint (when we get creds with an increased concurrency limit, add the sleep after the file)
        # seconds = 1
        # print(f"Sleeping {seconds} ...")
        # time.sleep(seconds)

    return data


def create_summary(data:[],
                   params: GenerateParams, 
                   summary_template_path:str,
                   summary_examples_path:str,
                   dotenv_path:str,
                   model_id:str = "google/flan-ul2"):
    
    creds = create_credentials(dotenv_path)
    model = create_model(params=params, model_id=model_id, creds=creds)

    # Load general templates, instruction, and data examples
    summary_template = load_txt_file(summary_template_path)
    # print(summary_template)
    instruction_examples = load_local_file_as_json(file_path = summary_examples_path)
    # print(instruction_examples)

    # Prompt config
    pt = PromptPattern.from_watsonx(credentials=creds, name="prompt_builder", template = summary_template)
    # print(pt)
    options = Options(
        watsonx_template=pt,
        watsonx_data = instruction_examples
    )

    inputs = []
    for api in data: inputs.append(api["api_description"]) 

    responses = model.generate(inputs, options=options)

    # for api, resp in zip(data,responses):
    #     print(f"API: {api['api_name']}")
    #     if resp is not None:
    #         print(f"INPUT_TEXT:{resp.input_text}\
    #             \n\nGENERATED_TEXT:{resp.generated_text}")
    #     else:
    #         print("NONE")
    #     print("---------------------------")
            
    for api, resp in zip(data,responses):
        if resp is not None:
            api["desc_summary"] = resp.generated_text
        else:
            api["desc_summary"] = ""

    return data
    

