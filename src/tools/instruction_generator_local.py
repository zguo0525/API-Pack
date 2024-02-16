import pystache
import json
import os
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

import random
import torch
import pyarrow
import transformers
import tensor_parallel as tp
import concurrent.futures

from tools.data_manger import load_local_file_as_json, load_txt_file

GEN_ERROR = "<<ERROR: INSTRUCTION NOT GENERATED>>"

def get_ppl(predictions, model, tokenizer, prediction0, batch_size: int=1, add_start_token: bool=True, device="cuda", max_length=None):
    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer.apply_chat_template(
        predictions,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings
    attn_masks = torch.ones(encoded_texts.shape, dtype=torch.long).to(device)

    encodings0 = tokenizer.apply_chat_template(
        prediction0,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    )

    encoded_texts0 = encodings0
    
    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    mean_logprobs = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for start_index in range(0, len(encoded_texts), batch_size):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        encoded_batch0 = encoded_texts0[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        # Calculate mean log probability for the last ten tokens of each sequence in the batch
        for batch_idx in range(shift_logits.size(0)):

            NUM_LAST_TOKENS = len(encoded_batch[batch_idx]) - len(encoded_batch0[batch_idx])
            
            # Extract the last NUM_LAST_TOKENS tokens' logits, labels, and attention mask
            last_tokens_logits = shift_logits[batch_idx, -NUM_LAST_TOKENS:, :]
            last_tokens_labels = shift_labels[batch_idx, -NUM_LAST_TOKENS:]
            last_tokens_attention_mask = shift_attention_mask_batch[batch_idx, -NUM_LAST_TOKENS:]
        
            # Initialize variables for total log probability and count of valid tokens
            logprob_total = 0
            valid_tokens_count = 0
        
            # Iterate over each of the last NUM_LAST_TOKENS tokens
            for token_idx in range(NUM_LAST_TOKENS):
                # Calculate log softmax of logits for the current token
                token_logprob = torch.log_softmax(last_tokens_logits[token_idx, :], dim=-1)
        
                # Gather the log probabilities of the actual tokens (from labels)
                token_logprob = token_logprob[last_tokens_labels[token_idx]].item()
        
                # Determine if the current token is a valid token or padding
                is_valid_token = last_tokens_attention_mask[token_idx].item()
        
                # Add the log probability if it's a valid token
                if is_valid_token:
                    logprob_total += token_logprob
                    valid_tokens_count += 1

            # Calculate the mean log probability for the last ten tokens
            mean_logprob = logprob_total / valid_tokens_count if valid_tokens_count > 0 else float('nan')
            mean_logprobs.append(mean_logprob)

    return {"mean_logprobs": mean_logprobs}

def parse_candidates_info(responses, input_, input_ids, tokenizer, use_greedy_method=False):
    idx_lst = []
    instructions = []
    inputs = []
    means = []

    input_length = input_ids.shape[1] if tokenizer.model_max_length > input_ids.shape[1] else tokenizer.model_max_length

    for idx, resp in enumerate(responses, start=1):
        print(f"Processing response {idx}")  # Debugging
        if resp is not None:
            generated_tokens = resp.sequences[0][input_length:]
            print(f"Number of generated tokens: {len(generated_tokens)}")  # Debugging
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if use_greedy_method:
                logprob_total = sum(torch.log_softmax(score, dim=-1).max(dim=-1).values for score in resp.scores[input_length:])
                print(f"Greedy method logprob_total: {logprob_total}")  # Debugging
            else:
                logprob_total = 0
                print(f"Entering token loop, scores length: {len(resp.scores[input_length:])}")  # Debugging
                for token_idx, score in enumerate(resp.scores[input_length:]):
                    token_logprob = torch.log_softmax(score, dim=-1)[0, generated_tokens[token_idx]]
                    if token_logprob == float('-inf'):
                        token_logprob = torch.tensor(-1e9)  # Replace -inf with a large negative number
                    logprob_total += token_logprob
                    print(f"Token index: {token_idx}, Token ID: {generated_tokens[token_idx]}, Token log probability: {token_logprob}")  # Debugging

            if isinstance(logprob_total, torch.Tensor):
                mean_logprob = logprob_total.item() / len(generated_tokens) if generated_tokens.numel() > 0 else float('nan')
            else:
                mean_logprob = logprob_total / len(generated_tokens) if generated_tokens.numel() > 0 else float('nan')
            print(f"Mean log probability: {mean_logprob}")  # Debugging

            idx_lst.append(idx)
            instructions.append(generated_text)
            inputs.append(input_)
            means.append(mean_logprob)
        else:
            idx_lst.append(idx)
            instructions.append("GEN_ERROR")
            inputs.append("GEN_ERROR")
            means.append(float('nan'))

    df_candidates = pd.DataFrame(list(zip(idx_lst, instructions, inputs, means)),
                                 columns=['idx', 'candidate', 'input_text', 'gen_tokens_mean'])
    
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

def generate_instructions(data:[],
                          model,
                          tokenizer,
                          inputs_template_path:str, 
                          prompt_template_path:str, 
                          ins_ex_path:str,
                          dotenv_path:str,
                          candidates_max:int = 5):

    prompt_prefix = "Your task is to create a user query that effectively utilizes a specific API. The API's functionality, description, and name will be provided to you. Your query should be designed in a way that makes the best use of this API's unique capabilities. When crafting your query, focus on:\n\n1. **API Name Integration:** Clearly include the API's name in your query to ensure relevance.\n2. **Specificity:** Replace broad or vague terms with precise, concrete details relevant to the API's purpose.\n3. **Conciseness:** Keep your query as brief as possible while still fully conveying the needed information. Avoid unnecessary verbosity.\n4. **Excluding API Endpoint:** Do not include the API's endpoint in your query; focus only on the user's need and how the API fulfills it.\n\nCreate a query that a user might realistically use when interacting with the given API. Think about typical scenarios or problems that the API is designed to solve and formulate your query accordingly.\n\nExamples for practice:\n"

    instruction_examples_data = load_local_file_as_json(file_path=ins_ex_path)['list']
    #print("instruction_examples_data:", instruction_examples_data)

    # Check if the data loaded is a list and has at least three items
    if isinstance(instruction_examples_data, list) and len(instruction_examples_data) >= 1:
        # Randomly select one, two, or three items from the list
        select = min(len(instruction_examples_data), 3)
        selected_examples = random.sample(instruction_examples_data, select)

        for selected_example in selected_examples:
            template = "###Input:\nFunctionality: {functionality}\nDescription: {description}\nEndpoint: {endpoint}\nAPI: {api_name}\n###Output:\n{output}"

            filled_template = template.format(
                functionality=selected_example["functionality"],
                description=selected_example["description"],
                endpoint=selected_example["endpoint"],
                api_name=selected_example["API name"],
                output=selected_example["output"]
            )

            prompt_prefix = prompt_prefix + filled_template + "\n\n"

    else:
        selected_examples = None
        prompt_prefix = "Your task is to create a user query that effectively utilizes a specific API. The API's functionality, description, and name will be provided to you. Your query should be designed in a way that makes the best use of this API's unique capabilities. When crafting your query, focus on:\n\n1. **API Name Integration:** Clearly include the API's name in your query to ensure relevance.\n2. **Specificity:** Replace broad or vague terms with precise, concrete details relevant to the API's purpose.\n3. **Conciseness:** Keep your query as brief as possible while still fully conveying the needed information. Avoid unnecessary verbosity.\n4. **Excluding API Endpoint:** Do not include the API's endpoint in your query; focus only on the user's need and how the API fulfills it.\n\nCreate a query that a user might realistically use when interacting with the given API. Think about typical scenarios or problems that the API is designed to solve and formulate your query accordingly.\n\nRemember, the goal is to demonstrate how a user would benefit from this specific API in a realistic scenario, using precise and clear language. Here is the actual task for you:\n\n"

    # print("selected_examples:", selected_examples)
    # print("prompt_prefix:", prompt_prefix)
    
    # Candidates generation
    for data_point in data:
        # Create input
        if selected_examples is not None:
            input_template = "###Input:\nFunctionality: {functionality}\nDescription: {description}\nEndpoint: {endpoint}\nAPI: {api_name}\n###Output:\n"

            input_ = input_template.format(
                functionality=data_point["functionality"],
                description=data_point["description"],
                endpoint=data_point["endpoint"],
                api_name=data_point["api_name"]
            )

            input = prompt_prefix + "Remember, the goal is to demonstrate how a user would benefit from this specific API in a realistic scenario, using precise and clear language. Here is the actual task for you:\n\n" + input_
            
            # remove the extra strings for the input_
            # which later will be used as the input_text
            len1 = len("###Input:\n")
            len2 = len("\n###Output:\n")
            input_ = input_[len1:-len2]
            
        else:
            input_ = data_point["api_description"]
            input = prompt_prefix + input_

        messages=[{ 'role': 'user', 'content': f"{input}"}]
        # Apply chat template and tokenize inputs
        inputs = tokenizer.apply_chat_template(
            messages,
            padding=False,
            return_tensors="pt",
        ).to("cuda")

        # Manually create an attention mask if it's not returned
        attention_mask = torch.ones(inputs.shape, dtype=torch.long).to("cuda")

        responses = []
        
        for n in range(candidates_max):

            condition_met = True
            attempts = 0  # Initialize a counter

            while condition_met and attempts < 5:
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
                        
                        outputs = future.result(timeout=40)  # Timeout set to 40 seconds
            
                except concurrent.futures.TimeoutError:
                    print(f"Timeout occurred. Skipping...")
                    attempts += 1  # Increment attempts counter
                    continue
                except RuntimeError as e:
                    print(f"Runtime error: {e}")
                    attempts += 1  # Increment attempts counter
                    continue
        
                # Process generated tokens
                generated_tokens = outputs.sequences[0][inputs.shape[1]:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
                if ":" in generated_text:
                    attempts += 1  # Increment attempts counter
                    continue  # If colon is present, continue to regenerate
                else:
                    condition_met = False  # Exit the loop if no colon is found
        
            # Here, handle the case where a suitable text wasn't generated after 5 attempts
            if attempts >= 5:
                print("Maximum attempts reached, moving to next candidate.")
                print("generated text:", generated_text)
                responses.append(outputs)
                continue
                
            responses.append(outputs)

        # Calculate mean log probs
        # print_candidates_info(responses=responses) #TEST
        df_candidates = parse_candidates_info(responses, input_, inputs, tokenizer, use_greedy_method=False)
        print("=df_candidates=")
        print(df_candidates) # TEST
        # for index, row in df_candidates.iterrows():
        #     print(f"--------{index}----------------{index}-------")
        #     print(row['mean_logprob'])
        #     print(row['input_text'])
        #     print(row['candidate'])
        #     time.sleep(0.25)
            
        # Save instruction candidates
        data_point["instruction_candidates"] = df_candidates.to_dict(orient="records")
        data_point["instruction_prompt"] = input

    return data

# Back translation -> Regenerate the metadata based on the candidate and respective ground truth (real metadata).
def back_translation(data:[],
                     model,
                     tokenizer,
                     back_trans_prompt_template_path:str,
                     back_trans_input_template_path:str,
                     ins_ex_path:str,
                     dotenv_path:str):

    ### load in-context example
    instruction_examples_data = load_local_file_as_json(file_path=ins_ex_path)['list']

    # Check if the data loaded is a list and has at least three items
    if isinstance(instruction_examples_data, list) and len(instruction_examples_data) >= 1:
        # Randomly select one, two, or three items from the list
        select = min(len(instruction_examples_data), 3)
        selected_examples = random.sample(instruction_examples_data, select)

        example_prompt = ""

        for selected_example in selected_examples:
            
            template = "###Input:\n{output}\n###Output:\nFunctionality: {functionality}\nDescription: {description}\nEndpoint: {endpoint}\nAPI: {api_name}"

            filled_template = template.format(
                functionality=selected_example["functionality"],
                description=selected_example["description"],
                endpoint=selected_example["endpoint"],
                api_name=selected_example["API name"],
                output=selected_example["output"]
            )

            example_prompt = example_prompt + filled_template + "\n\n"

    else:
        selected_examples = None
        example_prompt = ""

    prompt_prefix = "Your task involves a reverse-engineering process where you will analyze a user query to infer specific details about an API endpoint. Based on the given user query, you are expected to:\n\n1. **Identify the Endpoint's Identifier:** Derive the endpoint identifier that aligns with the functionality implied by the user query.\n2. **Determine Endpoint Functionality:** Interpret the user query to understand and describe the functionality of the endpoint.\n3. **Describe the Endpoint:** Provide a detailed description of the endpoint based on the needs and context presented in the user query.\n4. **Specify the API Name:** Identify and state the name of the API to which this endpoint belongs, as suggested by the user query.\n\nYour response should clearly articulate these four elements (identifier, functionality, description, API name) in a manner that reflects an accurate understanding of the user query. Consider the query as a real-world scenario or problem that the endpoint is designed to address."

    if selected_examples is not None:
        
        prompt_prefix = prompt_prefix + "\n\nExamples for practice:\n\n" + example_prompt + "\nThe goal is to showcase your ability to connect a user's needs with the appropriate API endpoint, demonstrating an understanding of how the endpoint’s features align with user requirements. Your response should be precise, insightful, and reflective of the query's implications.\nHere is the actual task for you:\n\n"

    else:

        prompt_prefix = prompt_prefix + "\nThe goal is to showcase your ability to connect a user's needs with the appropriate API endpoint, demonstrating an understanding of how the endpoint’s features align with user requirements. Your response should be precise, insightful, and reflective of the query's implications.\nHere is the actual task for you:\n\n"

    # ['idx','candidate','input_text','mean']
    for data_point in data:
        # Create inputs and tokenizers for bt
        inputs = []
        tokenized_metadata_lst = []
        for candidate in data_point["instruction_candidates"]:  
        
            final_template_input = "###Input:\n{candidate}"
            final_template_output = "\n###Output:\n{metadata}"

            final_filled_template_input = final_template_input.format(
                candidate=candidate['candidate']
            )
            final_filled_template_output = final_template_output.format(
                metadata=candidate['input_text']
            )

            total_prompt = prompt_prefix + final_filled_template_input + final_filled_template_output

            message0=[{ 'role': 'user', 'content': f"{prompt_prefix + final_filled_template_input}"}]
            messages=[{ 'role': 'user', 'content': f"{total_prompt}"}]

            # print("prompt_prefix:", prompt_prefix + final_filled_template_input)
            # print("total_prompt:", total_prompt)

            mean_logprob = get_ppl(messages, model, tokenizer, message0)["mean_logprobs"][0]
        
            # Assign bt results to the instruction candidates
            candidate["back_trans_input"] = total_prompt
            candidate["back_trans_gen_text"] = "empty"
            candidate["back_trans_input_tokens_mean"] = mean_logprob

    return data