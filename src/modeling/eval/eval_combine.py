import json
import os
import re
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import pandas as pd
import warnings

from urllib.parse import unquote

threshold = 0.9

def do_java_request_urls_match(code_str1, code_str2):
    """
    Extracts the URL part from Request.Builder sections in Java code strings and checks if they match.

    Args:
    code_str1 (str): The first Java code string.
    code_str2 (str): The second Java code string.

    Returns:
    bool: True if the URLs match, False otherwise.
    """
    # Regular expression to extract URL from the Request.Builder section
    pattern = r'\.url\("([^"]+)"\)'
    
    # Extract URLs
    url1_match = re.search(pattern, code_str1)
    url2_match = re.search(pattern, code_str2)

    if not url1_match or not url2_match:
        return False  # One or both URLs are missing

    url1 = url1_match.group(1)
    url2 = url2_match.group(1)

    # Compare URLs
    return similarity(url1, url2) > threshold

def do_connection_and_request_domains_match(code_str1, code_str2):
    """
    Extracts the domain part from HTTPConnection or HTTPSConnection instantiation and conn.request call in Python code strings and checks if they match.

    Args:
    code_str1 (str): The first Python code string.
    code_str2 (str): The second Python code string.

    Returns:
    bool: True if the domains match, False otherwise.
    """
    # Regular expression to extract domain from the connection strings
    conn_pattern = r'http\.client\.HTTP[S]?Connection\("([^"]+)"\)'
    request_pattern = r'conn\.request\("[GETPOSTPUTDELETE]*", "([^"]+)"'

    # Extract domains from HTTP[S]Connection
    domain1_conn_match = re.search(conn_pattern, code_str1)
    domain2_conn_match = re.search(conn_pattern, code_str2)

    # Extract URL part from conn.request
    domain1_request_match = re.search(request_pattern, code_str1)
    domain2_request_match = re.search(request_pattern, code_str2)

    if not domain1_conn_match or not domain2_conn_match or not domain1_request_match or not domain2_request_match:
        return False  # One or more domains/URLs are missing

    domain1_conn = domain1_conn_match.group(1)
    domain2_conn = domain2_conn_match.group(1)
    domain1_request = unquote(domain1_request_match.group(1))
    domain2_request = unquote(domain2_request_match.group(1))

    # Normalize the domains by removing trailing periods and slashes
    domain1_conn = domain1_conn.rstrip('.').rstrip('/')
    domain2_conn = domain2_conn.rstrip('.').rstrip('/')
    domain1_request = domain1_request.rstrip('.').rstrip('/')
    domain2_request = domain2_request.rstrip('.').rstrip('/')

    # Compare both connection and request domains
    return (similarity(domain1_conn, domain2_conn) > threshold) and (similarity(domain1_request, domain2_request) > threshold)

def are_curl_urls_equal(curl_command1, curl_command2):
    """
    Check if the URLs in two curl commands are equal.

    Args:
    curl_command1 (str): The first curl command.
    curl_command2 (str): The second curl command.

    Returns:
    bool: True if the URLs are equal, False otherwise.
    """
    # Extract URLs from the curl commands
    url1_match = re.search(r"--url\s+(\S+)", curl_command1)
    url2_match = re.search(r"--url\s+(\S+)", curl_command2)

    if not url1_match or not url2_match:
        return False  # One or both URLs are missing

    url1 = unquote(url1_match.group(1))  # Decode URL
    url2 = unquote(url2_match.group(1))  # Decode URL

    # Compare URLs
    return similarity(url1, url2) > threshold

def are_go_request_urls_equal(go_code1, go_code2):
    """
    Extracts the URL from http.NewRequest method calls in Go code strings and checks if they match.

    Args:
    go_code1 (str): The first Go code string.
    go_code2 (str): The second Go code string.

    Returns:
    bool: True if the URLs match, False otherwise.
    """
    # Regular expression to extract URL variable assignment and its use in http.NewRequest
    url_var_pattern = r'url\s*:=\s*"([^"]+)"'
    req_pattern = r'http\.NewRequest\("[A-Z]+",\s*url,'

    # Extract URL variable assignment
    url1_var_match = re.search(url_var_pattern, go_code1)
    url2_var_match = re.search(url_var_pattern, go_code2)

    # Check if URL is used in http.NewRequest
    url1_req_match = re.search(req_pattern, go_code1)
    url2_req_match = re.search(req_pattern, go_code2)

    if not all([url1_var_match, url2_var_match, url1_req_match, url2_req_match]):
        return False  # URL assignment or usage in NewRequest is missing

    # Extract URLs
    url1 = url1_var_match.group(1)
    url2 = url2_var_match.group(1)
    
    return similarity(url1, url2) > threshold

def are_js_xhr_urls_equal(js_code1, js_code2):
    """
    Extracts the URL from xhr.open method calls in JavaScript code strings and checks if they match.

    Args:
    js_code1 (str): The first JavaScript code string.
    js_code2 (str): The second JavaScript code string.

    Returns:
    bool: True if the URLs match, False otherwise.
    """
    # Regular expression to extract URL from xhr.open
    pattern = r'xhr\.open\("[A-Z]+",\s*"([^"]+)"'

    # Extract URLs
    url1_match = re.search(pattern, js_code1)
    url2_match = re.search(pattern, js_code2)

    if not url1_match or not url2_match:
        return False  # One or both URLs are missing

    # Extract URLs
    url1 = url1_match.group(1)
    url2 = url2_match.group(1)

    # Compare URLs
    return similarity(url1, url2) > threshold

def are_libcurl_urls_equal(c_code1, c_code2):
    """
    Extracts the URL from curl_easy_setopt function calls in C code strings for CURLOPT_URL and checks if they match.

    Args:
    c_code1 (str): The first C code string.
    c_code2 (str): The second C code string.

    Returns:
    bool: True if the URLs match, False otherwise.
    """
    # Regular expression to extract URL from curl_easy_setopt with CURLOPT_URL
    pattern = r'curl_easy_setopt\(\s*hnd,\s*CURLOPT_URL,\s*"([^"]+)"\s*\)'

    # Extract URLs
    url1_match = re.search(pattern, c_code1)
    url2_match = re.search(pattern, c_code2)

    if not url1_match or not url2_match:
        return False  # One or both URLs are missing

    # Extract URLs
    url1 = url1_match.group(1)
    url2 = url2_match.group(1)

    # Compare URLs
    return similarity(url1, url2) > threshold

def are_node_request_urls_equal(node_code1, node_code2):
    """
    Extracts the URL from options object in request library calls in Node.js code strings and checks if they match.

    Args:
    node_code1 (str): The first Node.js code string.
    node_code2 (str): The second Node.js code string.

    Returns:
    bool: True if the URLs match, False otherwise.
    """
    # Regular expression to extract URL from options object in request call
    pattern = r'url:\s*\'([^\']+)\''

    # Extract URLs
    url1_match = re.search(pattern, node_code1)
    url2_match = re.search(pattern, node_code2)

    if not url1_match or not url2_match:
        return False  # One or both URLs are missing

    # Extract URLs
    url1 = url1_match.group(1)
    url2 = url2_match.group(1)

    # Compare URLs
    return similarity(url1, url2) > threshold

def are_php_curl_urls_equal(php_code1, php_code2):
    """
    Extracts the URL from curl_setopt_array or curl_setopt function calls in PHP code strings for CURLOPT_URL and checks if they match.

    Args:
    php_code1 (str): The first PHP code string.
    php_code2 (str): The second PHP code string.

    Returns:
    bool: True if the URLs match, False otherwise.
    """
    # Regular expression to extract URL from CURLOPT_URL in curl_setopt_array or curl_setopt
    pattern = r'CURLOPT_URL\s*=>\s*"([^"]+)"'

    # Extract URLs
    url1_match = re.search(pattern, php_code1)
    url2_match = re.search(pattern, php_code2)

    if not url1_match or not url2_match:
        return False  # One or both URLs are missing

    # Extract URLs
    url1 = url1_match.group(1)
    url2 = url2_match.group(1)

    # Compare URLs
    return similarity(url1, url2) > threshold

def are_ruby_net_http_urls_equal(ruby_code1, ruby_code2):
    """
    Extracts the URL from URI object initialization in Ruby code strings using Net::HTTP and checks if they match.

    Args:
    ruby_code1 (str): The first Ruby code string.
    ruby_code2 (str): The second Ruby code string.

    Returns:
    bool: True if the URLs match, False otherwise.
    """
    # Regular expression to extract URL from URI initialization
    pattern = r'URI\("([^"]+)"\)'

    # Extract URLs
    url1_match = re.search(pattern, ruby_code1)
    url2_match = re.search(pattern, ruby_code2)

    if not url1_match or not url2_match:
        return False  # One or both URLs are missing

    # Extract URLs
    url1 = url1_match.group(1)
    url2 = url2_match.group(1)

    # Compare URLs
    return similarity(url1, url2) > threshold

def are_swift_urlsession_urls_equal(swift_code1, swift_code2):
    """
    Extracts the URL from NSMutableURLRequest initialization in Swift code strings and checks if they match.

    Args:
    swift_code1 (str): The first Swift code string.
    swift_code2 (str): The second Swift code string.

    Returns:
    bool: True if the URLs match, False otherwise.
    """
    # Regular expression to extract URL from NSMutableURLRequest or NSURL initialization
    pattern = r'NSURL\(string:\s*"([^"]+)"\)'

    # Extract URLs
    url1_match = re.search(pattern, swift_code1)
    url2_match = re.search(pattern, swift_code2)

    if not url1_match or not url2_match:
        return False  # One or both URLs are missing

    # Extract URLs
    url1 = url1_match.group(1)
    url2 = url2_match.group(1)

    # Compare URLs
    return similarity(url1, url2) > threshold

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

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

def read_json_file_line_by_line(file_path):
    """
    Reads a file line by line, attempting to parse each line as a JSON object.

    :param file_path: The path to the file that should be read.
    :return: A list of parsed JSON objects.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            try:
                json_object = json.loads(line)
                data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
    return data

def extract_domain_api_call(data_str):
    # Regular expression patterns to extract domain and API call
    domain_pattern = re.compile(r"\**domain\**:(.*?)\n\**api_call\**", re.DOTALL)
    api_call_pattern = re.compile(r"\**api_call\**:(.*?)\n\**api_provider\**", re.DOTALL)
    
    # Search for patterns and extract the necessary parts
    domain_match = domain_pattern.search(data_str)
    api_call_match = api_call_pattern.search(data_str)
    
    domain = domain_match.group(1) if domain_match else ""
    api_call = api_call_match.group(1) if api_call_match else ""
    
    return domain, api_call

def compare_domain_api(data1, data2, show=False):
    # Extract the domain and API call from both data strings
    domain1, api_call1 = extract_domain_api_call(data1)
    domain2, api_call2 = extract_domain_api_call(data2)
    
    # Compare domains and API calls
    domain_match = similarity(domain1, domain2) > threshold  # You can adjust the threshold
    api_call_match = similarity(api_call1, api_call2) > threshold

    if not api_call_match and show:
        print(similarity(api_call1, api_call2))
        print("output:", domain1, api_call1)
        print("gtruth:", domain2, api_call2)

    # if api_call_match:
    #     print("data1:", data1, "data2:", data2, "len:", len(api_call1))
    
    return api_call1, api_call2, api_call_match

def eval(model_name, trained_data, test_data, lang):
    # Define the directory where the file is located
    test_dir = f"../inference/{model_name}/{trained_data}"
    
    # Initialize dictionaries to track matches and totals for all APIs across all chunks
    domain_match_count = {}
    api_call_match_count = {}
    total_count = {}
    
    # Construct the file name for the chunk and API names
    test_file = f'{test_data}'
    test_path = os.path.join(test_dir, test_file)

    parsed_data = read_json_file_line_by_line(test_path)
    original_question_data = get_questions(f"../instr_data/{lang}/{test_data[:-5]}")
    #print(original_question_data)

    for data, origin_data in zip(parsed_data, original_question_data):
        try:
            api_name = data['api_name']
            # Ensure each API has a counter across all chunks
            endpoint = origin_data['api_call_data']['path']
            clean_endpoint = re.sub(r"/\{.*?\}", "", endpoint)
        except:
            continue
        
        if api_name not in total_count:
            total_count[api_name] = 0
            domain_match_count[api_name] = 0
            api_call_match_count[api_name] = 0

        api_call1, api_call2, api_call_match = compare_domain_api(data['output'], data['ground_truth'])
        # print(api_call1)
        # print(api_call2)

        # Increment counters for the API
        total_count[api_name] += 1
        if lang == "cleaned_curl":
            endpoint_match = are_curl_urls_equal(api_call1, api_call2)
        elif lang == "cleaned_python":
            endpoint_match = do_connection_and_request_domains_match(api_call1, api_call2)
        elif lang == "cleaned_java":
            endpoint_match = do_java_request_urls_match(api_call1, api_call2)
        elif lang == "cleaned_go":
            endpoint_match = are_go_request_urls_equal(api_call1, api_call2)
        elif lang == "cleaned_javascript":
            endpoint_match = are_js_xhr_urls_equal(api_call1, api_call2)
        elif lang == "cleaned_libcurl":
            endpoint_match = are_libcurl_urls_equal(api_call1, api_call2)
        elif lang == "cleaned_node":
            endpoint_match = are_node_request_urls_equal(api_call1, api_call2)
        elif lang == "cleaned_php":
            endpoint_match = are_php_curl_urls_equal(api_call1, api_call2)
        elif lang == "cleaned_ruby":
            endpoint_match = are_ruby_net_http_urls_equal(api_call1, api_call2)
        elif lang == "cleaned_swift":
            endpoint_match = are_swift_urlsession_urls_equal(api_call1, api_call2)
        if endpoint_match: 
            domain_match_count[api_name] += 1
        if api_call_match and endpoint_match:
            api_call_match_count[api_name] += 1
    
    # Aggregate counters for overall accuracy
    overall_domain_match_count = sum(domain_match_count.values())
    overall_api_call_match_count = sum(api_call_match_count.values())
    overall_total_count = sum(total_count.values())
    
    # Calculate and print the overall accuracy
    overall_domain_match_acc = overall_domain_match_count / overall_total_count if overall_total_count > 0 else 0
    overall_api_call_match_acc = overall_api_call_match_count / overall_total_count if overall_total_count > 0 else 0
    
    print('-------------------------------------------------')
    print("<<model>>:", model_name, "<<train data>>:", trained_data.replace("CodeLlama-13b-hf_", ""), "<<test data>>:", test_data.replace("total_testing_total_", ""))
    print("Number of testing APIs:", len(list(total_count.keys())))
    #print(f"Overall Endpoint Match Accuracy: {overall_domain_match_acc:.3f}")
    print(f"Overall API Call Match Accuracy: {overall_api_call_match_acc:.3f}")
    
    # # Additionally, print the accuracy for each API across all chunks
    # for api_name in total_count:
    #     domain_match_acc = domain_match_count[api_name] / total_count[api_name] if total_count[api_name] > 0 else 0
    #     api_call_match_acc = api_call_match_count[api_name] / total_count[api_name] if total_count[api_name] > 0 else 0
    #     print('-------------------------------------------------')
    #     print(f"API: {api_name}")
    #     print(f"Domain Match Accuracy: {domain_match_acc:.2f}")
    #     print(f"API Call Match Accuracy: {api_call_match_acc:.2f}")

    # Plotting
    # Extract API names, Domain Match Accuracies, and API Call Match Accuracies
    apis = list(total_count.keys())
    domain_match_accuracies = [domain_match_count[api] / total_count[api] for api in apis]
    api_call_match_accuracies = [api_call_match_count[api] / total_count[api] for api in apis]
    
    def autolabel(bars):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Assuming a fixed width per API grouping for the plot
    fixed_width_per_api = 0.4  # This can be adjusted based on actual needs and aesthetics
    fig_width = len(apis) * fixed_width_per_api
    
    fig, ax = plt.subplots(figsize=(fig_width, 4))  # Dynamically setting the figure width
    bar_width = 0.4
    r1 = range(len(apis))
    r2 = [x + bar_width for x in r1]
    
    # Make the plot
    bars1 = ax.bar(r1, domain_match_accuracies, color='b', width=bar_width, label='Category Match Acc')
    bars2 = ax.bar(r2, api_call_match_accuracies, color='r', width=bar_width, label='API Call Match Acc')
    
    # Add some text for labels and title
    ax.set_xlabel('APIs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracies', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} trained on {trained_data[:-5]} and test on {test_data[:-10]}', fontsize=12)
    ax.set_xticks([r + bar_width/2 for r in range(len(apis))])
    ax.set_xticklabels(apis, rotation=90)
    ax.legend()
    
    # Insert overall accuracies in the plot
    plt.axhline(y=overall_domain_match_acc, color='b', linestyle='dashed', label='Overall Category Match Acc')
    plt.axhline(y=overall_api_call_match_acc, color='r', linestyle='dashed', label='Overall API Call Match Acc')
    
    # Call the function to label the bars
    autolabel(bars1)
    autolabel(bars2)

    save_file_path = f"{model_name}_{trained_data}_{test_data}.png"
    # Extract folder name from the file path
    folder_name = os.path.dirname(save_file_path)
    
    # Create the folder if it does not exist
    if not os.path.exists(folder_name) and folder_name != '':
        os.makedirs(folder_name)

    # Ignore the specific UserWarning related to tight_layout
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # Save the figure
        plt.tight_layout()
        plt.savefig(save_file_path)
        plt.close()

    return len(list(total_count.keys())), overall_domain_match_acc, overall_api_call_match_acc

if __name__ == "__main__":

    #method = "cleaned_sub_combined"
    #method = "cleaned_curl"
    method = "cleaned_sub_50k"
    #method = "magic_coder"
    langs = ["cleaned_curl", "cleaned_python", "cleaned_java", "cleaned_go", "cleaned_javascript", "cleaned_libcurl", "cleaned_node", "cleaned_php", "cleaned_ruby", "cleaned_swift"]#[1:2]
    #sizes = [210328]
    sizes = [236380]
    #sizes = [186380]
    num_of_IC = 3
    num_of_rerank = 10
    #model_names = ["granite-20b-code", "granite-13b-base-v2", "Llama-2-13b-hf", "CodeLlama-13b-hf"]
    #model_names = ["CodeLlama-13b-hf", "Mistral-7B-v0.1"]
    #model_names = ["Llama-2-13b-hf"]
    model_names = ["CodeLlama-13b-hf"]
                   
    data = []
    
    for size in sizes:
        for lang in langs:
            for model_name in model_names:
                trained_datas = [#f"total_{lang}/tokenized_simple_data_{model_name}_{size}",
                                 f"{method}/tokenized_{method}_data_{model_name}_{size}/{lang}"]
                # trained_datas = [f"{method}/tokenized_combined_data_{model_name}_{size}"]
                for trained_data in trained_datas:
                    test_datas = [f"total_testing_{lang}_level_1.json.json", 
                                  f"total_testing_{lang}_level_2.json.json", 
                                  f"total_testing_{lang}_level_3.json.json", 
                                  # f"total_testing_total_{lang}_level_1_IC_3.json.json",
                                  # f"total_testing_total_{lang}_level_2_IC_3.json.json",
                                  # f"total_testing_total_{lang}_level_3_IC_3.json.json", 
                                  f"total_testing_{lang}_level_1_retrieval_IC_3.json.json", 
                                  f"total_testing_{lang}_level_2_retrieval_IC_3.json.json", 
                                  f"total_testing_{lang}_level_3_retrieval_IC_3.json.json"]
                    for test_data in test_datas:
                        api_nums, domain_acc, api_acc = eval(model_name, trained_data, test_data, lang)
                        data.append([model_name[:-3], trained_data[len("cleaned_curl/tokenized_"):], test_data[6+8:-10], api_nums, domain_acc, api_acc])

    df = pd.DataFrame(data, columns=['Model', 'Training Data', 'Test Data', 'API Numbers', 'Domain Accuracy', 'API_Call Accuracy'])
    print(df)