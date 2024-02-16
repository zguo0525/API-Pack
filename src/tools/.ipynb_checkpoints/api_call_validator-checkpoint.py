import re

# Validate URL

    # Query parameters might be bewteen <>
        # https://us-south.ml.cloud.ibm.com/ml/v4/trainings/:training_id?hard_delete=<boolean>&version=2020-09-01&space_id=<string>&project_id=<string>
    # Domain names may contain $
        # https://$BUCKET.s3.$REGION.cloud-object-storage.appdomain.cloud/?replication
    # Path parameters might be between {} <>
        # http://schematics.cloud.ibm.com/v2/settings/agents/{agent_id}
        # https://schematics.cloud.ibm.com/v2/agents/<agent_id>/prs

def validate_url(url:str, include_protocol:bool = True):
    if include_protocol:
        # url_pattern = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$" 
        url_example_pattern = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=$<>{}]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=<>{}]*)$" 
    else:
        # url_pattern = "^[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
        url_example_pattern = "^[-a-zA-Z0-9@:%._\\+~#=$<>{}]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=<>{}]*)$"

    res = re.match(url_example_pattern, url)

    return res # True , # False

# Extract URL from a string
def extract_url_from_str(text:str):
    # url_extract_pattern = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
    url_example_extract_pattern = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=$<>{}]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=<>{}]*)"

    res = re.findall(url_example_extract_pattern, text) # returns ['https://uibakery.io']
    
    return res


# if __name__=="__main__":
    # print("==TEST URL EXTRACTION==")
    # cv.extract_url_from_str('You can view more details at https://uibakery.io or just ping via email.')

    # print("==TEST VALID URL WITH PROTOCOL=")
    # print(cv.validate_url('https://uibakery.io'))

    # print("==TEST INVALID URL WITH PROTOCOL=")
    # print(cv.validate_url('https:/uibakery.io'))

    # print("==TEST VALID URL WITHOUT PROTOCOL=")
    # print(cv.validate_url(url = 'uibakery.io', include_protocol = False))pwd
