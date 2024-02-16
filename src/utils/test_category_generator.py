import os
import pandas as pd

from tools.category_generator import generate_categories

# Info Loading
TEMP_DIR = "./data/temp"
OUTPUT_DIR = "./data/output"
INSTRUCTIONS_OUTPUT_FILE = "GDPS_REST_API_V4R6GM-4.6.0_instructions.json"

df = pd.read_json(os.path.join(OUTPUT_DIR,INSTRUCTIONS_OUTPUT_FILE))

hyper_params = {
    "custom_umap_args": {
        'n_neighbors': 2, # original; 15 -> The larger the values put more emphasis on global over local structure
        'n_components': 5,
        'metric': 'cosine'},
    "custom_hdbscan_args": {
        'min_cluster_size': 2, # original: 15 -> larger values have a higher chance of merging unrelated document clusters.
        'metric': 'euclidean',
        'cluster_selection_method': 'eom'},
    "min_count": 10 # CHECK AGAIN THIS ONE!
}

generate_categories(df,hyper_params, TEMP_DIR)




