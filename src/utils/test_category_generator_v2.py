import os
import argparse

import pandas as pd
import json

from top2vec import Top2Vec

# from tools.category_generator import generate_categories
from tools.category_generator import create_corpus, create_model, get_all_info, create_matrix

TEMP_DIR = "./data/temporal_files"
MODEL_NAME = "model_v1"
MATRIX = "matrix.csv"
EXAMPLES = "examples.csv"
TOPICS_JSON = "topics.json"

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--temporal_dir",
                    type=str,
                    required=False,
                    default="./data/temporal_files", 
                    help="Directory to load and/or save all temporal files.") 
    parser.add_argument("--instructions_temp_file",
                    type=str,
                    required=True,
                    help="The name of temporal file containing the API info and instructions in json format. It cannot be empty.")
    # TO DO: add non required args for other temp files (matrix.csv,examples.csv,topics.json,model)
    return parser.parse_args()

def generate_categories(df, 
                        hyper_params:dict, 
                        temp_dir:str = TEMP_DIR):
    # Create corpus
    original_docs, docs= create_corpus(df)

    # Create/Load model
    if os.path.isfile(os.path.join(TEMP_DIR,MODEL_NAME)):
        # Load model saved as file
        model = Top2Vec.load(os.path.join(TEMP_DIR,MODEL_NAME))
    else:
        # Create and save model
        model = create_model(docs, hyper_params)
        model.save(os.path.join(TEMP_DIR,MODEL_NAME))
    
    # Number of topics found
    total_num_topics = model.get_num_topics()
    # print(total_num_topics)

    # Number of documents most similar to each topic. 
    topic_sizes, topic_idxs = model.get_topic_sizes() # Big size: topics with more documents, Small size: topics with less documents 
    top_words, cos_sim_scores, topic_idxs = model.get_topics(total_num_topics)

    # Generate json with all words and documents per topic
    topic_list = get_all_info(model, 
                              original_docs=original_docs,
                              top_words=top_words, 
                              cos_sim_scores=cos_sim_scores, 
                              topic_idxs=topic_idxs, 
                              topic_sizes=topic_sizes)

    topic_list_json = json.dumps(topic_list, indent=4)

    with open(os.path.join(TEMP_DIR,TOPICS_JSON), "w") as outfile:
        outfile.write(topic_list_json)

    # Create a matrix (docs x topics)
    df_matrix = create_matrix(model, 
                              rows=len(docs),
                              cols=total_num_topics,
                              top_words=top_words, 
                              cos_sim_scores=cos_sim_scores, 
                              topic_idxs=topic_idxs, 
                              topic_sizes=topic_sizes)

    # Print matrix to csv file
    df_matrix.to_csv(os.path.join(TEMP_DIR,MATRIX), index=True)

    # Create df with most representative doc per topic
    df_max = pd.DataFrame({'topic': pd.Series(dtype='int'),
                            'doc_id': pd.Series(dtype='int'),
                            'doc_score': pd.Series(dtype='float'),
                            'doc': pd.Series(dtype='str'),
                            'topic_key_words': pd.Series(dtype='str')})
    for topic in df_matrix:
        doc_id = df_matrix[topic].idxmax()
        score = df_matrix[topic].max()
        doc = original_docs[doc_id]

        # dict_top_words = dict(zip(top_words[topic],cos_sim_scores[topic]))
        # dict_top_words_sorted = dict(sorted(dict_top_words.items(), key=lambda x:x[1], reverse=True))

        # pos_top_words = []
        # for word in dict_top_words_sorted:
        #     word_score = dict_top_words_sorted[word]
        #     if(word_score>0):
        #         pos_top_words.append(word)

        # top_words list is sorted by score
        pos_top_words = []
        for topic_word, topic_score in zip(top_words[topic],cos_sim_scores[topic]):
            if(topic_score>0):
                pos_top_words.append(topic_word)

        key_words = " ".join(pos_top_words)
        list_row = [topic, doc_id, score, doc, key_words]
        df_max.loc[len(df_max)] = list_row  
    # print(df_max.dtypes)

    # print("===Most representative doc per topic===")
    # print(df_max)
    print("===Check duplicates===")
    print(df_max.duplicated(keep=False))

    # Print max to csv file
    df_max.to_csv(os.path.join(TEMP_DIR,EXAMPLES), index=False)

if __name__=="__main__":

    args = parse_arguments()

    # TEMPORAL FILE
    TEMPORAL_DIR = args.temporal_dir
    INSTRUCTIONS_FILE = args.instructions_temp_file

    df = pd.read_json(os.path.join(TEMPORAL_DIR,INSTRUCTIONS_FILE))

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

    generate_categories(df,hyper_params, TEMPORAL_DIR)

    # TO DO: Move generate_categories method to this script!