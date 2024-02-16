import os
import re

import pandas as pd
import numpy as np
import json

import gensim
import spacy
from nltk.corpus import stopwords
import matplotlib
import matplotlib.pyplot as plt

from top2vec import Top2Vec

# Remove punctuation, numbers, and converts all to min-caps
def basic_cleanning(docs:list):
    final = []
    for doc in docs:
        new = gensim.utils.simple_preprocess(doc) # returns a list
        new = " ".join(new)
        final.append(new)
    return(final)

def remove_custom_stop_words(docs, stops): # Modify to include prepositions only
    new_docs = []
    for doc in docs:
        words = doc.split()
        cleanned_words = []
        for word in words:
            if word not in stops:
                cleanned_words.append(word)
        new_doc = " ".join(cleanned_words)
        new_docs.append(new_doc)
    return new_docs

def create_corpus(df):
    # create docs
    functionality = df.functionality.to_list()
    description = df.description.to_list()

    # From 'functionality' and 'description' use longer text
    docs = []
    for  func, desc in zip(functionality, description):
        tokens_func = len(func.split())
        tokens_desc = len(desc.split())
        if (tokens_func < tokens_desc):
            docs.append(desc)
        else:
            docs.append(func)
    
    # Clean docs
    original_docs = basic_cleanning(docs)

    # Remove stopwords
    # print(stopwords.words("english"))
    docs = remove_custom_stop_words(original_docs, stopwords.words("english"))

    print("===Cleanned docs===")
    print(f"# original docs: {len(original_docs)}")
    print(f"# clenned docs: {len(docs)}")

    return original_docs, docs

def create_model(docs:list, hyper_params:dict):
    model = Top2Vec(docs, speed= "deep-learn", 
                    min_count=hyper_params["min_count"], 
                    embedding_model = "doc2vec", 
                    ngram_vocab = False, 
                    hdbscan_args = hyper_params["custom_hdbscan_args"], 
                    umap_args = hyper_params["custom_umap_args"])
    return model

def get_all_info(model, original_docs:list, top_words:list, cos_sim_scores:list, topic_idxs:list, topic_sizes:list):
    topic_list = []
    for top_word_lst, cos_sim_score_lst, topic_idx in zip(top_words, cos_sim_scores, topic_idxs):
        data_point = {}
        # Add topic index
        data_point["topic"] = int(topic_idx)

        # Add relevant words
        topic_words = []
        for top_word, cos_sim_score in zip(top_word_lst,cos_sim_score_lst):
            words_info = {}
            words_info["word"] = top_word
            words_info["score"]=float(cos_sim_score)
            topic_words.append(words_info)
        data_point["topic_words"] = topic_words

        # Example document(s)
        documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=topic_idx, num_docs=topic_sizes[topic_idx])
        docs = []
        for doc, score, doc_id in zip(documents, document_scores, document_ids):
            doc_info = {}
            doc_info["doc_id"] = int(doc_id)
            doc_info["score"] =  float(score)
            doc_info["doc"] = doc
            doc_info["original_doc"] = original_docs[int(doc_id)]
            docs.append(doc_info)

        data_point["docs"] = docs
        topic_list.append(data_point)
    return topic_list
    
def create_matrix(model, rows:int, cols:int, top_words:list, cos_sim_scores:list, topic_idxs:list, topic_sizes:list):
    matrix_size = (rows,cols)
    matrix = np.zeros(matrix_size, dtype = float)

    for top_word_lst, cos_sim_score_lst, topic_idx in zip(top_words, cos_sim_scores, topic_idxs ):
        documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=topic_idx, num_docs=topic_sizes[topic_idx])
        for doc, score, doc_id in zip(documents, document_scores, document_ids):
            matrix[doc_id,topic_idx] = float(score)

    df_matrix = pd.DataFrame(matrix)
    return df_matrix