# Knowledge Graphs 2023-2024
# Group 1
# Ana Moreira (54514), Carolina Rodrigues (62910), Cláudia Afonso (36273), João Lobato (62611), Tiago Assis (62609)


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Functions to compute cosine similarities from embeddings for all entities against all entities

def read_embedding_file(file_path, file_name, pairwise_comparison='No'):
    df = pd.read_csv(file_path + file_name, sep='\t|,', engine='python', header=None)
    if pairwise_comparison == 'No':
        df.drop(df.columns[0], axis=1, inplace=True)
    return df
    

def process_complex_embeddings(complex_embeddings):
    
    embeddings_split_stacked = []

    for i, row in complex_embeddings.iterrows():
        lst = row.to_numpy()
        lst = [complex(value) for value in lst]
        lst = [x for value in lst for x in np.hstack((np.real(value), np.imag(value)))]
        embeddings_split_stacked.append(lst)

    return embeddings_split_stacked


def calculate_cosine_similarity(embeddings):

    sims = cosine_similarity(embeddings, embeddings)

    sims_values = []
    for row in range(len(sims)):
        for column in range(len(sims)):
            if row != column:
                sims_values.append(sims[row, column])

    return sims_values


def create_cosine_similarity_data(embeddings):
    
    cosine_similarity_data = {}
    for method, emb in embeddings.items():
        if method.lower() in ['rotate', 'complex']:
            sim = calculate_cosine_similarity(process_complex_embeddings(emb))
        else:
            sim = calculate_cosine_similarity(emb.to_numpy())
        
        cosine_similarity_data[method] = sim

    return cosine_similarity_data

    
def load_embeddings_compute_cosine_similarities(embeddings_file_path):

    embeddings = {}

    for filename in os.listdir(embeddings_file_path):
        method = filename[:-4]
        embeddings[method] = read_embedding_file(embeddings_file_path, filename)
    cosine_similarity_data = create_cosine_similarity_data(embeddings)
    
    return cosine_similarity_data
    
#####################################################################################

# Functions to compute cosine similarities from embeddings only for the protein pairs

def get_pair_embeddings(pairs_df, embedding_method, embeddings_dict):

    embeddings_df = embeddings_dict[embedding_method]
    x, y = [], []

    for _, row in pairs_df.iterrows():
        protein1_id = row[0]
        protein2_id = row[2]
        
        emb_prot1 = np.array(embeddings_df[embeddings_df.iloc[:, 0] == protein1_id].iloc[:, 1:].squeeze().values)
        emb_prot2 = np.array(embeddings_df[embeddings_df.iloc[:, 0] == protein2_id].iloc[:, 1:].squeeze().values)
        
        if embedding_method.lower() in ['rotate', 'complex']:
            emb_prot1 = np.array([complex(val) for val in emb_prot1])
            emb_prot2 = np.array([complex(val) for val in emb_prot2])

        x.append(emb_prot1)
        y.append(emb_prot2)

    return x, y


def process_complex_embeddings_pairs(complex_embeddings_list):
    embeddings_split_stacked = []

    for complex_array in complex_embeddings_list:
        lst = [x for value in complex_array for x in np.hstack((np.real(value), np.imag(value)))]
        embeddings_split_stacked.append(lst)

    return embeddings_split_stacked


def compute_cosine_similarity_pairs(list1, list2):
    cosine_similarities = []

    for elem1, elem2 in zip(list1, list2):
        elem1 = np.array(elem1).reshape(1, -1)
        elem2 = np.array(elem2).reshape(1, -1)
        cosine_sim = cosine_similarity(elem1, elem2)[0][0]
        cosine_similarities.append(cosine_sim)

    return cosine_similarities


def compute_all_cosine_similarities_pairs(pairs_df, embeddings_dict):

    cosine_similarities_dict = {}

    for method, _ in embeddings_dict.items():
        prot1, prot2 = get_pair_embeddings(pairs_df, method, embeddings_dict)

        if method.lower() in ['rotate', 'complex']:
            prot1_proc = process_complex_embeddings_pairs(prot1)
            prot2_proc = process_complex_embeddings_pairs(prot2)
            cosine_similarities_dict[method] = compute_cosine_similarity_pairs(prot1_proc, prot2_proc)

        else:
            cosine_similarities_dict[method] = compute_cosine_similarity_pairs(prot1, prot2)

    return cosine_similarities_dict


def load_embeddings_compute_all_cosine_similarities_pairs(embeddings_file_path, pairs_file):

    pairs_df = pd.read_csv(pairs_file, sep=' ', header=None)
    
    embeddings = {}

    for filename in os.listdir(embeddings_file_path):
        method = filename[:-4]
        embeddings[method] = read_embedding_file(embeddings_file_path, filename, pairwise_comparison='Yes')
    cosine_similarity_data_pairs = compute_all_cosine_similarities_pairs(pairs_df, embeddings)
    return cosine_similarity_data_pairs
