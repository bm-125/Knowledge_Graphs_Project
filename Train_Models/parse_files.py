import numpy as np
import pandas as pd
import os


def parse_real(emb):
    try:
        return {v[0]: [float(n) for n in v[1].split(",")] for _, v in emb.iterrows()}
    except ValueError:
        return None
    

def parse_complex(emb):
    try:
        complex_embs = []
        for _, row in emb.iterrows():
            lst = row.to_numpy()[1].split(",")
            lst = [np.abs(complex(value)) for value in lst]
            complex_embs.append(lst)
        emb = pd.concat([emb[0], pd.Series(complex_embs)], axis=1)
        return {k: v for _, (k, v) in emb.iterrows()}
    except ValueError:
        return None


def format_embs(emb):
    formatted_emb = parse_real(emb)
    if formatted_emb is None:
        formatted_emb = parse_complex(emb)
        if formatted_emb is None:
            raise ValueError("Unsupported embedding format.")
    return formatted_emb


def get_emb_arrays(emb_folder):
    files = sorted([file for file in os.listdir(emb_folder)])
    raw_embs = {}
    for file in files:
        raw_embs[file[:-4]] = pd.read_csv(f"{emb_folder}/{file}", sep="\t", header=None)
    formatted_embs = []
    for id, emb in raw_embs.items():
        formatted_embs.append((id, format_embs(emb)))
    print("Embeddings loaded.")
    return formatted_embs


def get_pairs(pairs_folder):
    pairs = pd.read_csv(pairs_folder, sep="\t", header=None)
    pairs[0] = pairs[0].apply(lambda x: x.strip("http://www.uniprot.org/uniprot/"))
    pairs[1] = pairs[1].apply(lambda x: x.strip("http://www.uniprot.org/uniprot/"))
    pairs_array = pairs.to_numpy()
    print("\nEntity pairs loaded.")
    return pairs_array
