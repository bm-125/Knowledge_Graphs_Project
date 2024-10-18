import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from parse_files import get_emb_arrays, get_pairs
from plot_sne import plot_sne
from train_models import run_embedding_training, run_semantic_training
from constants import PROCESSED_ENTITY_PAIRS_OUTPUT, EMBEDDINGS_OUTPUT_FOLDER, SEMANTIC_SIMILARITIES_OUTPUT


# Load pairs, embeddings, and semantic similarities
pairs = get_pairs(PROCESSED_ENTITY_PAIRS_OUTPUT)
all_embeddings = get_emb_arrays(EMBEDDINGS_OUTPUT_FOLDER)
ssims = pd.read_csv(SEMANTIC_SIMILARITIES_OUTPUT)


# Generate embedding projections with 2D tSNE plots
sne_k = 2
vec_size = 200
vec_ops = ['Hadamard', 'Average', 'Concatenation', 'WL1', 'WL2']
# --- #
data_ml = []
print("\nGenerating t-SNE plots...")
for op in tqdm(vec_ops):
    data = plot_sne(all_embeddings, 
                    pairs, 
                    k=sne_k, 
                    vec_op=op, 
                    vec_size=vec_size, 
                    return_data=True)
    if data is not None:
        data_ml.append(data)


# Train with embedding vectors using ML models
train_type = 'ml'
test_size = 0.3
shuffle = True
model = RandomForestClassifier(random_state=42)
trained_params = None
# --- #
run_embedding_training(data=data_ml, 
                       train_type=train_type, 
                       model=model, 
                       test_size=test_size, 
                       shuffle=shuffle,
                       trained_params=trained_params)

# Train with embedding vectors using threshold-based predictors
train_type = 'threshold'
threshold = None
threshold_search = np.arange(0, 1.05, 0.05)
# --- #
run_embedding_training(embs=all_embeddings, 
                       pairs=pairs, 
                       train_type=train_type, 
                       vec_op="CosSim", 
                       vec_size=vec_size, 
                       threshold=threshold, 
                       threshold_search=threshold_search)



# Train with semantic similarities using threshold-based predictors
train_type = 'threshold'
threshold = None
threshold_search = np.arange(0, 1.05, 0.05)
# --- #
run_semantic_training(ssims, 
                      train_type=train_type, 
                      threshold=threshold, 
                      threshold_search=threshold_search)

# Train with semantic similarities using ML models
train_type = 'ml'
test_size = 0.3
shuffle = True
model = RandomForestClassifier(random_state=42)
train_with_all = True
trained_params = None
# --- #
run_semantic_training(ssims, 
                      train_type=train_type,
                      train_with_all=train_with_all,
                      model=model, 
                      test_size=test_size, 
                      shuffle=shuffle,
                      trained_params=trained_params)


# --- #
print("\nBenchmark done!")
