import shutil
shutil.copyfile("../constants.py", "./constants.py")
shutil.copyfile("../constants.py", "../Statistics/constants.py")
shutil.copyfile("../constants.py", "../Train_Models/constants.py")
import time
import pandas as pd
from run_sem_sim import main as run_sem_sim
from run_pykeen_embeddings import run_pykeen_model
from rdf_graph_generator import main as rdf_graph_generator
from run_rdf2vec import main as run_rdf2vec
from constants import PROCESSED_ENTITY_PAIRS_OUTPUT, PROCESSED_ENTITY_FILE_OUTPUT, \
                      ANNOTATIONS_FILE, OWL_ONTOLOGY_FILE, \
                      PYKEEN_MODELS, PYKEEN_DATA_SPLIT, RDF2VEC_CONFIGS, \
                      RUNNING_TIMES_OUTPUT


# Run semantic similarities
measures = ["Resnik", "Lin", "Jiang-Conrath", "SimGIC", "SimUI"]
mixings = ["avg", "max", "BMA"]
times = run_sem_sim(measures, mixings)


# Runs the models from PyKEEN
vec_size = 200
for model in PYKEEN_MODELS:
    t0 = time.time()
    run_pykeen_model(PROCESSED_ENTITY_PAIRS_OUTPUT, model, PYKEEN_DATA_SPLIT, vec_size)
    t1 = time.time()
    times.append({'Model': model, 'Running Time': round(t1-t0, 2)})


# Runs RDF2Vec
print("\nCreating embeddings using RDF2Vec models")
g, prots = rdf_graph_generator(OWL_ONTOLOGY_FILE, ANNOTATIONS_FILE, PROCESSED_ENTITY_FILE_OUTPUT)
for id, params in RDF2VEC_CONFIGS.items():
    model = f"RDF2Vec_{id}"
    start_time = time.time()
    run_rdf2vec(g, prots, **params)
    end_time = time.time()
    running_time = end_time - start_time
    times.append({'Model': model, 'Running Time': round(running_time, 2)})
print("\nFinished creating embeddings using RDF2Vec models!")

times_df = pd.DataFrame(times)
print("\n", times_df)
times_df.to_csv(RUNNING_TIMES_OUTPUT, sep='\t', index=False, header=True)

print("\nThe results have been saved to the output folder!")
