
DATA_FOLDER_PATH = '../Data/'
ORIGINAL_ENTITY_PAIRS_FILE = DATA_FOLDER_PATH + "entity_pairs.txt"
PROCESSED_ENTITY_PAIRS_OUTPUT = DATA_FOLDER_PATH + "edited_entity_pairs.tsv"
COSINE_PAIRS_INPUT_FILE = DATA_FOLDER_PATH + "cossim_pairs.tsv"
PROCESSED_ENTITY_FILE_OUTPUT = DATA_FOLDER_PATH + "processed_entities.csv"
NULL_ENTITY_PAIRS_OUTPUT = DATA_FOLDER_PATH + "null_entity_pairs.csv"
ENTITY_URL = 'http://www.uniprot.org/uniprot/'

ANNOTATIONS_FILE = DATA_FOLDER_PATH + "goa_human.gaf"
ANNOTATIONS_FILE_TYPE = 'gaf-2.0'
OWL_ONTOLOGY_FILE = DATA_FOLDER_PATH + "go-basic.owl"
OBO_ONTOLOGY_FILE = DATA_FOLDER_PATH + "go-basic.obo"
ONTOLOGY_TYPE = "GeneOntology"
ONTOLOGY_FILE_TYPE = "obo"

RESULTS_FILE_PATH = '../Results/'
EMBEDDINGS_OUTPUT_FOLDER = RESULTS_FILE_PATH + "Embeddings/"
SEMANTIC_SIMILARITIES_OUTPUT = RESULTS_FILE_PATH + "SSM/semantic_similarities.csv"
SEMANTIC_CORRELATION_HEATMAP = RESULTS_FILE_PATH + "SSM/semantic_correlation_heatmap.png"
RUNNING_TIMES_OUTPUT = RESULTS_FILE_PATH + "model_running_times.tsv"
PAIRWISE_T_TEST_OUTPUT = RESULTS_FILE_PATH + "Statistics/pairwise_t_test.csv"
KENDALLS_TAU_RANK_TEST_OUTPUT = RESULTS_FILE_PATH + "Statistics/kendalls_tau_rank_test.csv"
DESCRIPTIVE_SEM_SIM_STATS_OUTPUT = RESULTS_FILE_PATH + "Statistics/descriptive_sem_sim_stats.csv"

TRAINING_FOLDER = RESULTS_FILE_PATH + 'Training/'
PYKEEN_MODELS = ["convE", "distMult", "rotatE", "transD", "transE"]
PYKEEN_DATA_SPLIT = [0.8, 0.1, 0.1]
ML_MODEL_PARAMS_FILE = None

RDF2VEC_CONFIGS = {
        1: {
        'vector_size': 200,
        'n_walks': 500,
        'type_word2vec': 'skip-gram',
        'walk_depth': 8,
        'walker_type': 'random',
        'sampler_type': 'uniform',
        'output_file': EMBEDDINGS_OUTPUT_FOLDER + 'rdf2vec_id1.tsv'
    }, 
    2: {
        'vector_size': 200,
        'n_walks': 500,
        'type_word2vec': 'skip-gram',
        'walk_depth': 8,
        'walker_type': 'wl',
        'sampler_type': 'uniform',
        'output_file': EMBEDDINGS_OUTPUT_FOLDER + 'rdf2vec_id2.tsv'
    }, 
    3: {
        'vector_size': 200,
        'n_walks': 500,
        'type_word2vec': 'skip-gram',
        'walk_depth': 8,
        'walker_type': 'random',
        'sampler_type': 'predfreq',
        'output_file': EMBEDDINGS_OUTPUT_FOLDER + 'rdf2vec_id3.tsv'
    }, 
    4: {
        'vector_size': 200,
        'n_walks': 500,
        'type_word2vec': 'skip-gram',
        'walk_depth': 8,
        'walker_type': 'wl',
        'sampler_type': 'predfreq',
        'output_file': EMBEDDINGS_OUTPUT_FOLDER + 'rdf2vec_id4.tsv'
    },
    5: {
        'vector_size': 200,
        'n_walks': 500,
        'type_word2vec': 'skip-gram',
        'walk_depth': 8,
        'walker_type': 'random',
        'sampler_type': 'objfreq',
        'output_file': EMBEDDINGS_OUTPUT_FOLDER + 'rdf2vec_id5.tsv'
    }, 
    6: {
        'vector_size': 200,
        'n_walks': 500,
        'type_word2vec': 'skip-gram',
        'walk_depth': 8,
        'walker_type': 'wl',
        'sampler_type': 'objfreq',
        'output_file': EMBEDDINGS_OUTPUT_FOLDER + 'rdf2vec_id6.tsv'
    }, 
    7: {
        'vector_size': 200,
        'n_walks': 500,
        'type_word2vec': 'CBOW',
        'walk_depth': 8,
        'walker_type': 'wl',
        'sampler_type': 'uniform',
        'output_file': EMBEDDINGS_OUTPUT_FOLDER + 'rdf2vec_id7.tsv'
    }
}