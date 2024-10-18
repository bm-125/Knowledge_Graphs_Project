from constants import PROCESSED_ENTITY_PAIRS_OUTPUT, COSINE_PAIRS_INPUT_FILE, EMBEDDINGS_OUTPUT_FOLDER, SEMANTIC_SIMILARITIES_OUTPUT, \
                      PAIRWISE_T_TEST_OUTPUT, PROCESSED_ENTITY_PAIRS_OUTPUT, KENDALLS_TAU_RANK_TEST_OUTPUT, DESCRIPTIVE_SEM_SIM_STATS_OUTPUT
from plotting import plot_ss_distributions, plot_cosine_similarity_distributions
from statistical_analysis import process_pairs_entities_file, create_metric_dataframes, sort_metric_dataframes, descriptive_stats_ss_sims
from statistical_analysis import perform_kendall_tau_test, perform_pairwise_t_tests
from computing_cosine_similarities import load_embeddings_compute_cosine_similarities, load_embeddings_compute_all_cosine_similarities_pairs

print("\nGenerating statistics...\n")
process_pairs_entities_file(PROCESSED_ENTITY_PAIRS_OUTPUT, COSINE_PAIRS_INPUT_FILE)

# Peforming basic statistical analysis on embeddings methods
## Every protein entity against every protein entity
cosine_similarity_data = load_embeddings_compute_cosine_similarities(EMBEDDINGS_OUTPUT_FOLDER)
plot_cosine_similarity_distributions(cosine_similarity_data)
perform_pairwise_t_tests(cosine_similarity_data, PAIRWISE_T_TEST_OUTPUT)

## Protein pairs only
cosine_similarity_data_pairs = load_embeddings_compute_all_cosine_similarities_pairs(EMBEDDINGS_OUTPUT_FOLDER, COSINE_PAIRS_INPUT_FILE)
plot_cosine_similarity_distributions(cosine_similarity_data_pairs, pairs='Yes')
print("Statistical analysis performed for knowledge graph embedding models.")

# Peforming basic statistical analysis on traditional semantic similarity measures
ss_metric_dataframes = create_metric_dataframes(SEMANTIC_SIMILARITIES_OUTPUT)
plot_ss_distributions(ss_metric_dataframes)
descriptive_stats_ss_sims(ss_metric_dataframes, DESCRIPTIVE_SEM_SIM_STATS_OUTPUT)
ss_sorted_metric_dataframes = sort_metric_dataframes(ss_metric_dataframes)
perform_kendall_tau_test(ss_sorted_metric_dataframes, KENDALLS_TAU_RANK_TEST_OUTPUT)
print("\nStatistical analysis performed for traditional semantic similarity measures.")
