# Knowledge Graphs 2023-2024
# Group 1
# Ana Moreira (54514), Carolina Rodrigues (62910), Cláudia Afonso (36273), João Lobato (62611), Tiago Assis (62609)


import pandas as pd
from itertools import combinations
from scipy.stats import ttest_ind, t
from scipy import stats


def process_pairs_entities_file(input_file, output_file):
    with open(input_file, 'r') as in_file, open(output_file, 'w') as out_file:
        for line in in_file:
            if line.strip():
                proteins = line.split()
                if len(proteins) == 3:
                    accession1 = proteins[0].split('/')[-1]
                    accession2 = proteins[1].split('/')[-1]
                    interaction = proteins[2]
                    out_file.write(f"{accession1} {interaction} {accession2}\n")


def perform_pairwise_t_tests(cosine_similarity_data, output_file, significance_level=0.05):
    pairwise_tests = []
    
    for method1, method2 in combinations(cosine_similarity_data.keys(), 2):
        data1 = cosine_similarity_data[method1]
        data2 = cosine_similarity_data[method2]
        
        df = len(data1) + len(data2) - 2  # Degrees of freedom for pairwise t-test
        critical_value = t.ppf(1 - significance_level/2, df)
        
        t_statistic, p_value = ttest_ind(data1, data2)
        
        result = {
            'Method 1': method1,
            'Method 2': method2,
            't-statistic': t_statistic,
            'p-value': p_value,
            'Critical Value': critical_value,
            'Significantly Significant': 'Yes' if p_value < significance_level else 'No',
            'Statistically Different': 'Yes' if abs(t_statistic) > critical_value else 'No'
        }
        pairwise_tests.append(result)
    
    
    df_pairwise_tests = pd.DataFrame(pairwise_tests)
    df_pairwise_tests.to_csv(output_file, index=False)
    
    return df_pairwise_tests


def perform_kendall_tau_test(sorted_dataframes, output_file):

    metrics = list(sorted_dataframes.keys())
   
    results_df = pd.DataFrame(columns=['Similarity Metric 1', 'Similarity Metric 2', 'Correlation Coefficient', 'P-value', 'Statistically Significant', 'Correlation >= 0.5'])

    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            metric1 = metrics[i]
            metric2 = metrics[j]
            df_name1 = metric1.lower().replace(' ', '_').replace('(', '').replace(')', '')
            df_name2 = metric2.lower().replace(' ', '_').replace('(', '').replace(')', '')
            df1 = sorted_dataframes[df_name1]
            df2 = sorted_dataframes[df_name2]

            proteins1 = df1[['Ent1', 'Ent2']]
            proteins2 = df2[['Ent1', 'Ent2']]

            correlation, p_value = stats.kendalltau(proteins1.values.flatten(), proteins2.values.flatten())
            statistically_significant = 'Yes' if p_value < 0.05 else 'No'
            correlation_ge_0_5 = 'Yes' if abs(correlation) >= 0.5 else 'No'
            results_df = results_df.append({'Similarity Metric 1': metric1, 'Similarity Metric 2': metric2,
                                            'Correlation Coefficient': correlation, 'P-value': p_value,
                                            'Statistically Significant': statistically_significant,
                                            'Correlation >= 0.5': correlation_ge_0_5},
                                           ignore_index=True)

    results_df.to_csv(output_file, index=False)

    return results_df



# For traditional semantic similarity measures

def create_metric_dataframes(input_file):

    ss_df = pd.read_csv(input_file)
    metrics = ss_df.columns[3:].tolist()
    metric_dfs = {}
    
    for metric in metrics:
        metric_df = ss_df[['Ent1', 'Ent2', metric]].copy()
        
        df_name = metric.lower().replace(' ', '_').replace('(', '').replace(')', '')
        
        metric_dfs[df_name] = metric_df
    
    return metric_dfs


def sort_metric_dataframes(metric_dfs):
    
    sorted_dfs = {}

    for df_name, df in metric_dfs.items():
        metric = df.columns[-1]
        df_sorted = df.sort_values(by=metric, ascending=False)
        sorted_dfs[df_name] = df_sorted
    
    return sorted_dfs


def descriptive_stats_ss_sims(metric_dataframes, output_file):
    
    stats_list = []

    for df_name, df in metric_dataframes.items():
        metric_name = df_name.replace('df_', '')

        stats = df.describe().transpose()
        stats = stats.round(4)
        stats.insert(0, 'Metric', metric_name)

        stats_list.append(stats)

    stats_df = pd.concat(stats_list, ignore_index=True)

    stats_df.to_csv(output_file, index=False)
    