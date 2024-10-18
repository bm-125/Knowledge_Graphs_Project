import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from constants import RESULTS_FILE_PATH


def plot_cosine_similarity_distributions(cosine_similarity_data, pairs="No"):
    # Plot 1: Cumulative Distribution
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for method, sim_values in cosine_similarity_data.items():
        ax1.plot(np.sort(sim_values), np.linspace(0, 1, len(sim_values)), label=method)
    ax1.set_xlabel('Cosine Similarity Values')
    ax1.set_ylabel('Cumulative Distribution')
    if pairs == "Yes":
        ax1.set_title('Cumulative Distribution of Cosine Similarities from Embedding Methods for Protein Pairs')
        ax1.legend()
        plt.savefig(RESULTS_FILE_PATH + "Statistics/Cosine_Similarity_Cumulative_Distributions_Pairs.jpeg", dpi=300, format='jpeg')
        plt.close(fig1)
    else:
        ax1.set_title('Cumulative Distribution of Cosine Similarities from Embedding Methods')
        ax1.legend()
        plt.savefig(RESULTS_FILE_PATH + "Statistics/Cosine_Similarity_Cumulative_Distributions.jpeg", dpi=300, format='jpeg')
        plt.close(fig1)


    # Plot 2: Probability Density
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for method, sim_values in cosine_similarity_data.items():
        kde = gaussian_kde(sim_values)
        x = np.linspace(min(sim_values), max(sim_values), 200)
        ax2.plot(x, kde(x), label=method)
    ax2.set_xlabel('Cosine Similarity Values')
    ax2.set_ylabel('Probability Density')
    if pairs == "Yes":
        ax2.set_title('Probability Density of Cosine Similarities from Embedding Methods for Protein Pairs')
        ax2.legend()
        plt.savefig(RESULTS_FILE_PATH + "Statistics/Cosine_Similarity_Distributions_Pairs.jpeg", dpi=300, format='jpeg')
        plt.close(fig2)
    else:
        ax2.set_title('Probability Density of Cosine Similarities from Embedding Methods')
        ax2.legend()
        plt.savefig(RESULTS_FILE_PATH + "Statistics/Cosine_Similarity_Distributions.jpeg", dpi=300, format='jpeg')
        plt.close(fig2)

    # Plot 3: Normalization of cosine similarities
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for method, sim_values in cosine_similarity_data.items():
        kde = gaussian_kde(sim_values)
        x_values = np.linspace(min(sim_values), max(sim_values), 200)
        density = kde(x_values)
        density /= density.max()
        ax3.plot(x_values, density, label=method)
    ax3.set_xlabel('Cosine Similarity Values')
    ax3.set_ylabel('Normalized Probability Density')
    if pairs == "Yes":
        ax3.set_title('Normalized Probability Density of Cosine Similarities from Embedding Methods for Protein Pairs')
        ax3.legend()
        plt.savefig(RESULTS_FILE_PATH + "Statistics/Normalized_Cosine_Similarities_Pairs.jpeg", dpi=300, format='jpeg')
        plt.close(fig3)
    else:
        ax3.set_title('Normalized Probability Density of Cosine Similarities from Embedding Methods')
        ax3.legend()
        plt.savefig(RESULTS_FILE_PATH + "Statistics/Normalized_Cosine_Similarities.jpeg", dpi=300, format='jpeg')
        plt.close(fig3)


def plot_ss_distributions(ss_metric_dataframes):

    # Plot 1: Cumulative Distribution
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    for method_name, df in ss_metric_dataframes.items():
        values = df.iloc[:, 2]
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        sorted_values = np.sort(values)
        cumulative_distribution = np.linspace(0, 1, len(sorted_values))
        ax1.plot(sorted_values, cumulative_distribution, label=method_name)

    ax1.set_xlabel('Traditional Semantic Similarity Values')
    ax1.set_ylabel('Cumulative Distribution')
    ax1.set_title('Cumulative Distribution of Traditional Semantic Similarity Measures')
    ax1.legend()
    plt.savefig(RESULTS_FILE_PATH + "Statistics/Cumulative_Distribution_SS.jpg", dpi=300, format='jpg')
    plt.close(fig1)

    # Plot 2: Probability Density
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for method_name, df in ss_metric_dataframes.items():
        values = df.iloc[:, 2]
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        kde = gaussian_kde(values)
        x = np.linspace(min(values), max(values), 200)
        ax2.plot(x, kde(x), label=method_name)

    ax2.set_xlabel('Traditional Semantic Similarity Values')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Probability Density of Traditional Semantic Similarity Measures')
    ax2.legend()
    plt.savefig(RESULTS_FILE_PATH + "Statistics/Probability_Density_SS.jpg", dpi=300, format='jpg')
    plt.close(fig2)

    # Plot 3: Normalized Probability Density
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    for method_name, df in ss_metric_dataframes.items():
        values = df.iloc[:, 2]  
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        kde = gaussian_kde(values)
        x_values = np.linspace(min(values), max(values), 200)
        density = kde(x_values)
        density /= density.max()
        ax3.plot(x_values, density, label=method_name)

    ax3.set_xlabel('Traditional Semantic Similarity Values')
    ax3.set_ylabel('Normalized Probability Density')
    ax3.set_title('Normalized Probability Density of Traditional Semantic Similarity Measures')
    ax3.legend()
    plt.savefig(RESULTS_FILE_PATH + "Statistics/Normalized_Probability_Density_SS.jpg", dpi=300, format='jpg')
    plt.close(fig3)
    