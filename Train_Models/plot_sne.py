from sklearn import manifold
import seaborn as sns
import matplotlib.pyplot as plt
from train_models import vector_combination
from constants import TRAINING_FOLDER


def fit_sne(x, y, title, k=2):
    #Reducing pair representations to 2 dimensions using t-SNE reduction dimensionaluty
    dimension_reduction_technique = manifold.TSNE(n_components=k, random_state=42)
    X_transform = dimension_reduction_technique.fit_transform(x)
    #Plotting the t-SNE reduction in a 2D scatter plot (each point represents a protein pair and the color represents the label)
    X_transform_1dim = X_transform[:,0]
    X_transform_2dim = X_transform[:,1]
    sns.scatterplot(x=X_transform_1dim, y=X_transform_2dim, hue=y, palette=['#FF5050', '#70BB83'])
    plt.title(title)


def plot_sne(embs, pairs, k=2, vec_op='Hadamard', vec_size=200, return_data=False):
    assert vec_op != "CosSim", "Cosine similarity cannot be used to plot a tSNE as it turns the embedding vectors into a scalar."
    data_ml = {}
    plt.figure(figsize=(20, 5*(len(embs)//3+1)))
    plt.suptitle(f"Embedding projections with '{vec_op}' vector operations", y=0.95, fontsize='xx-large')
    for i, (id, emb) in enumerate(embs):
        plt.subplot(len(embs)//3+1, 3, i+1)
        x, y = vector_combination(emb, pairs, vector_operation=vec_op, vector_size=vec_size)
        if return_data:
            data_ml[(id, vec_op)] = (x, y)
        fit_sne(x, y, f'{id} Embedding Projections', k=k)
    plt.savefig(TRAINING_FOLDER + f'{vec_op}_tSNE.png')
    return data_ml
