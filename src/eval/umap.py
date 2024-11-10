import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from tqdm import tqdm
from typing import List, Tuple, Union
from sklearn.preprocessing import StandardScaler


def load_embeddings(inputs: str, proteins: Union[List, pd.Series]) -> Tuple:
    """
    Load embeddings for selected proteins from h5 file

    Args:
        h5_file: Path to h5 file containing embeddings
        proteins: List or Series of protein IDs to load

    Returns:
        embeddings: numpy array of embeddings
    """
    embeddings = []
    proteins = set(proteins)

    with h5py.File(inputs, "r") as f:
        for protein in tqdm(proteins):
            try:
                embeddings.append(f[protein][()])
            except KeyError:
                print(f"Warning: Protein {protein} not found in embeddings file")

    return np.array(embeddings)


def get_umap(embeddings, n_neighbors=30, min_dist=0.1, metric="cosine"):

    if metric == "euclidean":  # only for euclidean
        embeddings = StandardScaler().fit_transform(embeddings)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42
    )

    umap = reducer.fit_transform(embeddings)

    return umap


def visualize(umap, labels, save_path=None):
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x=umap[:, 0], y=umap[:, 1], hue=labels, palette="tab20", s=10, linewidth=0
    )
    plt.axis("off")
    plt.gca().set_aspect("equal", "datalim")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=3)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
