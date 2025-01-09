import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import h5py
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils.config import get_params


def load_paired_embeddings(
    inputs: Tuple[str, str],
    metadata: pd.DataFrame,
    group="train_set",
) -> Generator[Tuple[np.ndarray, List[np.ndarray]], None, None]:
    """
    Load paired embeddings from protein embedding file with specified group and from text embedding file.

    Args:
        inputs: Tuple of paths to protein and text embeddings
        proteins: List of protein IDs
        metadata: DataFrame with protein metadata
        group: Group name in protein embedding file e.g. train_set, test_set
    Yields:
        Tuple: Paired protein and text embeddings
    """

    with h5py.File(inputs[0], "r") as p, h5py.File(inputs[1], "r") as t:
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Loading embeddings"):
            protein_id = row["EntryID"]
            go_terms = row["positive_GO"].split(",")

            try:
                prot_emb = p[group][protein_id][()]
                agg_text_emb = [t[term][()] for term in go_terms]
                yield prot_emb, agg_text_emb
            except KeyError as e:
                logging.warning(f"can not find {e} in embeddings")


def get_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> np.ndarray:
    """Generate UMAP embeddings"""

    logging.info(f"fitting UMAP with {metric}")

    if metric == "euclidean":
        embeddings = StandardScaler().fit_transform(embeddings)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)

    return reducer.fit_transform(embeddings)


def plot_umap(
    umap_embedding: np.ndarray,
    labels: List,
    legend: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Visualize UMAP embedding with labels
    """
    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(
        x=umap_embedding[:, 0],
        y=umap_embedding[:, 1],
        hue=labels,
        palette="tab20",
        s=10,
        alpha=0.7,
        linewidth=0,
    )

    if title:
        plt.title(title)
    plt.axis("off")
    plt.gca().set_aspect("equal", "datalim")

    scatter.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        markerscale=3,
        title=legend,
        fontsize=8,
        title_fontsize=10,
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    logging.info(f"Saved UMAP plot to {save_path}")


def plot_heatmap(seq_sims: np.ndarray, text_sims: np.ndarray, save_path: str) -> float:
    """
    Create a 2D histogram (heatmap) of sequence and text embedding similarities with a frequency colorbar.

    Args:
        seq_sims (np.ndarray): A 2D numpy array of sequence embedding similarities.
        text_sims (np.ndarray): A 2D numpy array of text embedding similarities.
        save_path (str): The file path where the heatmap image will be saved.

    Returns:
        float:Pearson correlation coefficient between the two similarity
    """
    # Get upper triangle values
    seq_flat = seq_sims[np.triu_indices_from(seq_sims, k=1)]
    text_flat = text_sims[np.triu_indices_from(text_sims, k=1)]

    corr = np.corrcoef(seq_flat, text_flat)[0, 1]

    plt.figure(figsize=(10, 8))

    hist = plt.hist2d(
        seq_flat,
        text_flat,
        bins=50,
        cmap="hot_r",
        norm=matplotlib.colors.LogNorm(),
        density=True,
    )

    plt.colorbar(hist[3], label="Frequency (log scale)")

    plt.xlabel("Sequence Embedding Cosine Similarity")
    plt.ylabel("Text Embedding Cosine Similarity")
    plt.title(f"Embedding Similarity\n(Pearson r={corr:.3f})")

    z = np.polyfit(seq_flat, text_flat, 1)
    p = np.poly1d(z)
    plt.plot(seq_flat, p(seq_flat), "w--", alpha=0.8, label="Trend")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(Path(save_path) / "similarity_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved heatmap to {save_path}")

    return corr


def main():
    params = get_params("embed_vis")

    sampled = pd.read_csv(
        params["metadata"],
        sep="\t",
    ).sample(n=params["subsample"], random_state=params["seed"])

    paired_embeddings = load_paired_embeddings(
        (params["sequence_embeddings"], params["text_embeddings"]),
        sampled,
        group="train_set",
    )

    seq_embs, text_embs = [], []
    for prot_emb, text_embs_list in paired_embeddings:
        seq_embs.extend([prot_emb] * len(text_embs_list))
        text_embs.extend(text_embs_list)

    seq_embs = np.array(seq_embs)
    text_embs = np.array(text_embs)

    plot_heatmap(
        cosine_similarity(seq_embs),
        cosine_similarity(text_embs),
        get_params("plot")["out"],
    )

    seq_embs, text_embs = [], []
    for prot_emb, text_embs_list in paired_embeddings:
        seq_embs.append(prot_emb)
        text_embs.extend(text_embs_list)

    labels = sampled["kingdom"].tolist()

    print(f"seq_embs: {len(seq_embs)}")
    print(f"text_embs: {len(text_embs)}")

    umap_embedding = get_umap(
        np.array(seq_embs),
        n_neighbors=params["n_neighbors"],
        min_dist=params["min_dist"],
        metric=params["metric"],
    )

    plot_umap(
        umap_embedding,
        labels=labels,
        legend="Kingdom",
        save_path=Path(get_params("plot")["out"]) / f"umap_prot_{params['metric']}_{params['subsample']}.png",
        title=f"UMAP Projection of Protein Embeddings (n={len(labels)})",
    )

    go_map = (
        pd.read_csv(Path(params["GO"]), sep="\t", names=["term", "namespace"])
        .set_index("positive_GO")["aspect"]
        .to_dict()
    )
    labels = [go_map.get(go, "unknown") for go in sampled["positive_GO"].str.split(",").explode().unique()]

    umap_embedding = get_umap(
        np.array(text_embs),
        n_neighbors=params["n_neighbors"],
        min_dist=params["min_dist"],
        metric=params["metric"],
    )

    plot_umap(
        umap_embedding,
        labels=labels,
        legend="Aspect",
        save_path=Path(get_params("plot")["out"]) / f"umap_text_{params['metric']}_{params['subsample']}.png",
        title=f"UMAP Projection of Text Embeddings (n={len(labels)})",
    )


if __name__ == "__main__":
    main()
