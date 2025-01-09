import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyfaidx import Fasta
from tqdm import tqdm

from src.utils.annotation import GOParser
from src.utils.config import get_params
from src.utils.taxonomy import TaxonomyMapper

warnings.filterwarnings("ignore", category=FutureWarning)


def gather_protein_annotations(
    train_folder: Path, cluster_folder: Path, seed: int, rep_only: bool = True
) -> pd.DataFrame:
    """
    Gather all GO terms and aspects for each protein entry.

    Args:
        train_folder: Path to the folder with train_terms.tsv, train_taxonomy.tsv, train_sequences.fasta and go-basic.obo
        cluster_folder: Path to the folder with MMseqs cluster results
        seed: Seed for reproducibility in sampling hard negative labels
        rep_only: only representive sequences of MMseqs

    Returns:
        pd.Dataframe: DataFrame with proteins and their grouped annotations
    """
    cafa = (
        pd.read_csv(
            train_folder / "train_terms.tsv",
            sep="\t",
            names=["EntryID", "term", "aspect"],
            skiprows=1,
        )
        .groupby("EntryID")
        .agg({"term": lambda x: set(x), "aspect": lambda x: set(x)})
    )

    if rep_only:
        seq_rep = pd.read_csv(cluster_folder / "train_cluster.tsv", sep="\t", header=None)[0]
        cafa = cafa[cafa.index.isin(seq_rep)]

    grouped = defaultdict(lambda: {"terms": set(), "aspects": set()})
    for protein, row in tqdm(cafa.iterrows(), total=len(cafa), desc="Grouping GO terms"):
        grouped[protein]["terms"] = row["term"]
        grouped[protein]["aspects"] = row["aspect"]

    result = pd.DataFrame(
        [
            {
                "EntryID": protein,
                "GO_terms": data["terms"],
                "aspects": ",".join(data["aspects"]),
                "num_terms": len(data["terms"]),
            }
            for protein, data in grouped.items()
        ]
    )

    result = result.merge(
        pd.read_csv(
            train_folder / "train_taxonomy.tsv",
            sep="\t",
            names=["EntryID", "taxonomyID"],
        ),
        on="EntryID",
    )

    mapper = TaxonomyMapper()
    go_parser = GOParser(train_folder / "go-basic.obo", seed=seed)
    sentences = pd.DataFrame.from_dict(go_parser.go_sentences, orient="index").reset_index(names=["term"])

    tqdm.pandas(desc="processing taxonomy")
    result["species"] = result["taxonomyID"].map(mapper.map_taxonomy_to_species(result["taxonomyID"].unique().tolist()))
    result["kingdom"] = result["taxonomyID"].progress_apply(mapper.get_broad_taxonomy)

    prot_len = match_protein_length(result["EntryID"].tolist(), train_folder / "train_sequences.fasta")
    result["length"] = result["EntryID"].map(prot_len)

    positives, negatives = go_parser.get_annotations(result["GO_terms"].tolist())

    result["positive_GO"] = positives
    result["negative_GO"] = negatives

    return (
        result[
            [
                "EntryID",
                "length",
                "kingdom",
                "species",
                "taxonomyID",
                "num_terms",
                "aspects",
                "positive_GO",
                "negative_GO",
            ]
        ],
        sentences,
    )


def match_protein_length(prot_id: List[str], fasta: Path) -> dict:
    """
    Match protein IDs to their respective lengths.

    Args:
        prot_id: List of protein IDs
        fasta: Path to FASTA file with protein sequences

    Returns:
        Dictionary mapping protein IDs to their lengths
    """
    fasta = Fasta(fasta)
    return {prot: len(fasta[prot]) for prot in prot_id}


def plot(metadata: pd.DataFrame, out: str = get_params("plot")["out"]):
    """
    Plot the distribution of GO terms by protein length and contrast
    the number of GO terms before and after redundancy reduction.

    Args:
        metadata: DataFrame with protein metadata
        out: Path to save the plots
    """

    plt.figure(figsize=(12, 6))
    max_length_plot = 3000

    plt.subplot(121)
    hexbin_data = metadata[metadata["length"] <= max_length_plot]
    plt.hexbin(hexbin_data["length"], hexbin_data["num_terms"], gridsize=100, cmap="viridis", mincnt=1)
    plt.colorbar(label="Number of Proteins")
    plt.xlabel("Protein Length (aa)")
    plt.ylabel("Number of GO Terms")
    plt.title("GO Terms by Protein Length")

    plt.subplot(122)

    max_length = metadata["length"].max()
    length_bins = np.linspace(0, min(max_length, 3000), 100)
    metadata["length_bin"] = pd.cut(metadata["length"], bins=length_bins)

    binned = metadata.groupby("length_bin")["num_terms"].mean().reset_index()

    plt.plot(
        binned["length_bin"].astype(str), binned["num_terms"], color="green", marker="o", linestyle="-", markersize=2
    )
    plt.xlabel("Protein Length Bins (aa)")
    plt.ylabel("Average Number of GO Terms")
    plt.title("Binned Average: GO Terms by Protein Length")
    plt.xticks([])
    plt.tight_layout()

    plt.savefig(Path(out) / "go_terms_vs_length.png")
    plt.close()

    metadata["num_terms_reduced"] = metadata["positive_GO"].apply(lambda x: len(x.split(",")))

    print(f"Mean number of GO terms before reduction: {metadata['num_terms'].mean()}")
    print(f"Median number of GO terms before reduction: {metadata['num_terms'].median()}")
    print(f"Mean number of GO terms after reduction: {metadata['num_terms_reduced'].mean()}")
    print(f"Median number of GO terms after reduction: {metadata['num_terms_reduced'].median()}")

    plot_data = pd.melt(metadata, value_vars=["num_terms", "num_terms_reduced"], var_name="Type", value_name="GO_Terms")
    plot_data["Type"] = plot_data["Type"].map({"num_terms": "Original", "num_terms_reduced": "Reduced"})

    # Density Plot
    plt.figure(figsize=(12, 6))
    sns.kdeplot(metadata["num_terms"], shade=True, color="blue", label="Original")
    sns.kdeplot(metadata["num_terms_reduced"], shade=True, color="orange", label="Reduced")
    plt.xlabel("Number of GO Terms")
    plt.ylabel("Density")
    plt.title("GO Terms Before and After Redundancy Reduction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out) / "terms_density_comparison.png")
    plt.close()

    # Violin Plot
    plt.figure(figsize=(6, 6))
    sns.violinplot(x="Type", y="GO_Terms", data=plot_data, palette=["#1f77b4", "#ff7f0e"])
    plt.xlabel("GO Terms")
    plt.ylabel("Number of GO Terms")
    plt.title("GO Terms Before and After Redundancy Reduction")
    plt.tight_layout()
    plt.savefig(Path(out) / "terms_violin_comparison.png")
    plt.close()


def main():
    params = get_params("metadata")

    metadata, sentences = gather_protein_annotations(
        Path(params["data_folder"]), Path(params["cluster_folder"]), seed=params.get("seed", 42)
    )
    metadata.to_csv(Path(params["out"]) / "train_metadata.tsv", sep="\t", index=False)

    sentences.to_csv(Path(params["out"]) / "go_sentences.tsv", sep="\t", index=False)

    logging.info(f"All protein metadata saved to {params['out']}")

    plot(metadata)

    logging.info("Plots saved successfully.")


if __name__ == "__main__":
    main()
