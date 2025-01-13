import logging
import shutil
from pathlib import Path
from subprocess import run

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyfaidx import Fasta

from src.utils.config import get_params


def cluster_sequences(input_fasta: str, output_dir: str, is_test: bool = False):
    """Cluster sequences using MMseqs2"""

    params = get_params("cluster")["test" if is_test else "train"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(output_dir) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    process_fasta(Path(input_fasta), params["max_seq_len"])

    cmd = [
        "mmseqs",
        "easy-cluster",
        Path(input_fasta).with_suffix(".filtered.fasta").as_posix(),
        out_dir.as_posix(),
        tmp.as_posix(),
        "-c",
        str(params["coverage"]),
        "--min-seq-id",
        str(params["min_seq_id"]),
        "--cov-mode",
        str(params["cov_mode"]),
        "--cluster-mode",
        str(params["cluster_mode"]),
        "--threads",
        str(params["threads"]),
    ]

    if params["reassign"]:
        cmd.append("--cluster-reassign")

    try:
        run(cmd, check=True)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def process_fasta(fasta_path: Path, max_len=1022):
    """Remove sequences longer than max_len and save their identifiers to a file."""
    filtered_fasta_path = fasta_path.with_suffix(".filtered.fasta")
    long_sequences_path = fasta_path.with_suffix(".long_sequences.txt")

    with filtered_fasta_path.open("w") as filtered_fasta, long_sequences_path.open("w") as long_sequences:
        for header, seq in Fasta(str(fasta_path)).items():
            if len(seq) > max_len:
                long_sequences.write(f"{header.split()[0]}\n")
            else:
                filtered_fasta.write(f">{header}\n{seq}\n")

    return filtered_fasta_path


def plot(params: dict):
    train_clustered = pd.read_csv(
        Path(params["out"]) / "train_cluster.tsv", sep="\t", names=["representative", "member"]
    )
    test_clustered = pd.read_csv(Path(params["out"]) / "test_cluster.tsv", sep="\t", names=["representative", "member"])

    num_train = train_clustered["member"].nunique()
    num_test = test_clustered["member"].nunique()
    num_train_clu = train_clustered["representative"].nunique()
    num_test_clu = test_clustered["representative"].nunique()

    logging.info(f"train set has {num_train} and {num_train_clu} sequences after redundancy reduction")
    logging.info(f"test set has {num_test} and {num_test_clu} sequences after redundancy reduction")

    sequence_count = pd.DataFrame(
        {
            "Dataset": [
                f"Train(min-seq-id=0.3 , cov=0.8,\nmax-seq-len={params['train']['max_seq_len']})",
                f"Test(min-seq-id=0.5, cov=0.9,\nmax-seq-len={params['test']['max_seq_len']})",
            ],
            "Original": [num_train, num_test],
            "Clustered": [num_train_clu, num_test_clu],
        }
    )

    plt.figure(figsize=(6, 6))
    ax = sns.barplot(
        x="Dataset",
        y="value",
        hue="variable",
        data=pd.melt(sequence_count, ["Dataset"]),
        width=0.35,
        palette="Set2",
        dodge=True,
    )

    # Add value labels on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=8)

    plt.title("Number of Sequences Before and After Clustering")
    plt.xlabel("Dataset")
    plt.ylabel("Number of Sequences")

    plt.tight_layout()
    plt.savefig(Path(get_params("plot")["out"]) / "redunancy_reduction_summary.png")
    plt.close()

    train_clusters = train_clustered.groupby("representative").size().reset_index(name="count")
    test_clusters = test_clustered.groupby("representative").size().reset_index(name="count")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(train_clusters["count"], bins=30, kde=True, log_scale=(True, False))
    plt.title("Number of Sequences per Cluster in Train Set")
    plt.xlabel("Number of Sequences")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.histplot(test_clusters["count"], bins=30, kde=True, log_scale=(True, False))
    plt.title("Number of Sequences per Cluster in Test Set")
    plt.xlabel("Number of Sequences")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(Path(get_params("plot")["out"]) / "clusters_distribution.png")
    plt.close()


def main():
    params = get_params("cluster")

    for is_test in [False, True]:
        dataset = Path(params["test" if is_test else "train"]["sequence"])
        out_dir = Path(params["out"]) / ("test" if is_test else "train")

        cluster_sequences(str(dataset), str(out_dir), is_test)

    plot(params)


if __name__ == "__main__":
    main()
