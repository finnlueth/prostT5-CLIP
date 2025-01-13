import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta

from datasets import Dataset, Features, Value
from src.utils.config import get_params


class H5Reader:
    """Wrapper for h5py to provide dict-like access without loading all data to memory"""

    def __init__(self, file_path: str, group: Optional[str] = None):
        self.file = h5py.File(file_path, "r")
        self.dataset = self.file[group] if group else self.file

    def __getitem__(self, key):
        return torch.from_numpy(self.dataset[key][()].astype(np.float32))

    def __del__(self):
        self.file.close()


class ProteinGODataset(Dataset):
    def __init__(
        self,
        prot_emb_file: str,
        text_emb_file: str,
        group: str = "train_set",
        table: str = None,
    ):
        """
        Custom Dataset for Protein-GO term pairs with pre-computed embeddings.

        Args:
            prot_emb_file (str): Path to the protein embeddings H5 file.
            text_emb_file (str): Path to the GO term embeddings H5 file.
            group (str, optional): Group key in the H5 protein embeddings file. Defaults to "train_set".
            table (str, optional): Path to the TSV file containing protein-GO term pairs. Defaults to None.
        """
        super().__init__()
        self.table = table
        self.data = self._load_data()
        self.prot_emb = H5Reader(prot_emb_file, group)
        self.text_emb = H5Reader(text_emb_file)

    def _load_data(self) -> pd.DataFrame:
        logging.info("Exploding aggregated GO terms...")

        metadata = pd.read_csv(Path(self.table), sep="\t", usecols=["EntryID", "positive_GO", "negative_GO"])

        for col in ["positive_GO", "negative_GO"]:
            metadata[col] = metadata[col].str.split(",")

        exploded = pd.melt(
            metadata,
            id_vars=["EntryID"],
            value_vars=["positive_GO", "negative_GO"],
            var_name="go_source",
            value_name="go_term",
        ).explode("go_term")

        exploded["label"] = (exploded["go_source"] == "positive_GO").astype(int)

        exploded = (
            exploded.drop("go_source", axis=1)
            .reset_index(drop=True)
            .rename(columns={"EntryID": "prot", "go_term": "text"})
        )

        logging.info(f"Total datapoints after explosion: {len(exploded)}")

        return exploded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prot_emb = self.prot_emb[row["prot"]]
        text_emb = self.text_emb[row["text"]]
        label = torch.tensor(row["label"], dtype=torch.long)

        if not torch.isfinite(prot_emb).all() or not torch.isfinite(text_emb).all():
            logging.warning(f"NaN or Inf detected in datapoint index {idx}. Skipping this datapoint.")
            return None

        return {
            "prot_emb": prot_emb,
            "text_emb": text_emb,
            "label": label,
        }


class HuggingFaceDatasetCreator:
    def __init__(self, sequence: Path, metadata: Path, go_sentence: Path, seed: int = 42, test_size: float = 0.1):
        """
        Initialize the dataset creator.

        Args:
            sequence (Path): Path to the protein sequences FASTA file.
            metadata (Path): Path to the metadata.tsv file.
            go_sentence (Path): Path to the go_sentences.tsv file.
            seed (int, optional): Random seed for shuffling the dataset. Defaults to 42.
            test_size (float, optional): Test set size. Defaults to 0.1.
        """
        self.sequence = Fasta(sequence, as_raw=True)
        self.metadata = pd.read_csv(metadata, sep="\t", usecols=["EntryID", "GO_terms"])
        self.sentence = pd.read_csv(go_sentence, sep="\t").set_index("term")["sentence"].to_dict()
        self.seed = seed
        self.test_size = test_size

    def concatenate_sentences(self, go_terms: str) -> str:
        """
        Concatenate GO sentences for given GO terms.

        Args:
            go_terms (str): Comma-separated GO terms.

        Returns:
            str: Concatenated GO sentences.
        """
        terms = go_terms.split(",")
        sentences = [self.sentence[term] for term in terms]
        sentences = [s for s in sentences if s]
        return ", ".join(sentences) + "."

    def create_dataset(self) -> Dataset:
        """
        Create HuggingFace dataset using Dataset.from_dict().

        Returns:
            Dataset: HuggingFace Dataset object.
        """
        concatenated = [self.concatenate_sentences(go_terms) for go_terms in self.metadata["GO_terms"].tolist()]
        sequences = [str(self.sequence[prot]) for prot in self.metadata["EntryID"].tolist()]

        data = {
            "proteins": self.metadata["EntryID"].tolist(),
            "sequences": sequences,
            "sentences": concatenated,
        }

        features = Features({"proteins": Value("string"), "sequences": Value("string"), "sentences": Value("string")})

        return (
            Dataset.from_dict(data, features=features)
            .shuffle(seed=self.seed)
            .train_test_split(test_size=self.test_size)
        )


def main():
    params = get_params("dataset")

    creator = HuggingFaceDatasetCreator(
        Path(params["protein"]),
        Path(params["metadata"]),
        Path(params["go_sentence"]),
        seed=params["seed"],
        test_size=params["test_size"],
    )

    dataset = creator.create_dataset()
    dataset.save_to_disk(params["output"])


if __name__ == "__main__":
    main()
