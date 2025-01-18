import logging
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm

from datasets import ClassLabel, Dataset, Features, Value
from src.utils.config import get_params


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
        self.metadata = pd.read_csv(metadata, sep="\t", usecols=["EntryID", "GO_terms", "positive_GO", "negative_GO"])
        self.sentence = pd.read_csv(go_sentence, sep="\t").set_index("term")["sentence"].to_dict()
        self.seed = seed
        self.test_size = test_size

    def concatenate_sentences(self, go_terms: list, delimiter: str, start: str = "", end: str = "") -> str:
        """
        Concatenate GO sentences for given GO terms.

        Args:
            go_terms (list): List of GO terms.
            delimiter (str): Delimiter to join the sentences.
            start (str, optional): Start of the concatenated sentence. Defaults to "".
            end (str, optional): End of the concatenated sentence. Defaults to "".

        Returns:
            str: Concatenated GO sentences.
        """
        sentences = [self.sentence[term] for term in go_terms]
        sentences = [s for s in sentences if s]
        return start + delimiter.join(sentences) + end

    @abstractmethod
    def create_dataset(self) -> Dataset:
        """
        Create HuggingFace dataset using Dataset.from_dict().

        Returns:
            Dataset: HuggingFace Dataset object.
        """
        raise NotImplementedError


class ProteinGOContrastiveDataset(HuggingFaceDatasetCreator):
    def __init__(self, sequence: Path, metadata: Path, go_sentence: Path, seed: int = 42, test_size: float = 0.1):
        """
        Custom Dataset for Protein-GO term pairs with protein ID and GO term ID for contrastive learning.

        Args:
            sequence (Path): Path to the protein sequences FASTA file.
            metadata (Path): Path to the metadata.tsv file.
            go_sentence (Path): Path to the go_sentences.tsv file.
            seed (int, optional): Random seed for shuffling the dataset. Defaults to 42.
            test_size (float, optional): Test set size. Defaults to 0.1.
        """

        super().__init__(sequence, metadata, go_sentence, seed, test_size)

    def _load_data(self) -> pd.DataFrame:
        logging.info("Exploding aggregated GO terms...")

        for col in ["positive_GO", "negative_GO"]:
            self.metadata[col] = self.metadata[col].str.split(",")

        exploded = pd.melt(
            self.metadata,
            id_vars=["EntryID"],
            value_vars=["positive_GO", "negative_GO"],
            var_name="go_source",
            value_name="go_term",
        ).explode("go_term")

        exploded["label"] = (exploded["go_source"] == "positive_GO").astype(str)

        exploded = (
            exploded.drop("go_source", axis=1)
            .reset_index(drop=True)
            .rename(columns={"EntryID": "protein", "go_term": "term"})
        )

        logging.info(f"Total datapoints after explosion: {len(exploded)}")

        return exploded

    def create_dataset(self):
        features = Features(
            {
                "protein": Value("string"),
                "term": Value("string"),
                "label": ClassLabel(num_classes=2, names=["False", "True"]),
            }
        )

        return (
            Dataset.from_pandas(self._load_data(), features=features)
            .shuffle(seed=self.seed)
            .train_test_split(test_size=self.test_size, stratify_by_column="label", seed=self.seed)
        )


class ProteinGOConcatenatedDataset(HuggingFaceDatasetCreator):
    def __init__(self, sequence: Path, metadata: Path, go_sentence: Path, seed: int = 42, test_size: float = 0.1):
        super().__init__(sequence, metadata, go_sentence, seed, test_size)

    def create_dataset(self) -> Dataset:
        """
        Create a protein-GO concatenated dataset by concatenating only `true` GO sentences for each protein.

        Returns:
            Dataset: HuggingFace Dataset object.
        """
        concatenated = [
            self.concatenate_sentences(go_terms, delimiter=",", end=".")
            for go_terms in tqdm(self.metadata["GO_terms"].str.split(",").to_list(), desc="Concatenating sentences")
        ]

        data = {
            "proteins": self.metadata["EntryID"].tolist(),
            "sequences": [str(self.sequence[prot]) for prot in self.metadata["EntryID"].tolist()],
            "sentences": concatenated,
        }

        features = Features({"proteins": Value("string"), "sequences": Value("string"), "sentences": Value("string")})

        return (
            Dataset.from_dict(data, features=features)
            .shuffle(seed=self.seed)
            .train_test_split(test_size=self.test_size, seed=self.seed)
        )


class ProteinGOContrastiveConcatenatedDataset(HuggingFaceDatasetCreator):
    def __init__(self, sequence: Path, metadata: Path, go_sentence: Path, seed: int = 42, test_size: float = 0.1):
        super().__init__(sequence, metadata, go_sentence, seed, test_size)
        self.rng = np.random.default_rng(seed)

    def sample_negative(self, positive_go_terms: list, go_terms: list) -> list:
        """
        Sample negative GO terms for a given protein.

        Args:
            positive_go_terms (list): List of positive GO terms.
            go_terms (list): list of all GO terms which already redundancy reduced to ensure their highly specificity.

        Returns:
            list: List of negative GO terms.
        """

        sampled = list(self.rng.choice(list(set(go_terms) - set(positive_go_terms)), size=10, replace=False))

        """
        # If we don't have enough terms, sample more until we get 10
        remaining = 10 - len(negative_go_terms)
        if remaining > 0:
            available_terms = go_terms - set(positive_go_terms) - negative_go_terms
            if available_terms:
                additional = set(
                    rng.choice(list(available_terms), size=min(remaining, len(available_terms)), replace=False)
                )
                negative_go_terms.update(additional)
        """

        return sampled

    def create_dataset(self) -> Dataset:
        """
        Create a protein-GO concatenated dataset by concatenating both `true` and `negative` GO sentences for each protein.

        Returns:
            Dataset: HuggingFace Dataset object.
        """

        go_terms_pool = self.metadata["positive_GO"].str.split(",").explode().unique().tolist()

        concatenated = []
        for go_terms in tqdm(
            self.metadata["GO_terms"].str.split(",").to_list(), desc="Concatenating contrasive sentences"
        ):
            positive = self.concatenate_sentences(go_terms, delimiter=",", start="Relevant to: ", end=";")
            negative = self.concatenate_sentences(
                self.sample_negative(go_terms, go_terms_pool),
                delimiter=",",
                start="Not relevant to: ",
                end=".",
            )
            concatenated.append(" ".join([positive, negative]))

        data = {
            "proteins": self.metadata["EntryID"].tolist(),
            "sequences": [str(self.sequence[prot]) for prot in self.metadata["EntryID"].tolist()],
            "sentences": concatenated,
        }

        features = Features({"proteins": Value("string"), "sequences": Value("string"), "sentences": Value("string")})

        return (
            Dataset.from_dict(data, features=features)
            .shuffle(seed=self.seed)
            .train_test_split(test_size=self.test_size, seed=self.seed)
        )


def main():
    params = get_params("dataset")

    try:
        dataset_class = globals()[params["type"]]
    except KeyError:
        raise ValueError("Unknown dataset class.")

    creator = dataset_class(
        Path(params["protein"]),
        Path(params["metadata"]),
        Path(params["go_sentence"]),
        seed=params["seed"],
        test_size=params["test_size"],
    )

    dataset = creator.create_dataset()
    dataset.save_to_disk(Path(params["out"]))


if __name__ == "__main__":
    main()
