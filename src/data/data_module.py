import logging
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


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


class ProteinGODataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """
        Initializes the DataModule with parameters from a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration parameters loaded from YAML.
                Expected keys:
                    - table: Path to the TSV file containing protein-GO term pairs.
                    - prot_emb_file: Path to the protein embeddings H5 file.
                    - text_emb_file: Path to the GO term embeddings H5 file.
                    - dataset: Base path to save/load the cached datasets.
                    - batch_size: Batch size for DataLoader.
                    - num_workers: Number of workers for DataLoader.
                    - seed: Random seed for dataset splitting.
                    - test_size: Proportion of the training dataset to include in the validation split.
        """
        super().__init__()
        self.config = config

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for different stages.

        Args:
            stage (Optional[str]): Stage to set up ('fit', 'validate', 'test', 'predict').
        """
        if stage == "fit":
            ds = ProteinGODataset(
                table=self.config["table"],
                prot_emb_file=self.config["prot_emb_file"],
                text_emb_file=self.config["text_emb_file"],
                group="train_set",
            )

            test_size = self.config["test_size"]
            total_size = len(ds)
            val_size = int(test_size * total_size)
            train_size = total_size - val_size
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                ds, [train_size, val_size], generator=torch.Generator().manual_seed(self.config["seed"])
            )

    def collate_fn(self, batch):
        """
        Custom collate function to batch data.

        Args:
            batch (List[Optional[Dict[str, torch.Tensor]]]): List of samples, some of which may be None.

        Returns:
            Dict[str, torch.Tensor]: Batched tensors with valid datapoints.
        """
        valid_batch = [item for item in batch if item is not None]

        return {
            "prot_embs": torch.stack([item["prot_emb"] for item in valid_batch]),
            "text_embs": torch.stack([item["text_emb"] for item in valid_batch]),
            "labels": torch.stack([item["label"] for item in valid_batch]),
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        pass

    def teardown(self, stage: Optional[str] = None):
        """
        Cleanup after training/testing.

        Args:
            stage (Optional[str]): Stage that is ending ('fit', 'validate', 'test', 'predict').
        """
        pass
