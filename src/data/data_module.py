from typing import Any, Dict, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Sampler

from datasets import load_from_disk


class H5Reader:
    """Wrapper for h5py to provide dict-like access without loading all data to memory"""

    def __init__(self, file_path: str, group: Optional[str] = None):
        self.file = h5py.File(file_path, "r")
        self.dataset = self.file[group] if group else self.file

    def __getitem__(self, key):
        return torch.from_numpy(self.dataset[key][()].astype(np.float32))

    def __del__(self):
        self.file.close()


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.positives = np.where(self.labels == 1)[0]
        self.negatives = np.where(self.labels == 0)[0]
        self.num_classes = 2
        assert self.batch_size % self.num_classes == 0, "Batch size must be divisible by number of classes"
        self.samples_per_class = self.batch_size // self.num_classes

    def __iter__(self):
        rng = np.random.default_rng()
        rng.shuffle(self.positives)
        rng.shuffle(self.negatives)

        num_batches = min(len(self.positives), len(self.negatives)) // self.samples_per_class
        for i in range(num_batches):
            pos_indices = self.positives[i * self.samples_per_class : (i + 1) * self.samples_per_class]
            neg_indices = self.negatives[i * self.samples_per_class : (i + 1) * self.samples_per_class]
            yield list(pos_indices) + list(neg_indices)

    def __len__(self):
        return min(len(self.positives), len(self.negatives)) // self.samples_per_class


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
                    - dataset: Base path to save/load the cached datasets.
                    - prot_emb_file: Path to the protein embeddings H5 file.
                    - text_emb_file: Path to the GO term embeddings H5 file.
                    - batch_size: Batch size for DataLoader.
                    - num_workers: Number of workers for DataLoader.
                    - seed: Random seed for dataset splitting.
                    - val_size: Proportion of the dataset to include in the validation split.
        """
        super().__init__()
        self.config = config
        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.prot_loader = H5Reader(config["prot_emb_file"])
        self.text_loader = H5Reader(config["text_emb_file"])

    def prepare_data(self):
        """load dataset from disk"""
        self.dataset = load_from_disk(self.config["dataset"])

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for different stages.

        Args:
            stage (Optional[str]): Stage to set up ('fit', 'validate', 'test', 'predict').
        """

        self.prepare_data()

        if stage in ("fit", "validate"):
            ds = self.dataset["train"]
            split = ds.train_test_split(
                test_size=self.config["val_size"], seed=self.config["seed"], stratify_by_column="label"
            )

            self.train_ds = split["train"]
            self.val_ds = split["test"]

        if stage == "test":
            self.test_ds = self.dataset["test"]

    def collate_fn(self, batch):
        """
        Custom collate function to batch data.

        Args:
            batch (List[Optional[Dict[str, torch.Tensor]]]): List of samples, some of which may be None.

        Returns:
            Dict[str, torch.Tensor]: Batched tensors with valid datapoints.
        """

        prot_id, term_id, labels = [], [], []

        for item in batch:
            if item is not None:
                prot_id.append(item["protein"])
                term_id.append(item["term"])
                labels.append(int(item["label"]))

        return {
            "prot_emb": torch.stack([self.prot_loader[pid] for pid in prot_id]),
            "text_emb": torch.stack([self.text_loader[tid] for tid in term_id]),
            "label": torch.tensor(labels, dtype=torch.long),
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
        return DataLoader(
            self.test_ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
        )

    def teardown(self, stage: Optional[str] = None):
        """
        Cleanup after training/testing.

        Args:
            stage (Optional[str]): Stage that is ending ('fit', 'validate', 'test', 'predict').
        """
        pass
