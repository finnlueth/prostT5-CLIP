from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datasets import ProteinGODataset


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
