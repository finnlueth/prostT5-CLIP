from transformers import Trainer
from torch.utils.data import DataLoader
from collections import defaultdict
import random

from transformers.trainer_utils import (
    seed_worker,
)

from transformers.utils import (
    is_datasets_available,
)

import datasets
import torch


def _group_by_protein(dataset):
    """Group dataset entries by protein."""
    protein_groups = defaultdict(list)
    for i, entry in enumerate(dataset):
        protein = entry["identifier"]
        protein_groups[protein].append(i)
    return protein_groups


class ProteinSampleSubsetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protein_groups = _group_by_protein(self.train_dataset)
        self.base_dataset = self.train_dataset

    def _sample_subset(self):
        """Sample one text-pair per protein."""
        sampled_indices = [random.choice(indices) for indices in self.protein_groups.values()]
        selected_dataset = self.train_dataset.select(sampled_indices)
        return selected_dataset

    # based on https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/trainer.py#L978
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self._sample_subset()
        self.train_dataset = train_dataset
        
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
