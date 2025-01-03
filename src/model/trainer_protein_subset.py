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
        self.eval_sample_size = kwargs.pop("eval_sample_size", 32)
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

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        self.train_dataset = self.base_dataset
        return dataloader

    # based on https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/trainer.py#L1056
    # TODO: Subset evaluation dataset during training
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Samples the evaluation dataset and returns a subset of size self.eval_sample_size.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        ).select(random.sample(range(len(self.eval_dataset)), self.eval_sample_size))
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
