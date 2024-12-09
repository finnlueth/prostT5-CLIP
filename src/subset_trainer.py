from transformers import Trainer
from torch.utils.data import DataLoader
from collections import defaultdict
import random

class SubsetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protein_groups = self._group_by_protein(self.train_dataset)

    def _group_by_protein(self, dataset):
        """Group dataset entries by protein."""
        protein_groups = defaultdict(list)
        for i, entry in enumerate(dataset):
            protein = entry["identifier"]
            protein_groups[protein].append(i)
        return protein_groups

    def _sample_subset(self):
        """Sample one text-pair per protein."""
        sampled_indices = [
            random.choice(indices) for indices in self.protein_groups.values()
        ]
        #print(sampled_indices)
        return self.train_dataset.select(sampled_indices)

    def get_train_dataloader(self,):
        # Sample a new subset of the dataset
        epoch_subset = self._sample_subset()

        # Create DataLoader for the subset
        dataloader = DataLoader(
            epoch_subset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return dataloader
