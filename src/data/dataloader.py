from pathlib import Path

import h5py
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from ..utils.config import get_params


class EmbeddingDataset(Dataset):
    def __init__(self, prot_emb_file: str, text_emb_file: str):
        self.prot_embs = h5py.File(Path(prot_emb_file).resolve(), "r")
        self.text_embds = h5py.File(Path(text_emb_file).resolve(), "r")
        self.prot_ids = list(self.prot_embs.keys())

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, idx):
        prot_id = self.prot_ids[idx]
        prot_emb = torch.from_numpy(self.prot_embs[prot_id][()])
        text_emb = torch.from_numpy(self.text_embds[prot_id][()])
        return prot_emb, text_emb


def get_train_test_dataloaders(k: int = 5, batch_size: int = 32, num_workers: int = 8):
    dataset = EmbeddingDataset(
        get_params("dataset")["protein_embeddings"],
        get_params("dataset")["text_embeddings"],
    )

    kf = KFold(n_splits=k, shuffle=True, random_state=get_params("train")["seed"])

    loader_kwargs = {
        "batch_size": get_params("train")["batch_size"],
        "num_workers": get_params("train")["num_workers"],
        "pin_memory": True,
    }

    for _, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(dataset, sampler=val_sampler, **loader_kwargs)

        yield train_loader, val_loader
