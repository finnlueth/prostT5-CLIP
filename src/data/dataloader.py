import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EmbeddingDataset(Dataset):
    def __init__(self, prot_emb_file: str, text_emb_file: str):
        """Load protein and text embeddings from h5 files"""
        self.prot_embs = h5py.File(prot_emb_file, "r")
        self.text_embds = h5py.File(text_emb_file, "r")

        self.prot_ids = list(self.seq_embs.keys())

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self):
        # Random sample a protein embedding
        prot_id = np.random.choice(self.prot_ids)
        prot_emb = torch.from_numpy(self.seq_embs[prot_id][()])

        text_emb = torch.from_numpy(self.text_embds[prot_id][()])

        return prot_emb, text_emb

    def close(self):
        self.protein_data.close()
        self.text_data.close()


def get_dataloader(batch_size=32, num_workers=8):
    dataset = EmbeddingDataset(
        "embeddings/protein_embeddings.h5", "embeddings/text_embeddings.h5"
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
