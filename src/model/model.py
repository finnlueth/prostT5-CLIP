from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
import umap
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchmetrics import MeanMetric

import wandb


class BinaryContrastiveLoss(nn.Module):
    """
    Binary contrastive loss module that computes loss in a symmetrical way.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes binary cross-entropy loss in a symmetrical way:
        once on logits, once on logits transposed.

        Args:
            logits (torch.Tensor): Similarity matrix of shape [B, B].
            labels (torch.Tensor): Flattened binary labels of shape [B*B].

        Returns:
            torch.Tensor: Symmetrical binary contrastive loss.
        """
        assert logits.shape == labels.shape, "Logits and labels must have the same shape."

        loss_a = self.criterion(logits, labels)
        loss_b = self.criterion(logits.t(), labels)

        return 0.5 * (loss_a + loss_b)


class CosineContrastiveLoss(nn.Module):
    """
    Cosine contrastive loss module that computes loss in a symmetrical way.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.criterion = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, prot_norm: torch.Tensor, text_norm: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine contrastive loss in a symmetrical way:
        once on prot_norm, once on text_norm.

        Args:
            prot_norm (torch.Tensor): normalized Protein embeddings of shape [B, D].
            text_norm (torch.Tensor): normalized Text embeddings of shape [B, D].
            labels (torch.Tensor): Binary labels (1 or 0) of shape [B].

        Returns:
            torch.Tensor: Symmetrical cosine contrastive loss.
        """
        targets = labels.clone()
        targets[targets == 0] = -1

        loss = self.criterion(prot_norm, text_norm, targets)

        return loss


class ContrastiveNTXentLoss(nn.Module):
    """
    Contrastive Normalized Temperature-scaled Cross-Entropy Loss.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, prot_emb: torch.Tensor, text_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes contrastive loss for positive and negative pairs.

        Args:
            prot_emb (torch.Tensor): Embeddings of shape [B, D].
            text_emb (torch.Tensor): Embeddings of shape [B, D].
            labels (torch.Tensor): Binary labels of shape [B].

        Returns:
            torch.Tensor: Contrastive loss.
        """
        prot_norm = F.normalize(prot_emb, p=2, dim=-1)
        text_norm = F.normalize(text_emb, p=2, dim=-1)

        similarity = torch.sum(prot_norm * text_norm, dim=-1) / self.temperature

        return self.criterion(similarity, labels)


class EuclideanContrastiveLoss(nn.Module):
    """
    Euclidean contrastive loss module that encourages small distances for positive pairs
    and distances greater than margin for negative pairs.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initializes the EuclideanContrastiveLoss module.

        Args:
            margin (float): Margin for negative pairs. Distances above this margin incur no loss.
        """
        super().__init__()
        self.margin = margin
        self.criterion = nn.TripletMarginLoss(margin=margin, p=2)


class ProjectionHead(nn.Module):
    """
    A flexible projection head that can have a variable number of hidden layers.
    Each hidden layer consists of Linear -> LayerNorm -> GELU -> Dropout.
    """

    def __init__(self, input_dim: int, intermediate_dim: int, projected_dim: int, num_hidden: int, dropout: float):
        super(ProjectionHead, self).__init__()
        layers = []
        current_dim = input_dim

        for _ in range(num_hidden):
            layers.extend(
                [
                    nn.Linear(current_dim, intermediate_dim),
                    nn.LayerNorm(intermediate_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = intermediate_dim

        layers.extend([nn.Linear(current_dim, projected_dim), nn.LayerNorm(projected_dim)])

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ProteinCLIP(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()

        self.save_hyperparameters()

        self.text_projection = ProjectionHead(
            input_dim=config["text_embed_dim"],
            intermediate_dim=config["intermediate_dim"],
            projected_dim=config["projected_dim"],
            num_hidden=config["num_hidden"],
            dropout=config["dropout"],
        )

        self.prot_projection = ProjectionHead(
            input_dim=config["prot_embed_dim"],
            intermediate_dim=config["intermediate_dim"],
            projected_dim=config["projected_dim"],
            num_hidden=config["num_hidden"],
            dropout=config["dropout"],
        )

        self.learning_rate = config.get("learning_rate", 1e-4)
        self.temperature = nn.Parameter(
            torch.tensor(config.get("temperature", 0.07)), requires_grad=config["logit_scale_required"]
        )

        self.loss = globals()[config["loss"]](**config["loss_params"])

        self.train_cosine = MeanMetric()
        self.val_cosine = MeanMetric()

        self.input_embeddings = {"prot_id": [], "go_term": [], "prot_emb": [], "text_emb": [], "label": []}
        self.output_embeddings = {"prot_id": [], "go_term": [], "prot_emb": [], "text_emb": [], "label": []}

        self.go_sentence = config["go_sentence"]
        self.out = config["out"]
        self.reducer = umap.UMAP(**config["umap_params"])

    def forward(self, prot_emb: torch.Tensor, text_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prot_norm = F.normalize(self.prot_projection(prot_emb), p=2, dim=-1)
        text_norm = F.normalize(self.text_projection(text_emb), p=2, dim=-1)

        return prot_norm, text_norm

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        prot_embs, text_embs, labels = batch["prot_emb"], batch["text_emb"], batch["label"]

        prot_norm, text_norm = self(prot_embs, text_embs)

        # logit_scale = self.model.logit_scale.exp().clamp(max=50)
        loss = self.loss(prot_norm, text_norm, labels)

        cosine_sim = F.cosine_similarity(prot_norm, text_norm, dim=-1)
        positive_cos_sim = cosine_sim[labels == 1]

        self.train_cosine(positive_cos_sim)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_cosine", self.train_cosine, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        prot_embs, text_embs, labels = batch["prot_emb"], batch["text_emb"], batch["label"]

        prot_norm, text_norm = self(prot_embs, text_embs)

        # logit_scale = self.model.logit_scale.exp().clamp(max=50)
        loss = self.loss(prot_norm, text_norm, labels)

        cosine_sim = F.cosine_similarity(prot_norm, text_norm, dim=-1)
        positive_cos_sim = cosine_sim[labels == 1]

        self.val_cosine(positive_cos_sim)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_cosine", self.val_cosine, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        prot_embs, text_embs, labels = batch["prot_emb"], batch["text_emb"], batch["label"]
        prot_norm, text_norm = self(prot_embs, text_embs)

        loss = self.loss(prot_norm, text_norm, labels)

        cosine_sim = F.cosine_similarity(prot_norm, text_norm, dim=-1)
        positive_cos_sim = cosine_sim[labels == 1]
        negative_cos_sim = cosine_sim[labels == 0]

        """
        self.input_embeddings["prot_id"].extend(batch["prot_id"])
        self.input_embeddings["go_term"].extend(batch["go_term"])
        self.input_embeddings["prot_emb"].append(prot_norm.detach().cpu().numpy())
        self.input_embeddings["text_emb"].append(text_norm.detach().cpu().numpy())
        self.input_embeddings["label"].extend(labels.cpu().numpy())
        """

        self.output_embeddings["prot_id"].extend(batch["prot_id"])
        self.output_embeddings["go_term"].extend(batch["go_term"])
        self.output_embeddings["prot_emb"].append(prot_norm.detach().cpu().numpy())
        self.output_embeddings["text_emb"].append(text_norm.detach().cpu().numpy())
        self.output_embeddings["label"].extend(labels.cpu().numpy())

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("positive_cosine", positive_cos_sim.mean(), prog_bar=True, on_step=False, on_epoch=True)
        self.log("negative_cosine", negative_cos_sim.mean(), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self) -> None:
        """
        Save output embeddings into numpy .npz file.

        Args:
            out (Path): Output directory to save embeddings.
        """
        prot_embs = np.vstack(self.output_embeddings["prot_emb"])
        text_embs = np.vstack(self.output_embeddings["text_emb"])

        to_save = {
            "prot_id": np.array(self.output_embeddings["prot_id"]),
            "go_term": np.array(self.output_embeddings["go_term"]),
            "prot_emb": prot_embs,
            "text_emb": text_embs,
            "label": np.array(self.output_embeddings["label"]),
        }

        np.savez_compressed(self.out, **to_save)

        wandb.log({"output_embeddings": wandb.Table(dataframe=pd.DataFrame(to_save["prot_emb"]))})

        combined = np.concatenate((to_save["prot_emb"], to_save["text_emb"]), axis=1)

        namespace = self.load_go_mapping(self.go_sentence)

        sc = StandardScaler()
        X_s = sc.fit_transform(combined)
        X_umap = self.reducer.fit_transform(X_s)

        umap = pd.DataFrame(
            {
                "UMAP1": X_umap[:, 0],
                "UMAP2": X_umap[:, 1],
                "Namespace": [namespace[go_term] for go_term in to_save["go_term"]],
                "Label": to_save["label"],
            }
        )

        umap = umap[umap["Label"] == 1]

        wandb.log({"umap": wandb.Table(dataframe=umap)})

        sns.scatterplot(
            data=umap,
            x="UMAP1",
            y="UMAP2",
            hue="Namespace",
            palette="tab20",
            marker=".",
            s=10,
        ).get_figure().savefig("umap.png")

        self.output_embeddings = {
            "prot_id": [],
            "go_term": [],
            "prot_emb": [],
            "text_emb": [],
            "label": [],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    @staticmethod
    def pad_embeds(prot_emb: torch.Tensor, target: int = 3072) -> torch.Tensor:
        """
        Pads ProtT5 embeddings from (1024,) to (3072,) with zeros.

        Args:
            prot_emb (torch.Tensor): ProtT5 embedding of shape (1024,).
            target_dim (int): Target embedding dimension (default is 3072).

        Returns:
            torch.Tensor: Padded embedding of shape (3072,).
        """
        batch_size, emb_dim = prot_emb.size()
        padding = torch.zeros(batch_size, target - emb_dim, device=prot_emb.device)
        padded_emb = torch.cat([prot_emb, padding], dim=1)
        return padded_emb

    @staticmethod
    def load_go_mapping(go_sentence: Path) -> dict[str, str]:
        """
        Load GO term mapping from preprocessed GO sentence file.

        Args:
            go_sentence (Path): Path to the GO term mapping file.

        Returns:
            dict[str, str]: GO term mapping.
        """

        metadata = pd.read_csv(go_sentence, sep="\t")

        return pd.Series(metadata.namespace.values, index=metadata.term).to_dict()
