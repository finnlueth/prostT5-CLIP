import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import MeanMetric


def binary_contrastive_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes binary contrastive loss.

    Args:
        logits (torch.Tensor): Pairwise cosine similarity scores.
        labels (torch.Tensor): Binary labels (0 or 1).

    Returns:
        torch.Tensor: Binary contrastive loss.
    """
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def binary_clip_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes binary cross-entropy loss in a symmetrical way:
    once on logits, once on logits transposed.

    Args:
        logits (torch.Tensor): Similarity matrix of shape [B, B].
        labels (torch.Tensor): Flattened binary labels of shape [B*B].
    """
    loss_a = binary_contrastive_loss(logits, labels)
    loss_b = binary_contrastive_loss(logits.t(), labels)

    return 0.5 * (loss_a + loss_b)


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
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_cosine = MeanMetric()
        self.val_cosine = MeanMetric()

    def forward(self, prot_emb: torch.Tensor, text_emb: torch.Tensor):
        if self.hparams.config["padding"]:
            prot_emb = self.pad_embeds(prot_emb)

        text_norm = F.normalize(self.text_projection(text_emb), p=2, dim=-1)

        prot_norm = F.normalize(self.prot_projection(prot_emb), p=2, dim=-1)

        return text_norm, prot_norm

    def contrastive_loss(self, text_norm: torch.Tensor, prot_norm: torch.Tensor, labels: torch.Tensor):
        # Compute cosine similarity
        logits_text_prot = (text_norm @ prot_norm.t()) / self.temperature
        logits_prot_text = (prot_norm @ text_norm.t()) / self.temperature

        loss_text_prot = self.criterion(logits_text_prot, labels.float())
        loss_prot_text = self.criterion(logits_prot_text, labels.float())

        return 0.5 * (loss_text_prot + loss_prot_text)

    def training_step(self, batch, batch_idx):
        prot_embs, text_embs, labels = batch["prot_emb"], batch["text_emb"], batch["label"]

        text_norm, prot_norm = self(prot_embs, text_embs)

        # logit_scale = self.model.logit_scale.exp().clamp(max=50)
        loss = self.contrastive_loss(text_norm, prot_norm, torch.diag(labels))

        cosine_sim_matrix = torch.diag(torch.matmul(prot_norm, text_norm.t()))
        postive_cos_sim = cosine_sim_matrix[labels == 1]

        self.train_cosine(postive_cos_sim)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_cosine", self.train_cosine, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        prot_embs, text_embs, labels = batch["prot_emb"], batch["text_emb"], batch["label"]

        text_norm, prot_norm = self(prot_embs, text_embs)

        # logit_scale = self.model.logit_scale.exp().clamp(max=50)
        loss = self.contrastive_loss(text_norm, prot_norm, torch.diag(labels))

        cosine_sim_matrix = torch.diag(torch.matmul(prot_norm, text_norm.t()))
        postive_cos_sim = cosine_sim_matrix[labels == 1]

        self.val_cosine(postive_cos_sim)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_cosine", self.val_cosine, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
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
