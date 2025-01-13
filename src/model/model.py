import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import CLIPModel
from transformers.models.clip.modeling_clip import clip_loss


class ProteinCLIP(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        self.model = CLIPModel.from_pretrained(config["model"])

        self.model.config.text_config.projection_dim = config.get(
            "projection_dim", 1024
        )
        self.model.config.vision_config.projection_dim = config.get(
            "projection_dim", 1024
        )

        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if "projection" in name:
                param.requires_grad = True
            if "logit_scale" in name:
                param.requires_grad = config.get("logit_scale_requried", False)

        """
        self.compress = nn.Sequential(
            nn.Linear(3072, 2048), nn.LayerNorm(2048), nn.GELU(), nn.Linear(2048, 1024), nn.LayerNorm(1024)
        )
        """

        self.model.train()

        self.learning_rate = config.get("learning_rate", 1e-4)

    def forward(self, prot_emb: torch.Tensor, go_emb: torch.Tensor):
        if self.hparams.config["padding"]:
            prot_emb = self.pad_embeds(prot_emb)

        prot_emb = F.normalize(prot_emb, dim=-1)
        go_emb = F.normalize(go_emb, dim=-1)

        return prot_emb, go_emb

    def training_step(self, batch, batch_idx):
        prot_embs, text_embs = batch["prot_embs"], batch["text_embs"]
        prot_embs = prot_embs.detach()

        prot_emb, go_emb = self(prot_embs, text_embs)

        logit_scale = self.model.logit_scale.exp().clamp(max=50)
        logits = logit_scale * torch.matmul(prot_emb, go_emb.t())

        loss = clip_loss(logits)

        cosine_sim = F.cosine_similarity(prot_emb, go_emb, dim=-1).mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_cosine_similarity",
            cosine_sim,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        prot_embs, text_embs = batch["prot_embs"], batch["text_embs"]
        prot_embs = prot_embs.detach()

        prot_emb, go_emb = self(prot_embs, text_embs)

        logit_scale = self.model.logit_scale.exp().clamp(max=50)
        logits = logit_scale * torch.matmul(prot_emb, go_emb.t())

        loss = clip_loss(logits)

        cosine_sim = F.cosine_similarity(prot_emb, go_emb, dim=-1).mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_cosine_similarity",
            cosine_sim,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3
        )
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
