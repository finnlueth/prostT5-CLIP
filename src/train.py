import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.data_module import ProteinGODataModule
from src.models.model import ProteinCLIP
from src.utils.config import get_params


def main():
    logging.basicConfig(level=logging.INFO)

    config = get_params(Path(__file__).stem)

    wandb_logger = WandbLogger(
        project=config.get("project", "protein-clip"),
        name=config.get("run_name", "experiment"),
        config=config,
    )

    data_module = ProteinGODataModule(config=config)
    data_module.setup(stage="fit")

    model = ProteinCLIP(config=config)

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode="min",
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best-checkpoint",
    )

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu",
        devices=1,
        precision="32-true",
        gradient_clip_val=None,
        callbacks=[early_stop, checkpoint],
        logger=wandb_logger,
        enable_model_summary=True,
        enable_progress_bar=False,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)


if __name__ == "__main__":
    main()
