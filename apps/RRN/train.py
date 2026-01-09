"""
Main entry point for training the Recursive Reasoning Network (RRN) model.

This script initializes the Hydra configuration, sets up the data module and model,
and starts the training process using PyTorch Lightning.
"""

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from src.dataloading.datamodule import RRNDataModule
from src.models.rrn_module import RRNModule


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Configuration object containing all settings for training.
    """
    # Set up the data module
    data_module = RRNDataModule(cfg)

    # Set up the model
    model = RRNModule(cfg)

    # Set up the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.hyperparameters.num_epochs,
        num_nodes=cfg.num_nodes,
        callbacks=cfg.callbacks,
    )

    # Start training
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
