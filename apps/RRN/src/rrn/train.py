"""
Main entry point for training the Recursive Reasoning Network (RRN) model.

This script initializes the Hydra configuration, chooses the appropriate model (batched/exact)
and starts the training process (full RRN = recursive updates + MLPs) using PyTorch Lightning.
"""

import os
import warnings

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar

# Load environment variables from .env file
load_dotenv()
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar

from .dataloading.datamodule import RRNDataModule
from .models.rrn_module import RRNSystem

REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/rrn", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Configuration object containing all settings for training.
    """

    logger.info(f"Starting RRN training with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    pl.seed_everything(42, workers=True)

    # Initialize DataModule
    # and populate schema
    logger.info("Initializing DataModule...")
    dm = RRNDataModule(cfg)
    classes, relations = dm.get_classes_and_relations()
    logger.info(f"Found {len(classes)} classes and {len(relations)} relations.")

    # Initialize Model

    logger.info("Initializing RRN System...")

    # Check if they are actually populated
    if not classes or not relations:
        logger.warning("Classes or Relations list is empty. Model might not initialize correctly.")

    logger.info(f"Passed {len(classes)} classes and {len(relations)} relations to RRNSystem.")
    model = RRNSystem(cfg, classes=classes, relations=relations)

    # Initialize Callbacks
    callbacks = [
        # Use Rich for the nice colorful progress bar
        RichProgressBar(),
        # Use Rich for the nice colorful model summary table
        RichModelSummary(max_depth=2),
    ]

    if "callbacks" in cfg:
        for cb_name, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Initialize Trainer
    logger.info("Initializing Trainer...")
    # Determine max_epochs
    max_epochs = cfg.max_epochs if cfg.max_epochs is not None else 100

    # Initialize Logger
    pl_logger = True
    if "logger" in cfg:
        logger.info(f"Instantiating logger <{cfg.logger._target_}>")
        pl_logger = hydra.utils.instantiate(cfg.logger)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,  # num of GPUs
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=pl_logger,
        default_root_dir=cfg.log_dir,
        precision="16-mixed" if cfg.get("use_amp", False) else 32,
        # num_nodes=cfg.get("num_nodes", 1), # for multi-node training
        log_every_n_steps=1,
    )

    # 6. Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule=dm)
    logger.info("Training finished.")

    # 7. Test
    if cfg.get("do_test", True):
        logger.info("Starting testing...")
        trainer.test(model, datamodule=dm)
        logger.info("Testing finished.")


if __name__ == "__main__":
    # Add this near the top of train.py
    warnings.filterwarnings("ignore", ".*Precision 16-mixed is not supported by the model summary.*")
    warnings.filterwarnings("ignore", ".*You have overridden `transfer_batch_to_device` in `LightningModule`.*")

    train()
