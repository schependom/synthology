"""
Main entry point for training the Recursive Reasoning Network (RRN) model.

This script initializes the Hydra configuration, chooses the appropriate model (batched/exact)
and starts the training process (full RRN = recursive updates + MLPs) using PyTorch Lightning.
"""

import os

import hydra
import pytorch_lightning as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from .dataloading.datamodule import RRNDataModule
from .models.rrn_module import RRNSystem

REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../../..")


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

    # 1. Initialize DataModule
    logger.info("Initializing DataModule...")
    dm = RRNDataModule(cfg)

    # 2. Prepare schema (IMPORTANT: Must be done before initializing model to populate cfg.model.classes/relations)
    logger.info("Scanning schema from training data...")
    try:
        dm.prepare_schema()
    except Exception as e:
        logger.error(f"Failed to scan schema: {e}")
        # Potentially try to load from config if available
        if not (cfg.model.get("classes") and cfg.model.get("relations")):
            raise e

    if not cfg.model.get("classes") or not cfg.model.get("relations"):
        raise ValueError("Model classes or relations not defined in config or schema.")

    logger.info(f"Found {len(cfg.model.classes)} classes and {len(cfg.model.relations)} relations.")

    # 3. Initialize Model
    logger.info("Initializing RRN System...")

    classes, relations = dm.schema.class_names, dm.schema.relation_names
    classes_list = list(classes)
    relations_list = list(relations)

    model = RRNSystem(cfg.model, classes=classes_list, relations=relations_list)

    # 4. Initialize Callbacks
    callbacks = []
    if "callbacks" in cfg:
        for cb_name, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # 5. Initialize Trainer
    logger.info("Initializing Trainer...")
    # Determine max_epochs
    max_epochs = cfg.get("train", {}).get("max_epochs", 100)
    # Check if hyperparams has iterations, logic might require mapping iteration to epochs?
    # RRN usually runs for N iterations PER STEP in inference. But training epochs is different.

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=True,
        default_root_dir=cfg.log_dir,
        precision="16-mixed" if cfg.get("use_amp", False) else 32,
        num_nodes=cfg.get("num_nodes", 1),
        log_every_n_steps=1,
    )

    # 6. Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule=dm)
    logger.info("Training finished.")


if __name__ == "__main__":
    train()
