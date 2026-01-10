"""
Main entry point for training the Recursive Reasoning Network (RRN) model.

This script initializes the Hydra configuration, chooses the appropriate model (batched/exact)
and starts the training process (full RRN = recursive updates + MLPs) using PyTorch Lightning.
"""

import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/rrn", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Configuration object containing all settings for training.
    """

    logger.info(f"Starting RRN training with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # ....


if __name__ == "__main__":
    train()
