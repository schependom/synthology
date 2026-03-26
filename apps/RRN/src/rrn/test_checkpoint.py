"""Evaluate an RRN checkpoint on the configured test split."""

import glob
import os
import warnings

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from .dataloading.datamodule import RRNDataModule
from .models.rrn_module import RRNSystem

# Load environment variables from .env file
load_dotenv()

REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")


def _resolve_checkpoint_path(cfg: DictConfig) -> str:
    explicit_path = cfg.get("test", {}).get("checkpoint_path", None)
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Configured checkpoint_path does not exist: {explicit_path}")

    checkpoint_glob = cfg.get("test", {}).get("checkpoint_glob", None)
    if not checkpoint_glob:
        ckpt_cfg = cfg.get("callbacks", {}).get("model_checkpoint", {})
        dirpath = ckpt_cfg.get("dirpath", "./checkpoints")
        filename = ckpt_cfg.get("filename", "best-checkpoint")
        checkpoint_glob = os.path.join(dirpath, f"{filename}*.ckpt")

    matches = glob.glob(checkpoint_glob)
    if not matches:
        raise FileNotFoundError(f"No checkpoints matched pattern: {checkpoint_glob}")

    # Pick the most recently modified checkpoint.
    return max(matches, key=os.path.getmtime)


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/rrn", config_name="config")
def test_checkpoint(cfg: DictConfig) -> None:
    logger.info(f"Starting RRN checkpoint evaluation with configuration:\n{OmegaConf.to_yaml(cfg)}")

    pl.seed_everything(42, workers=True)

    logger.info("Initializing DataModule...")
    dm = RRNDataModule(cfg)
    classes, relations = dm.get_classes_and_relations()
    logger.info(f"Found {len(classes)} classes and {len(relations)} relations.")

    model = RRNSystem(cfg, classes=classes, relations=relations)

    checkpoint_path = _resolve_checkpoint_path(cfg)
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    pl_logger = True
    if "logger" in cfg:
        logger.info(f"Instantiating logger <{cfg.logger._target_}>")
        pl_logger = hydra.utils.instantiate(cfg.logger)

    precision = "16-mixed" if cfg.get("use_amp", False) and torch.cuda.is_available() else 32

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=pl_logger,
        default_root_dir=cfg.log_dir,
        precision=precision,
        log_every_n_steps=1,
    )

    logger.info("Starting checkpoint evaluation on test split...")
    trainer.test(model, datamodule=dm)
    logger.info("Checkpoint evaluation finished.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", ".*You have overridden `transfer_batch_to_device` in `LightningModule`.*")
    test_checkpoint()
