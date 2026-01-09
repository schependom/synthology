"""
RRNModule class encapsulates
    -   the RRN model architecture,
    -   forward pass,
    -   training logic,
    -   validation logic,
    -   testing logic,
using PyTorch Lightning.
"""

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from .rrn_net import RRNNetwork


class RRNModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super(RRNModule, self).__init__()
        self.config = config
        self.model = RRNNetwork(config.model)  # Initialize the RRN network with model config
        self.criterion = instantiate(config.loss_function)  # Loss function from config

    def forward(self, x):
        return self.model(x)  # Forward pass through the RRN network

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        inputs, targets = batch["inputs"], batch["targets"]
        outputs = self(inputs)  # Forward pass
        loss = self.criterion(outputs, targets)  # Compute loss
        self.log("train_loss", loss)  # Log the training loss
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.config.hyperparameters.optimizer, params=self.parameters())
        return optimizer

    def validation_step(self, batch, batch_idx):
        if hasattr(batch, "get"):  # Check if batch is a dict
            inputs = batch.get("inputs")
            targets = batch.get("targets")
        else:  # Fallback or if batch is already unpacked
            inputs, targets = batch

        # If inputs is None, we skip (or handle properly)
        if inputs is None:
            return

        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss)  # Log the validation loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch["inputs"], batch["targets"]
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("test_loss", loss)
