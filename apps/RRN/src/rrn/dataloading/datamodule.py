"""
Module defining a PyTorch Lightning DataModule for loading KG datasets.
"""

import pytorch_lightning as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from .dataset import RRNDataset
from .schema import Schema, scan_schema


def collate_graph_batch(batch):
    return batch[0]


class RRNDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None
        self.test_dataset: Dataset = None
        self.schema: Schema = None

        self.populate_schema()

    def populate_schema(self):
        """Helper to scan schema and populate config."""

        # Use train path for scanning schema
        train_path = self.cfg.data.dataset.train_path
        self.schema = scan_schema(train_path)

        # If classes and relations are found, populate the config accordingly
        if "model" in self.cfg:
            logger.warning("Populating model classes and relations from scanned schema.")
            OmegaConf.set_struct(self.cfg, False)  # Allow modifications
            self.cfg.model.classes = self.schema.class_names
            self.cfg.model.relations = self.schema.relation_names
            OmegaConf.set_struct(self.cfg, True)  # Prevent further modifications

    def setup(self, stage: str) -> None:
        if self.schema is None:
            self.populate_schema()

        # Load datasets based on the stage
        if stage == "fit" or stage is None:
            self.train_dataset = self.load_dataset(self.cfg.data.dataset.train_path)
            self.val_dataset = self.load_dataset(self.cfg.data.dataset.val_path)

        if stage == "test" or stage is None:
            self.test_dataset = self.load_dataset(self.cfg.data.dataset.test_path)

    def load_dataset(self, data_path: str) -> Dataset:
        # Use our CSVReader dataset
        return RRNDataset(data_path, self.schema)

    def train_dataloader(self) -> DataLoader:
        # One batch = one graph
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_graph_batch,
            num_workers=self.cfg.data.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        # One batch = one graph
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=collate_graph_batch,
            num_workers=self.cfg.data.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # One batch = one graph
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=collate_graph_batch,
            num_workers=self.cfg.data.num_workers,
        )

    def get_classes_and_relations(self):
        return self.schema.class_names, self.schema.relation_names
