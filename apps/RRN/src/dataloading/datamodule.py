from typing import Any, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class RRNDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None
        self.test_dataset: Dataset = None

    def prepare_data(self) -> None:
        # Implement data downloading or preprocessing here if needed
        pass

    def setup(self, stage: str) -> None:
        # Load datasets based on the stage
        if stage == "fit" or stage is None:
            self.train_dataset = self.load_dataset(self.cfg.data.train_data_path)
            self.val_dataset = self.load_dataset(self.cfg.data.val_data_path)

        if stage == "test" or stage is None:
            self.test_dataset = self.load_dataset(self.cfg.data.test_data_path)

    def load_dataset(self, data_path: str) -> Dataset:
        # Placeholder for dataset loading logic
        # Currently left unimplemented
        return NotImplemented

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.cfg.data.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.cfg.data.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.cfg.data.batch_size)
