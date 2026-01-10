import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from .csv_reader import RRNDataset, Schema, scan_schema


class RRNDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None
        self.test_dataset: Dataset = None
        self.schema: Schema = None

    def prepare_data(self) -> None:
        # Scan schema from training data to ensure we know all classes/relations
        # TODO

    def prepare_schema(self) -> None:
        """Helper to scan schema and populate config."""
        # Use train path for scanning schema
        train_path = self.cfg.data.dataset.train_path
        self.schema = scan_schema(train_path)

        # Inject schema into config so model can initialize correctly
        if "model" in self.cfg:
            OmegaConf.set_struct(self.cfg, False)
            self.cfg.model.classes = self.schema.class_names
            self.cfg.model.relations = self.schema.relation_names
            OmegaConf.set_struct(self.cfg, True)

    def setup(self, stage: str) -> None:
        if self.schema is None:
            self.prepare_schema()

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
        # collate_fn might be needed if inputs are complex objects (dictionaries with lists)
        # However, PyTorch default collate works for dicts of tensors.
        # But 'triples' is a List[Triple], which default collate might complain about (can't stack Objects).
        # We need a custom collate_fn if we want to batch multiple graphs.
        # But RRN usually processes 1 graph at a time or we force batch_size=1

        # If batch_size > 1, default collate tries to stack everything.
        # List[Triple] cannot be stacked.

        # For this fix, let's force batch_size=1 or use a custom collate that basically returns a list of graph dicts?
        # Standard RRN usually takes ONE graph per step if the graph is large.
        # Here we have small family trees.

        def collate_fn(batch):
            # batch is a list of results from __getitem__
            # We want to keep them separate because each is a different graph structure
            # But Lightning expects a Batch object usually.
            # If we just return the list, Lightning might not like it passed to 'forward'.

            # Since RRNNetwork forward expects keys "triples", "memberships",
            # it likely expects ONE graph corresponding to the batch
            # OR the model handles a LIST of graphs.
            # Reading RRNNetwork.forward again:
            # "triples: List[TripleProtocol]"
            # This implies ONE graph.
            # So batch_size MUST be 1 for this model implementation.
            return batch[0]

        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self) -> DataLoader:
        def collate_fn(batch):
            return batch[0]

        return DataLoader(self.val_dataset, batch_size=1, collate_fn=collate_fn)

    def test_dataloader(self) -> DataLoader:
        def collate_fn(batch):
            return batch[0]

        return DataLoader(self.test_dataset, batch_size=1, collate_fn=collate_fn)
