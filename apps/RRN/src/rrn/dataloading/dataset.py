"""
Module defining a PyTorch Dataset for loading and preprocessing knowledge graph samples.
"""

from pathlib import Path

from loguru import logger
from torch.utils.data import Dataset

from rrn.utils.preprocess import preprocess_knowledge_graph
from synthology.data_structures import KnowledgeGraph

from .schema import Schema


class RRNDataset(Dataset):
    def __init__(self, data_path: str, schema: Schema):
        self.data_path = Path(data_path)
        self.schema = schema
        self.files = sorted(self.data_path.glob("sample_*.csv")) if self.data_path.exists() else []

        if not self.files:
            logger.warning(f"No samples found in {data_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]

        kg = KnowledgeGraph.from_csv(file_path=file_path)

        preprocessed = preprocess_knowledge_graph(kg)

        return preprocessed
