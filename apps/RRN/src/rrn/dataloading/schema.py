"""
Module for scanning and representing the schema of a relational dataset.
It keeps track of all unique classes and relations found in the dataset.
"""

import csv
from pathlib import Path
from typing import Dict, List

from loguru import logger

from synthology.data_structures import Class, Relation


class Schema:
    def __init__(self):
        self.class_map: Dict[str, int] = {}
        self.relation_map: Dict[str, int] = {}
        self.classes: List[Class] = []
        self.relations: List[Relation] = []

    @property
    def class_names(self) -> List[str]:
        return [c.name for c in self.classes]

    @property
    def relation_names(self) -> List[str]:
        return [r.name for r in self.relations]

    def add_class(self, name: str) -> int:
        if name not in self.class_map:
            idx = len(self.classes)
            self.class_map[name] = idx
            self.classes.append(Class(idx, name))
        return self.class_map[name]

    def add_relation(self, name: str) -> int:
        if name not in self.relation_map:
            idx = len(self.relations)
            self.relation_map[name] = idx
            self.relations.append(Relation(idx, name))
        return self.relation_map[name]

    def get_class_index(self, name: str) -> int:
        return self.class_map[name]

    def get_relation_index(self, name: str) -> int:
        return self.relation_map[name]


def scan_schema(data_path: str) -> Schema:
    logger.info(f"Scanning schema from data path: {data_path}")

    schema = Schema()
    path = Path(data_path)
    if not path.exists():
        logger.warning(f"Data path {data_path} does not exist.")
        return schema

    files = sorted(path.glob("*.csv"))
    if not files:
        logger.warning(f"No CSV files found in {data_path}")
        return schema

    # Scan all files to ensure global consistency
    logger.info(f"Found {len(files)} files to scan for schema.")
    for file_path in files:
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fact_type = row["fact_type"]
                if fact_type == "membership":
                    schema.add_class(row["object"])
                elif fact_type == "triple":
                    schema.add_relation(row["predicate"])

    logger.info(f"Scanned schema from {data_path}: {len(schema.classes)} classes, {len(schema.relations)} relations.")
    return schema
