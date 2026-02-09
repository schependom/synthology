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

    # Check for Standard Format files
    facts_path = path / "facts.csv"
    targets_path = path / "targets.csv"
    
    files_to_scan = []
    if facts_path.exists():
        files_to_scan.append(facts_path)
    if targets_path.exists():
        files_to_scan.append(targets_path)
        
    # If standard format files not found, fallback to globbing csvs (Legacy support or flexible)
    if not files_to_scan:
        files_to_scan = sorted(path.glob("sample_*.csv"))
        if not files_to_scan:
             logger.warning(f"No facts.csv/targets.csv or sample_*.csv files found in {data_path}")
             return schema

    logger.info(f"Scanning {len(files_to_scan)} files for schema.")

    for file_path in files_to_scan:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Standard Format: subject, predicate, object
                # Legacy: subject, predicate, object, val, fact_type...

                s = row.get("subject")
                p = row.get("predicate")
                o = row.get("object")
                
                if not p: continue # boiler plate safety

                if p == "rdf:type":
                    schema.add_class(o)
                else:
                    schema.add_relation(p)

    logger.info(f"Scanned schema from {data_path}: {len(schema.classes)} classes, {len(schema.relations)} relations.")
    return schema
