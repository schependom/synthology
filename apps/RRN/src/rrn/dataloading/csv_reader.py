import csv
import logging
from collections import namedtuple
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Simple structures satisfying Protocols in rrn_net.py
# Using namedtuple for memory efficiency and simplicity
Entity = namedtuple("Entity", ["index", "name"])
Class = namedtuple("Class", ["index", "name"])
Relation = namedtuple("Relation", ["index", "name"])


class Triple:
    def __init__(self, subject: Entity, predicate: Relation, object: Entity, positive: bool):
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.positive = positive


class Schema:
    def __init__(self):
        self.classes: Dict[str, int] = {}
        self.relations: Dict[str, int] = {}
        self.class_names: List[str] = []
        self.relation_names: List[str] = []

    def add_class(self, name: str) -> int:
        if name not in self.classes:
            self.classes[name] = len(self.class_names)
            self.class_names.append(name)
        return self.classes[name]

    def add_relation(self, name: str) -> int:
        if name not in self.relations:
            self.relations[name] = len(self.relation_names)
            self.relation_names.append(name)
        return self.relations[name]

    def get_class_index(self, name: str) -> int:
        return self.classes.get(name)

    def get_relation_index(self, name: str) -> int:
        return self.relations.get(name)


def scan_schema(data_path: str) -> Schema:
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
    # (Simplified: assume first file represents schema or scan all if needed)
    # For correctness we should scan all, but let's scan first 10 for performance approximation
    # or just scan one and assume consistent ontology.
    # Better: Scan ALL because different samples might use different subsets of relations/classes
    # if the ontology is large.
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


class RRNDataset(Dataset):
    def __init__(self, data_path: str, schema: Schema):
        self.data_path = Path(data_path)
        self.schema = schema
        self.files = sorted(self.data_path.glob("sample_*.csv")) if self.data_path.exists() else []

        if not self.files:
            logger.warning(f"No samples found in {data_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        # Local to this graph
        individuals: Dict[str, int] = {}
        individual_names: List[str] = []

        def get_individual(name):
            if name not in individuals:
                individuals[name] = len(individual_names)
                individual_names.append(name)
            return Entity(individuals[name], name)

        # We need two passes or store rows. Since files are small-ish (samples), storing rows is fine.
        rows = []
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # First pass: identify individuals
        for row in rows:
            get_individual(row["subject"])
            if row["fact_type"] == "triple":
                get_individual(row["object"])

        num_individuals = len(individual_names)
        num_classes = len(self.schema.classes)

        # Initialize memberships with 0
        memberships = [[0] * num_classes for _ in range(num_individuals)]

        input_triples: List[Triple] = []

        for row in rows:
            subj = get_individual(row["subject"])

            fact_type = row["fact_type"]
            label = int(row["label"])
            is_inferred = int(row["is_inferred"])

            if fact_type == "membership":
                class_name = row["object"]
                if class_name in self.schema.classes:
                    c_idx = self.schema.classes[class_name]
                    # If it's a base fact (is_inferred=0) and True (label=1), add to memberships input
                    if is_inferred == 0 and label == 1:
                        memberships[subj.index][c_idx] = 1

            elif fact_type == "triple":
                pred_name = row["predicate"]
                if pred_name in self.schema.relations:
                    p_idx = self.schema.relations[pred_name]
                    predicate = Relation(p_idx, pred_name)
                    obj = get_individual(row["object"])

                    # If it's a base fact and True, add to input triples
                    if is_inferred == 0 and label == 1:
                        input_triples.append(Triple(subj, predicate, obj, positive=True))

        # Prepare the return dict
        # The key 'inputs' will be passed to model(inputs)
        model_inputs = {"triples": input_triples, "memberships": memberships}

        return {
            "inputs": model_inputs,
            "targets": torch.tensor([]),  # Dummy target
        }
