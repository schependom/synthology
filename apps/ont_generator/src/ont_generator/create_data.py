"""
DESCRIPTION

    Train/Test Data Generator

    Generates independent knowledge graph samples for training and testing.
    Each sample is a complete KG with unique individuals, base facts,
    derived inferences, and balanced positive/negative examples.

AUTHOR

    Vincent Van Schependom
"""

import csv
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import networkx as nx
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ont_generator.generate import KGenerator, atoms_to_knowledge_graph, extract_proof_map
from ont_generator.negative_sampler import NegativeSampler
from ont_generator.utils.validator import Validator
from synthology.data_structures import KnowledgeGraph


class KGEDatasetGenerator:
    """
    Generates training and testing datasets for knowledge graph embedding models.
    """

    def __init__(
        self,
        cfg: DictConfig,
        verbose: bool,
    ):
        """
        Initialize dataset generator.

        Args:
            cfg:        Hydra configuration object
            verbose:    Enable detailed logging
        """
        if cfg.dataset.seed is not None:
            random.seed(cfg.dataset.seed)

        self.verbose = verbose
        self.max_recursion_cap = cfg.generator.max_recursion
        self.individual_pool_size = cfg.generator.individual_pool_size
        self.individual_reuse_prob = cfg.generator.individual_reuse_prob
        self.export_proofs = cfg.export_proofs
        self.output_dir = cfg.dataset.output_dir
        self.proof_output_dir = cfg.get("proof_output_path", os.path.join(self.output_dir, "proofs")) if self.export_proofs else None

        self.min_proof_roots = cfg.generator.get("min_proof_roots", 5)
        self.max_proof_roots = cfg.generator.get("max_proof_roots", 15)

        # Initialize KGenerator
        self.generator = KGenerator(
            cfg=cfg,
            verbose=False,  # Keep generator quiet during batch generation
        )

        # Store schema references
        # schemas map names to Class/Relation/Attribute objects
        self.schema_classes = self.generator.schema_classes
        self.schema_relations = self.generator.schema_relations
        self.schema_attributes = self.generator.schema_attributes
        self.rules = self.generator.parser.rules

        # Initialize NegativeSampler
        self.negative_sampler = NegativeSampler(
            schema_classes=self.schema_classes,
            schema_relations=self.schema_relations,
            cfg=cfg,
            domains=self.generator.parser.domains,
            ranges=self.generator.parser.ranges,
            verbose=verbose,
        )

        # Initialize Validator
        self.validator = Validator(
            constraints=self.generator.parser.constraints,
            domains=self.generator.parser.domains,
            ranges=self.generator.parser.ranges,
            verbose=verbose,
        )

        # Negative sampling config
        self.neg_strategy = cfg.negative_sampling.strategy
        self.neg_ratio = cfg.negative_sampling.ratio
        self.neg_corrupt_base_facts = cfg.negative_sampling.corrupt_base_facts

        # Track rule usage for coverage analysis
        self.train_rule_usage: Dict[str, int] = defaultdict(int)
        self.test_rule_usage: Dict[str, int] = defaultdict(int)

        # Validation stats
        self.validation_errors: Dict[str, List[str]] = defaultdict(list)
        self.discarded_samples: Dict[str, List[str]] = defaultdict(list)

        # Detailed tracking for debugging
        self.rule_selection_count: Dict[str, int] = defaultdict(int)
        self.rule_success_count: Dict[str, int] = defaultdict(int)

        if self.verbose:
            logger.info(f"Loaded {len(self.rules)} rules from ontology")
            logger.info(
                f"Schema: {len(self.schema_classes)} classes, "
                f"{len(self.schema_relations)} relations, "
                f"{len(self.schema_attributes)} attributes"
            )
            logger.info(f"Constraints: {len(self.generator.parser.constraints)}")

    def generate_dataset(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        min_individuals: int,
        max_individuals: int,
        min_rules: int = 1,
        max_rules: int = 5,
        min_proofs_per_rule: int = 5,
    ) -> Tuple[List[KnowledgeGraph], List[KnowledgeGraph], List[KnowledgeGraph]]:
        """
        Generate complete training and testing datasets.

        Args:
            n_train:                Number of training samples
            n_test:                 Number of test samples
            min_individuals:        Minimum individuals per sample
            max_individuals:        Maximum individuals per sample
            min_rules:              Min rules to trigger per sample
            max_rules:              Max rules to trigger per sample
            min_proofs_per_rule:    Minimum proofs to select per rule

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        logger.info("GENERATING KGE DATASET")
        # logger.info(f"\t{'=' * 80}")
        logger.info(f"\tTarget: {n_train} train, {n_val} val, {n_test} test samples")
        logger.info(f"\tIndividual range: {min_individuals}-{max_individuals}")
        logger.info(f"\tRules per sample: {min_rules}-{max_rules}")
        logger.info(f"\tMin proofs per rule: {min_proofs_per_rule}")
        # logger.info(f"\t{'=' * 80}")

        # Generate training samples
        # logger.info("Generating training samples...")
        train_samples = self._generate_samples(
            n_samples=n_train,
            min_individuals=min_individuals,
            max_individuals=max_individuals,
            min_rules=min_rules,
            max_rules=max_rules,
            min_proofs_per_rule=min_proofs_per_rule,
            sample_type="TRAIN",
        )

        # Generate validation samples
        # logger.info("Generating validation samples...")
        val_samples = self._generate_samples(
            n_samples=n_val,
            min_individuals=min_individuals,
            max_individuals=max_individuals,
            min_rules=min_rules,
            max_rules=max_rules,
            min_proofs_per_rule=min_proofs_per_rule,
            sample_type="VAL",
        )

        # Generate test samples (independent)
        # logger.info("Generating test samples...")
        test_samples = self._generate_samples(
            n_samples=n_test,
            min_individuals=min_individuals,
            max_individuals=max_individuals,
            min_rules=min_rules,
            max_rules=max_rules,
            min_proofs_per_rule=min_proofs_per_rule,
            sample_type="TEST",
        )

        # Print summary
        self._print_dataset_summary(train_samples, val_samples, test_samples)

        # Print summary
        self._print_dataset_summary(train_samples, val_samples, test_samples)

        # Save in Standard Format
        save_standard_dataset(train_samples, os.path.join(self.output_dir, "train"))
        save_standard_dataset(val_samples, os.path.join(self.output_dir, "val"))
        save_standard_dataset(test_samples, os.path.join(self.output_dir, "test"))

        return train_samples, val_samples, test_samples

    def _generate_samples(
        self,
        n_samples: int,
        min_individuals: int,
        max_individuals: int,
        min_rules: int,
        max_rules: int,
        min_proofs_per_rule: int,
        sample_type: str,
    ) -> List[KnowledgeGraph]:
        """
        Generate a list of independent knowledge graph samples.

        Args:
            n_samples:              Number of samples to generate
            min_individuals:        Minimum individuals per sample
            max_individuals:        Maximum individuals per sample
            min_rules:              Minimum rules to trigger per sample
            max_rules:              Maximum rules to trigger per sample
            min_proofs_per_rule:    Minimum proofs to select per rule
            stratified_selection:   Whether to use stratified rule selection
            sample_type:            "TRAIN", "VAL", or "TEST" (for logging)

        Returns:
            List of generated KG samples
        """
        samples = []
        failed_attempts = 0
        max_failed_attempts = n_samples * 10  # TODO: tune

        print("=" * 80)
        pbar = tqdm(total=n_samples, desc=f"Generating {sample_type} samples")

        while len(samples) < n_samples and failed_attempts < max_failed_attempts:
            if sample_type == "TEST":
                prefix = "test_"
            elif sample_type == "VAL":
                prefix = "val_"
            else:
                prefix = "train_"

            # Reset individual pool for each sample
            self.generator.chainer.reset_individual_pool(name_prefix=prefix)

            # Generate one, individual, knowledge graph sample
            sample = self._generate_one_sample(
                min_individuals=min_individuals,
                max_individuals=max_individuals,
                min_rules=min_rules,
                max_rules=max_rules,
                min_proofs_per_rule=min_proofs_per_rule,
                sample_type=sample_type,
            )

            if sample is not None:
                # Validate sample
                val_result = self.validator.validate(sample)
                if not val_result["valid"]:
                    if self.verbose:
                        logger.warning(f"  [WARN] Sample validation failed: {val_result['errors']}")

                    # DISCARD SAMPLE
                    # Store errors for summary (keyed by unique ID)
                    discard_id = f"{sample_type}_discarded_{failed_attempts}"
                    self.discarded_samples[discard_id] = val_result["errors"]
                    failed_attempts += 1
                    # Do not add to samples, do not update pbar
                else:
                    samples.append(sample)
                    pbar.update(1)
            else:
                failed_attempts += 1

        pbar.close()

        if len(samples) < n_samples:
            logger.warning(
                f"Warning: Only generated {len(samples)}/{n_samples} samples after {failed_attempts} failed attempts"
            )

        return samples

    def _generate_one_sample(
        self,
        min_individuals: int,
        max_individuals: int,
        min_rules: int,
        max_rules: int,
        min_proofs_per_rule: int,
        sample_type: str,
    ) -> Optional[KnowledgeGraph]:
        """
        Generate one complete, independent knowledge graph sample.

        Args:
            min_individuals:    Minimum individuals required
            max_individuals:    Maximum individuals allowed
            min_rules:          Minimum rules to trigger
            max_rules:          Maximum rules to trigger
            sample_type:        "TRAIN", "VAL", or "TEST"

        Returns:
            Generated KG sample, or None if generation failed
        """
        if not self.rules:
            return None

        # We use different variance strategies between each KG (sample)
        # to ensure diversity across the dataset.

        # VARIANCE STRATEGY 1: Vary recursion depth
        current_recursion = random.randint(1, self.max_recursion_cap)
        self.generator.chainer.max_recursion_depth = current_recursion

        # VARIANCE STRATEGY 2: Rule selection
        n_rules = random.randint(min_rules, min(max_rules, len(self.rules)))
        selected_rules = random.sample(self.rules, n_rules)

        # Track rule usage
        if sample_type == "TRAIN":
            rule_usage = self.train_rule_usage
        elif sample_type == "TEST":
            rule_usage = self.test_rule_usage
        else:
            # TODO: track VAL separately if needed
            rule_usage = defaultdict(int)

        for rule in selected_rules:
            if sample_type in ["TRAIN", "TEST"]:
                # Update TRAIN/TEST rule usage
                rule_usage[rule.name] += 1
            # Update overall rule usage
            self.rule_selection_count[rule.name] += 1

        # Generate proofs and build proof map
        sample_proof_map = defaultdict(list)  # 'sample' = one KG
        atoms_found = False

        for rule in selected_rules:
            # Generate proofs with Instance Looping for volume
            # Randomly decide how many "instances" (chains) to generate for this rule
            # e.g., generate 5-15 diverse root facts (Fathers) for this rule
            # This range provides a good balance of content diversity
            n_proof_roots_for_rule = random.randint(self.min_proof_roots, self.max_proof_roots)

            proofs = self.generator.generate_proofs_for_rule(
                rule.name, n_proof_roots=n_proof_roots_for_rule, max_proofs=None
            )

            if not proofs:
                continue

            self.rule_success_count[rule.name] += 1

            # Select random subset of proofs if still too many
            # We enforce a cap to avoid memory issues with extremely prolific rules
            MAX_PROOFS_CAP = 10000
            n_select = random.randint(min(len(proofs), min_proofs_per_rule), min(len(proofs), MAX_PROOFS_CAP))
            selected = random.sample(proofs, n_select)

            for proof in selected:
                extracted_map = extract_proof_map(proof)
                for atom, proof_list in extracted_map.items():
                    sample_proof_map[atom].extend(proof_list)
                atoms_found = True

        if not atoms_found:
            if self.verbose:
                logger.debug(f"  [DEBUG] Sample rejected: No atoms found for rules {[r.name for r in selected_rules]}")
            return None

        # Convert to KG
        kg = atoms_to_knowledge_graph(
            atoms=set(sample_proof_map.keys()),
            schema_classes=self.schema_classes,
            schema_relations=self.schema_relations,
            schema_attributes=self.schema_attributes,
            proof_map=sample_proof_map,
        )

        # Validate size
        if not (min_individuals <= len(kg.individuals) <= max_individuals):
            if self.verbose:
                logger.debug(
                    f"  [DEBUG] Sample rejected: Size mismatch. Got {len(kg.individuals)} individuals (min: {min_individuals}, max: {max_individuals})"
                )
            return None

        # Complete schema (apply domain/range rules)
        self._complete_schema(kg)

        # Add negatives via NegativeSampler
        kg = self.negative_sampler.add_negative_samples(
            kg,
            strategy=self.neg_strategy,
            ratio=self.neg_ratio,
            corrupt_base_facts=self.neg_corrupt_base_facts,
            export_proofs=self.export_proofs,
            output_dir=self.proof_output_dir,
        )

        return kg

    def _complete_schema(self, kg: KnowledgeGraph) -> None:
        """
        Deterministically apply schema rules (domain/range) to ensure consistency.

        This fixes "Schema Violation" errors where a triple exists but the
        implied class memberships for subject/object are missing.
        """
        from synthology.data_structures import Membership

        # We need to look up Class objects by name
        class_map = self.schema_classes

        # Helper to add membership if missing
        def add_membership(ind, cls_name):
            if cls_name not in class_map:
                return  # Should not happen if schema is consistent

            cls = class_map[cls_name]

            # Check if already exists
            for m in kg.memberships:
                if m.individual == ind and m.cls == cls and m.is_member:
                    return

            # Add new membership
            # Note: We treat this as an inferred fact (no specific proof stored here for simplicity,
            # or we could create a dummy proof)
            new_mem = Membership(individual=ind, cls=cls, is_member=True, proofs=[])
            kg.memberships.append(new_mem)

        # Iteratively apply rules until fixpoint
        # Ensures transitive closure of domain/range implications
        domains = self.generator.parser.domains
        ranges = self.generator.parser.ranges

        while True:
            added_count = 0
            current_mem_len = len(kg.memberships)

            for t in kg.triples:
                if not t.positive:
                    continue

                prop_name = t.predicate.name

                # Apply Domain
                if prop_name in domains:
                    for domain_cls in domains[prop_name]:
                        add_membership(t.subject, domain_cls)
                        added_count += 1

                # Apply Range
                if prop_name in ranges:
                    for range_cls in ranges[prop_name]:
                        add_membership(t.object, range_cls)
                        added_count += 1

            if len(kg.memberships) == current_mem_len:
                break  # No changes, fixpoint reached

    @staticmethod
    def check_structural_isomorphism(kg1: KnowledgeGraph, kg2: KnowledgeGraph) -> bool:
        """
        Check if two KGs are structurally isomorphic.

        Ignores individual names but preserves:
        - Graph topology (relations)
        - Class memberships (node attributes)
        - Attribute values (node attributes)

        Args:
            kg1: First knowledge graph
            kg2: Second knowledge graph

        Returns:
            True if structurally isomorphic, False otherwise
        """

        def to_nx(kg):
            G = nx.MultiDiGraph()

            # Nodes with attributes
            for ind in kg.individuals:
                clss = frozenset([m.cls.name for m in kg.memberships if m.individual == ind and m.is_member])
                attrs = tuple(
                    sorted([(at.predicate.name, str(at.value)) for at in kg.attribute_triples if at.subject == ind])
                )
                G.add_node(ind.name, classes=clss, attrs=attrs)

            # Edges (relations)
            for t in kg.triples:
                if t.positive:
                    G.add_edge(t.subject.name, t.object.name, label=t.predicate.name)

            return G

        G1 = to_nx(kg1)
        G2 = to_nx(kg2)

        nm = nx.algorithms.isomorphism.categorical_node_match(["classes", "attrs"], [frozenset(), tuple()])
        em = nx.algorithms.isomorphism.categorical_edge_match("label", None)

        return nx.is_isomorphic(G1, G2, node_match=nm, edge_match=em)

    def _print_dataset_summary(
        self,
        train_samples: List[KnowledgeGraph],
        val_samples: List[KnowledgeGraph],
        test_samples: List[KnowledgeGraph],
    ) -> None:
        """Print summary statistics for generated datasets."""

        # Calculate stats
        def get_stats(samples):
            stats = {
                "n_samples": len(samples),
                "n_individuals": 0,
                "n_triples": 0,
                "n_pos_triples": 0,
                "n_neg_triples": 0,
                "n_memberships": 0,
                "n_pos_mems": 0,
                "n_neg_mems": 0,
                "n_base_facts": 0,
                "n_inferred_facts": 0,
                "n_neg_base_source": 0,
                "n_neg_inferred_shallow": 0,
                "n_neg_inferred_deep": 0,
                "n_neg_unknown": 0,
            }
            for kg in samples:
                stats["n_individuals"] += len(kg.individuals)
                stats["n_triples"] += len(kg.triples)
                stats["n_memberships"] += len(kg.memberships)

                for t in kg.triples:
                    if t.positive:
                        stats["n_pos_triples"] += 1
                        if t.is_base_fact:
                            stats["n_base_facts"] += 1
                        else:
                            stats["n_inferred_facts"] += 1
                    else:
                        stats["n_neg_triples"] += 1
                        source = t.metadata.get("source_type", "unknown")
                        if source == "base":
                            stats["n_neg_base_source"] += 1
                        elif source == "inferred":
                            stats["n_neg_inferred_shallow"] += 1
                        elif source == "propagated_inferred":
                            stats["n_neg_inferred_deep"] += 1
                        else:
                            stats["n_neg_unknown"] += 1

                for m in kg.memberships:
                    if m.is_member:
                        stats["n_pos_mems"] += 1
                        if m.is_base_fact:
                            stats["n_base_facts"] += 1
                        else:
                            stats["n_inferred_facts"] += 1
                    else:
                        stats["n_neg_mems"] += 1
                        source = m.metadata.get("source_type", "unknown")
                        if source == "base":
                            stats["n_neg_base_source"] += 1
                        elif source == "inferred":
                            stats["n_neg_inferred_shallow"] += 1
                        elif source == "propagated_inferred":
                            stats["n_neg_inferred_deep"] += 1
                        else:
                            stats["n_neg_unknown"] += 1

            # Averages
            if stats["n_samples"] > 0:
                for key in list(stats.keys()):
                    if key != "n_samples":
                        # Remove "n_" prefix for avg key
                        avg_key_name = key[2:]
                        stats[f"avg_{avg_key_name}"] = stats[key] / stats["n_samples"]
            return stats

        train_stats = get_stats(train_samples)
        val_stats = get_stats(val_samples)
        test_stats = get_stats(test_samples)

        logger.success("Train/val/test dataset generation complete.")

        # Check isomorphism
        if len(train_samples) * len(test_samples) < 1000:
            logger.info("Checking structural isomorphism (this may be slow)...")
            isomorphic_count = 0
            # Check Train vs Test
            for train_kg in train_samples:
                for test_kg in test_samples:
                    if self.check_structural_isomorphism(train_kg, test_kg):
                        isomorphic_count += 1
                        break

            if isomorphic_count > 0:
                logger.warning(f"Warning: Found {isomorphic_count} isomorphic samples between train/test")
            else:
                logger.info("No structural isomorphism between train and test")

        else:
            logger.info("Skipping isomorphism check due to large dataset size.")

        # Rule coverage
        logger.info("=" * 80)
        logger.info("RULE COVERAGES")
        logger.info("=" * 80)

        def log_rule_usage(title, usage_dict, total_selections):
            logger.info(f"{title}:")
            logger.info(f"{'Rule Name':<60} | {'Count':<10} | {'Percentage':<10}")
            logger.info("-" * 90)
            for rule_name, count in sorted(usage_dict.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_selections) * 100 if total_selections > 0 else 0
                logger.info(f"{rule_name:<60} | {count:<10} | {percentage:.1f}%")

        total_train_selections = sum(self.train_rule_usage.values())
        log_rule_usage("TRAIN rule usage", self.train_rule_usage, total_train_selections)

        total_test_selections = sum(self.test_rule_usage.values())
        log_rule_usage("TEST rule usage", self.test_rule_usage, total_test_selections)

        unused_in_train = set(r.name for r in self.rules) - set(self.train_rule_usage.keys())
        unused_in_test = set(r.name for r in self.rules) - set(self.test_rule_usage.keys())

        if unused_in_train:
            logger.warning(f"\nWarning: {len(unused_in_train)} rules unused in TRAINING")
            for r in sorted(unused_in_train):
                logger.warning(f"  - {r}")
        if unused_in_test:
            logger.warning(f"Warning: {len(unused_in_test)} rules unused in TESTING")
            for r in sorted(unused_in_test):
                logger.warning(f"  - {r}")

        # Detailed breakdown of unused rules
        all_unused = unused_in_train.union(unused_in_test)
        if all_unused:
            logger.info("Unused Rules Analysis:")
            logger.info(f"{'Rule Name':<60} | {'Reason':<20} | {'Attempts':<10}")
            logger.info("-" * 100)
            for rule_name in sorted(all_unused):
                attempts = self.rule_selection_count[rule_name]
                if attempts == 0:
                    reason = "Never Selected"
                else:
                    reason = "Failed to Generate"
                logger.info(f"{rule_name:<60} | {reason:<20} | {attempts:<10}")

        # Negative Strategy Usage
        strategy_usage = self.negative_sampler.strategy_usage
        if strategy_usage:
            logger.info("=" * 80)
            logger.info("NEGATIVE SAMPLING STRATEGY USAGE")
            logger.info("=" * 80)
            logger.info(f"{'Strategy':<20} | {'Count':<10} | {'Percentage':<10}")
            logger.info("-" * 50)
            total_negatives = sum(strategy_usage.values())
            for strategy, count in sorted(strategy_usage.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_negatives) * 100 if total_negatives > 0 else 0
                logger.info(f"{strategy:<20} | {count:<10} | {percentage:.1f}%")

        # Validation Report
        logger.info("=" * 40)
        logger.info("DATA VALIDATION REPORT")
        logger.info("=" * 40)
        if not self.discarded_samples:
            logger.success("All generated samples passed validation checks (0 discarded).")
        else:
            # Group errors by type for cleaner reporting
            error_counts = defaultdict(int)
            total_errors = 0
            for errors in self.discarded_samples.values():
                total_errors += len(errors)
                if errors:
                    # To ensure the counts add up to the total number of discarded samples,
                    # we attribute each discarded sample to its *first* validation error type.
                    first_error = errors[0]
                    base_err = first_error.split(":")[0] if ":" in first_error else first_error
                    error_counts[base_err] += 1

            logger.warning(f"Discarded {len(self.discarded_samples)} samples:")

            for err_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                logger.warning(f"  - {err_type}: {count}")

            # logger.debug("\nDetailed Discard Reasons (First 5):")
            # for i, (sample_id, errors) in enumerate(self.discarded_samples.items()):
            #     if i >= 5:
            #         break
            #     logger.debug(f"  {sample_id}: {errors}")

        print(f"\n{'=' * 80}")
        print("DATASET STATISTICS")
        print(f"{'=' * 80}")

        def print_split_stats(name: str, stats: dict):
            print(f"\n{name} SET:")
            print(f"  Samples:           {stats.get('n_samples', 0)}")
            print(f"  Avg individuals:   {stats.get('avg_individuals', 0):.1f}")
            print(f"  Avg triples:       {stats.get('avg_triples', 0):.1f}")
            print(f"    - Positive:      {stats.get('avg_pos_triples', 0):.1f}")
            print(f"    - Negative:      {stats.get('avg_neg_triples', 0):.1f}")
            print(f"  Avg memberships:   {stats.get('avg_memberships', 0):.1f}")
            print(f"    - Positive:      {stats.get('avg_pos_mems', 0):.1f}")
            print(f"    - Negative:      {stats.get('avg_neg_mems', 0):.1f}")
            print("  Fact Types (Pos):")
            print(f"    - Base Facts:    {stats.get('avg_base_facts', 0):.1f}")
            print(f"    - Inferred:      {stats.get('avg_inferred_facts', 0):.1f}")
            print("  Negative Logic (Avg per sample):")
            print(f"    - Base Fact Corruption:      {stats.get('avg_neg_base_source', 0):.1f} (Direct corruption of base facts)")
            print(f"    - Inferred Fact (Shallow):   {stats.get('avg_neg_inferred_shallow', 0):.1f} (Direct corruption of inferred facts)")
            print(f"    - Inferred Fact (Deep/Prop): {stats.get('avg_neg_inferred_deep', 0):.1f} (Propagated consequences)")
            print(f"    - Unknown/Other:             {stats.get('avg_neg_unknown', 0):.1f}")

        print_split_stats("TRAINING", train_stats)
        print_split_stats("VAL", val_stats)
        print_split_stats("TEST", test_stats)

        print(f"{'=' * 80}\n")


# ============================================================================ #
#                         CSV SERIALIZATION METHODS                            #
# ============================================================================ #


def save_dataset_to_csv(
    samples: List[KnowledgeGraph],
    output_dir: str,
    prefix: str = "sample",
) -> None:
    """
    Save dataset to CSV files.

    Each sample saved as separate CSV with format:
        subject, predicate, object, label, fact_type

    Args:
        samples: List of KG samples
        output_dir: Directory to save files
        prefix: Prefix for file names
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(samples)} samples to {output_dir}")

    for idx, kg in enumerate(samples):
        file_path = output_path / f"{prefix}_{idx:05d}.csv"
        kg.to_csv(str(file_path))

        if (idx + 1) % 100 == 0 or (idx + 1) == len(samples):
            logger.info(f"\tSaved {idx + 1}/{len(samples)}")

    logger.success(f"Dataset CSVs saved to {output_dir}")



def save_standard_dataset(samples: List[KnowledgeGraph], output_base_dir: str) -> None:
    """
    Saves the dataset in the Standard Format:
    - facts.csv: Base facts (positives, hops=0)
    - targets.csv: Everything else (queries, negatives, inferred)

    Args:
        samples (List[KnowledgeGraph]): List of samples to save.
        output_base_dir (str): output directory (e.g. data/train).
    """
    path = Path(output_base_dir)
    path.mkdir(parents=True, exist_ok=True)

    facts_rows = []
    targets_rows = []

    for idx, kg in enumerate(samples):
        # Sample ID is 1000 + idx to avoid 0-indexing issues if any
        sample_id = str(1000 + idx)
        
        all_rows = kg.to_standard_rows(sample_id)
        
        for row in all_rows:
            # FACTS: Only Positive Base Facts
            if row["type"] == "base_fact" and row["label"] == 1:
                # Minimal columns for facts.csv: sample_id, subject, predicate, object
                facts_rows.append({
                    "sample_id": row["sample_id"],
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"]
                })
                # Base facts are ALSO targets (trivial logic)
                targets_rows.append(row)
            else:
                # Everything else is a target
                targets_rows.append(row)

    # Write facts.csv
    facts_path = path / "facts.csv"
    if facts_rows:
        with open(facts_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "subject", "predicate", "object"])
            writer.writeheader()
            writer.writerows(facts_rows)
    
    # Write targets.csv
    targets_path = path / "targets.csv"
    if targets_rows:
         # dynamic fieldnames based on row keys
        keys = list(targets_rows[0].keys())
        with open(targets_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(targets_rows)

    logger.success(f"Saved standard dataset to {output_base_dir} ({len(facts_rows)} facts, {len(targets_rows)} targets)")


def load_dataset_from_csv(
    input_dir: str,
    prefix: str = "sample",
    n_samples: Optional[int] = None,
) -> List[KnowledgeGraph]:
    """
    Load dataset from CSV files.

    Args:
        input_dir: Directory containing CSV files
        prefix: Prefix of files to load
        n_samples: Max samples to load (None = all)

    Returns:
        List of loaded KG samples
    """
    input_path = Path(input_dir)
    pattern = f"{prefix}_*.csv"
    csv_files = sorted(input_path.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {input_dir}/{pattern}")

    if n_samples is not None:
        csv_files = csv_files[:n_samples]

    logger.info(f"Loading {len(csv_files)} samples from {input_dir}/")

    samples = []
    for idx, file_path in enumerate(csv_files):
        kg = KnowledgeGraph.from_csv(str(file_path))
        samples.append(kg)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(csv_files):
            logger.info(f"  Loaded {idx + 1}/{len(csv_files)}")

    logger.info(f"Dataset loaded from {input_dir}/")
    return samples


# ============================================================================ #
#                              MAIN ENTRY POINT                                #
# ============================================================================ #


REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/ont_generator", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for dataset generation."""

    # Add file sink for logging
    output_dir = cfg.dataset.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "generation.log")
    logger.add(log_path, mode="w", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

    logger.info(f"Running Ontology Knowledge Graph Generator with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize generator
    generator = KGEDatasetGenerator(
        cfg=cfg,
        verbose=(cfg.logging.level == "DEBUG"),
    )

    # Generate datasets
    train_samples, val_samples, test_samples = generator.generate_dataset(
        n_train=cfg.dataset.n_train,
        n_val=cfg.dataset.n_val,
        n_test=cfg.dataset.n_test,
        min_individuals=cfg.dataset.min_individuals,
        max_individuals=cfg.dataset.max_individuals,
        min_rules=cfg.dataset.min_rules,
        max_rules=cfg.dataset.max_rules,
        min_proofs_per_rule=cfg.dataset.min_proofs_per_rule,
    )

    # Save to CSV (Individual Samples) - Optional
    output_dir = cfg.dataset.output_dir

    if cfg.dataset.get("save_individual_samples", False):
        logger.info("Saving individual sample CSVs...")
        save_dataset_to_csv(train_samples, f"{output_dir}/train", prefix="train_sample")
        save_dataset_to_csv(val_samples, f"{output_dir}/val", prefix="val_sample")
        save_dataset_to_csv(test_samples, f"{output_dir}/test", prefix="test_sample")



    logger.success("Ontology-based Knowledge Graph Dataset generation complete!")

    # Export Graphs (a few samples)
    if cfg.export_graphs:
        logger.info("Exporting all graph visualizations...")
        graph_dir = os.path.join(output_dir, "graphs")
        os.makedirs(graph_dir, exist_ok=True)

        # Limit the number of visualizations to avoid excessive output
        max_visualizations_train = min(3, len(train_samples))
        max_visualizations_test = min(3, len(test_samples))

        subset_train_samples = random.sample(train_samples, max_visualizations_train)
        subset_test_samples = random.sample(test_samples, max_visualizations_test)

        for i, sample in enumerate(subset_train_samples):
            sample.save_visualization(
                output_path=graph_dir,
                output_name=f"train_sample_{i}",
                title=f"Train Sample {i}",
            )

        for i, sample in enumerate(subset_test_samples):
            sample.save_visualization(
                output_path=graph_dir,
                output_name=f"test_sample_{i}",
                title=f"Test Sample {i}",
            )


if __name__ == "__main__":
    main()
