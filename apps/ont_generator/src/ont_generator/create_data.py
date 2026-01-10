"""
DESCRIPTION

    KGE Model Train/Test Data Generator

    Generates independent knowledge graph samples for KGE model training.
    Each sample is a complete KG with unique individuals, base facts,
    derived inferences, and balanced positive/negative examples.

AUTHOR

    Vincent Van Schependom
"""

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

from synthology.data_structures import KnowledgeGraph

from .generate import KGenerator, atoms_to_knowledge_graph, extract_proof_map
from .negative_sampler import NegativeSampler
from .utils.validator import Validator


class KGEDatasetGenerator:
    """
    Generates training and testing datasets for KGE models.

    Orchestrates:
    - Sample generation via KGenerator
    - Negative sampling via NegativeSampler
    - Train/test splitting
    - Validation and export
    """

    def __init__(
        self,
        cfg: DictConfig,
        verbose: bool,
    ):
        """
        Initialize dataset generator.

        Args:
            cfg: Hydra configuration object
            verbose: Enable detailed logging
        """
        if cfg.dataset.seed is not None:
            random.seed(cfg.dataset.seed)

        self.verbose = verbose
        # Access safely checks
        self.max_recursion_cap = cfg.generator.max_recursion
        self.individual_pool_size = cfg.generator.individual_pool_size
        self.individual_reuse_prob = cfg.generator.individual_reuse_prob
        self.export_proofs = cfg.dataset.get("export_proofs", False)
        self.output_dir = cfg.dataset.output_dir

        # Initialize KGenerator
        self.generator = KGenerator(
            cfg=cfg,
            verbose=False,  # Keep generator quiet during batch generation
        )

        # Store schema references
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
            n_train: Number of training samples
            n_test: Number of test samples
            min_individuals: Minimum individuals per sample
            max_individuals: Maximum individuals per sample
            min_rules: Min rules to trigger per sample
            max_rules: Max rules to trigger per sample
            min_proofs_per_rule: Minimum proofs to select per rule

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        logger.info(f"\n{'=' * 80}")
        logger.info("GENERATING KGE DATASET")
        logger.info(f"{'=' * 80}")
        logger.info(f"Target: {n_train} train, {n_val} val, {n_test} test samples")
        logger.info(f"Individual range: {min_individuals}-{max_individuals}")
        logger.info(f"Rules per sample: {min_rules}-{max_rules}")
        logger.info(f"Min proofs per rule: {min_proofs_per_rule}")
        logger.info(f"{'=' * 80}")

        # Generate training samples
        logger.info("Generating training samples...")
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
        logger.info("Generating validation samples...")
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
        logger.info("Generating test samples...")
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
            n_samples: Number of samples to generate
            min_individuals: Min individuals per sample
            max_individuals: Max individuals per sample
            min_rules: Min rules to trigger per sample
            max_rules: Max rules to trigger per sample
            min_proofs_per_rule: Minimum proofs to select per rule
            stratified_selection: Whether to use stratified rule selection
            sample_type: "TRAIN" or "TEST" (for logging)

        Returns:
            List of generated KG samples
        """
        samples = []
        failed_attempts = 0
        max_failed_attempts = n_samples * 10

        pbar = tqdm(total=n_samples, desc=f"Generating {sample_type} samples")

        while len(samples) < n_samples and failed_attempts < max_failed_attempts:
            # Reset individual pool for each sample
            # INDUCTIVE SPLIT: Use different individual name prefix for Test set
            # For Val, we can use "val_" or stick to "train_" if we consider it part of training distribution visually?
            # It's better to be clean: "val_"
            if sample_type == "TEST":
                prefix = "test_"
            elif sample_type == "VAL":
                prefix = "val_"
            else:
                prefix = "train_"
            self.generator.chainer.reset_individual_pool(name_prefix=prefix)

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

        Strategy:
        1. Randomly vary recursion depth (structural diversity)
        2. Randomly select subset of rules (content diversity)
        3. Generate proofs for selected rules
        4. Convert to KG
        5. Add negative samples via NegativeSampler

        Args:
            min_individuals: Minimum individuals required
            max_individuals: Maximum individuals allowed
            min_rules: Minimum rules to trigger
            max_rules: Maximum rules to trigger
            sample_type: "TRAIN" or "TEST"

        Returns:
            Generated KG sample, or None if generation failed
        """
        if not self.rules:
            return None

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
            # Just don't track or track separately? Let's just not crash.
            # We only log train/test usage explicitly in summary.
            # Let's use a dummy dict or similar if we wanted, but for now just skip modification if not train/test.
            rule_usage = defaultdict(int)

        for rule in selected_rules:
            if sample_type in ["TRAIN", "TEST"]:
                rule_usage[rule.name] += 1
            self.rule_selection_count[rule.name] += 1

        # Generate proofs and build proof map
        sample_proof_map = defaultdict(list)
        atoms_found = False

        for rule in selected_rules:
            # Generate proofs with Instance Looping for volume
            # Randomly decide how many "instances" (chains) to generate for this rule
            # e.g., generate 5-15 diverse root facts (Fathers) for this rule
            # This range provides a good balance of content diversity
            n_instances_for_rule = random.randint(5, 15)

            proofs = self.generator.generate_proofs_for_rule(
                rule.name, n_instances=n_instances_for_rule, max_proofs=None
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
            output_dir=self.output_dir,
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
        # (Simple version: just one pass is enough for domain/range
        # unless we have complex chains which we don't handle here yet)

        domains = self.generator.parser.domains
        ranges = self.generator.parser.ranges

        for t in kg.triples:
            if not t.positive:
                continue

            prop_name = t.predicate.name

            # Apply Domain
            if prop_name in domains:
                for domain_cls in domains[prop_name]:
                    add_membership(t.subject, domain_cls)

            # Apply Range
            if prop_name in ranges:
                for range_cls in ranges[prop_name]:
                    add_membership(t.object, range_cls)

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
                "n_neg_base_facts": 0,
                "n_neg_inferred_facts": 0,
                "n_neg_propagated_facts": 0,
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
                            stats["n_neg_base_facts"] += 1
                        elif source == "inferred":
                            stats["n_neg_inferred_facts"] += 1
                        elif source == "propagated_inferred":
                            stats["n_neg_propagated_facts"] += 1

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
                            stats["n_neg_base_facts"] += 1
                        elif source == "inferred":
                            stats["n_neg_inferred_facts"] += 1

            # Averages
            if stats["n_samples"] > 0:
                for key in list(stats.keys()):
                    if key != "n_samples":
                        stats[f"avg_{key[2:]}"] = stats[key] / stats["n_samples"]
            return stats

        train_stats = get_stats(train_samples)
        val_stats = get_stats(val_samples)
        test_stats = get_stats(test_samples)

        logger.info(f"\n{'=' * 80}")
        logger.info("DATASET GENERATION COMPLETE")
        logger.info(f"{'=' * 80}")

        # Check isomorphism
        # Check isomorphism
        logger.info("Checking structural isomorphism...")
        isomorphic_count = 0
        # Check Train vs Test
        for train_kg in train_samples:
            for test_kg in test_samples:
                if self.check_structural_isomorphism(train_kg, test_kg):
                    isomorphic_count += 1
                    break

        # Check Val vs Test (if needed, but usually train vs test is key)

        if isomorphic_count > 0:
            logger.warning(f"Warning: Found {isomorphic_count} isomorphic samples between train/test")
        else:
            logger.info("No structural isomorphism between train and test")

        # Rule coverage
        logger.info("\n--- Rule Coverage ---")

        def log_rule_usage(title, usage_dict, total_selections):
            logger.info(f"\n{title}:")
            logger.info(f"{'Rule Name':<40} | {'Count':<10} | {'Percentage':<10}")
            logger.info("-" * 70)
            for rule_name, count in sorted(usage_dict.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_selections) * 100 if total_selections > 0 else 0
                logger.info(f"{rule_name:<40} | {count:<10} | {percentage:.1f}%")

        total_train_selections = sum(self.train_rule_usage.values())
        log_rule_usage("Train Rule Usage", self.train_rule_usage, total_train_selections)

        total_test_selections = sum(self.test_rule_usage.values())
        log_rule_usage("Test Rule Usage", self.test_rule_usage, total_test_selections)

        unused_in_train = set(r.name for r in self.rules) - set(self.train_rule_usage.keys())
        unused_in_test = set(r.name for r in self.rules) - set(self.test_rule_usage.keys())

        if unused_in_train:
            logger.warning(f"\nWarning: {len(unused_in_train)} rules unused in training")
            for r in sorted(unused_in_train):
                logger.warning(f"  - {r}")
        if unused_in_test:
            logger.warning(f"Warning: {len(unused_in_test)} rules unused in testing")
            for r in sorted(unused_in_test):
                logger.warning(f"  - {r}")

        # Detailed breakdown of unused rules
        all_unused = unused_in_train.intersection(unused_in_test)
        if all_unused:
            logger.info("\n--- Unused Rules Analysis ---")
            logger.info(f"{'Rule Name':<40} | {'Reason':<20} | {'Attempts':<10}")
            logger.info("-" * 80)
            for rule_name in sorted(all_unused):
                attempts = self.rule_selection_count[rule_name]
                if attempts == 0:
                    reason = "Never Selected"
                else:
                    reason = "Failed to Generate"
                logger.info(f"{rule_name:<40} | {reason:<20} | {attempts:<10}")

        # Negative Strategy Usage
        strategy_usage = self.negative_sampler.strategy_usage
        if strategy_usage:
            logger.info("\n--- Negative Strategy Usage ---")
            logger.info(f"{'Strategy':<20} | {'Count':<10} | {'Percentage':<10}")
            logger.info("-" * 50)
            total_negatives = sum(strategy_usage.values())
            for strategy, count in sorted(strategy_usage.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_negatives) * 100 if total_negatives > 0 else 0
                logger.info(f"{strategy:<20} | {count:<10} | {percentage:.1f}%")

        # Validation Report
        logger.info("\n--- Validation Report ---")
        if not self.discarded_samples:
            logger.info("All generated samples passed validation checks (0 discarded).")
        else:
            logger.warning(f"Discarded {len(self.discarded_samples)} samples due to validation errors:")

            # Group errors by type for cleaner reporting
            error_counts = defaultdict(int)
            for errors in self.discarded_samples.values():
                for err in errors:
                    # Simplify error message to group similar ones
                    # e.g., "Constraint Violation: Ind_5 ..." -> "Constraint Violation"
                    base_err = err.split(":")[0] if ":" in err else err
                    error_counts[base_err] += 1

            for err_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                logger.warning(f"  - {err_type}: {count}")

            if self.verbose:
                logger.debug("\nDetailed Discard Reasons (First 5):")
                for i, (sample_id, errors) in enumerate(self.discarded_samples.items()):
                    if i >= 5:
                        break
                    logger.debug(f"  {sample_id}: {errors}")

        print("\nTRAINING SET:")
        print(f"  Samples:           {train_stats.get('n_samples', 0)}")
        print(f"  Avg individuals:   {train_stats.get('avg_individuals', 0):.1f}")
        print(f"  Avg triples:       {train_stats.get('avg_triples', 0):.1f}")
        print(f"    - Positive:      {train_stats.get('avg_pos_triples', 0):.1f}")
        print(f"    - Negative:      {train_stats.get('avg_neg_triples', 0):.1f}")
        print(f"  Avg memberships:   {train_stats.get('avg_memberships', 0):.1f}")
        print(f"    - Positive:      {train_stats.get('avg_pos_mems', 0):.1f}")
        print(f"    - Negative:      {train_stats.get('avg_neg_mems', 0):.1f}")
        print("  Fact Types (Pos):")
        print(f"    - Base Facts:    {train_stats.get('avg_base_facts', 0):.1f}")
        print(f"    - Inferred:      {train_stats.get('avg_inferred_facts', 0):.1f}")
        print("  Fact Types (Neg):")
        print(f"    - Base Facts:    {train_stats.get('avg_neg_base_facts', 0):.1f}")
        print(f"    - Inferred:      {train_stats.get('avg_neg_inferred_facts', 0):.1f}")
        print(f"    - Propagated:    {train_stats.get('avg_neg_propagated_facts', 0):.1f}")

        print("\nVAL SET:")
        print(f"  Samples:           {val_stats.get('n_samples', 0)}")
        print(f"  Avg individuals:   {val_stats.get('avg_individuals', 0):.1f}")
        print(f"  Avg triples:       {val_stats.get('avg_triples', 0):.1f}")
        print(f"    - Positive:      {val_stats.get('avg_pos_triples', 0):.1f}")
        print(f"    - Negative:      {val_stats.get('avg_neg_triples', 0):.1f}")

        print("\nTEST SET:")
        print(f"  Samples:           {test_stats.get('n_samples', 0)}")
        print(f"  Avg individuals:   {test_stats.get('avg_individuals', 0):.1f}")
        print(f"  Avg triples:       {test_stats.get('avg_triples', 0):.1f}")
        print(f"    - Positive:      {test_stats.get('avg_pos_triples', 0):.1f}")
        print(f"    - Negative:      {test_stats.get('avg_neg_triples', 0):.1f}")
        print(f"  Avg memberships:   {test_stats.get('avg_memberships', 0):.1f}")
        print(f"    - Positive:      {test_stats.get('avg_pos_mems', 0):.1f}")
        print(f"    - Negative:      {test_stats.get('avg_neg_mems', 0):.1f}")
        print("  Fact Types (Pos):")
        print(f"    - Base Facts:    {test_stats.get('avg_base_facts', 0):.1f}")
        print(f"    - Inferred:      {test_stats.get('avg_inferred_facts', 0):.1f}")
        print("  Fact Types (Neg):")
        print(f"    - Base Facts:    {test_stats.get('avg_neg_base_facts', 0):.1f}")
        print(f"    - Inferred:      {test_stats.get('avg_neg_inferred_facts', 0):.1f}")
        print(f"    - Propagated:    {test_stats.get('avg_neg_propagated_facts', 0):.1f}")

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

    logger.info(f"Saving {len(samples)} samples to {output_dir}/")

    for idx, kg in enumerate(samples):
        file_path = output_path / f"{prefix}_{idx:05d}.csv"
        kg.to_csv(str(file_path))

        if (idx + 1) % 100 == 0 or (idx + 1) == len(samples):
            logger.info(f"  Saved {idx + 1}/{len(samples)}")

    logger.info(f"Dataset saved to {output_dir}/")


def save_explanations(
    samples: List[KnowledgeGraph], output_dir: str, filename: str = "negatives_explanations.csv"
) -> None:
    """
    Save explanations for negative samples to a separate CSV.
    Format: sample_id, subject, predicate, object, explanation
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename

    logger.info(f"Saving negative explanations to {file_path}")

    with open(file_path, "w", encoding="utf-8") as f:
        # Header
        f.write("sample_id,subject,predicate,object,explanation\n")

        count = 0
        for idx, kg in enumerate(samples):
            sample_id = f"sample_{idx:05d}"

            for t in kg.triples:
                if not t.positive and "explanation" in t.metadata:
                    # Escape CSV fields
                    expl = t.metadata["explanation"].replace('"', '""')
                    subj = getattr(t.subject, "name", str(t.subject))
                    pred = getattr(t.predicate, "name", str(t.predicate))
                    obj = getattr(t.object, "name", str(t.object))

                    f.write(f'{sample_id},{subj},{pred},{obj},"{expl}"\n')
                    count += 1

    logger.info(f"Saved {count} explanations.")


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


REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/ont_generator", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for dataset generation."""

    logger.info(f"Running Ontology Knowledge Graph Generator with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize generator
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

    # Save to CSV
    output_dir = cfg.dataset.output_dir
    save_dataset_to_csv(train_samples, f"{output_dir}/train", prefix="train_sample")
    save_explanations(train_samples, f"{output_dir}/train")

    save_dataset_to_csv(val_samples, f"{output_dir}/val", prefix="val_sample")
    save_explanations(val_samples, f"{output_dir}/val")

    save_dataset_to_csv(test_samples, f"{output_dir}/test", prefix="test_sample")
    save_explanations(test_samples, f"{output_dir}/test")

    logger.info("Dataset generation complete!")

    # Export Graphs (a few samples)
    if cfg.dataset.export_graphs:
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
