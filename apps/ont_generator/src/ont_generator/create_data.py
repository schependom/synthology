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
import json
import os
import random
import time
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
        self._timing = {}
        parsing_start = time.perf_counter()
        """
        Initialize dataset generator.

        Args:
            cfg:        Hydra configuration object
            verbose:    Enable detailed logging
        """
        if cfg.seed is not None:
            random.seed(cfg.seed)

        self.cfg = cfg
        self.verbose = verbose
        self.max_recursion_cap = cfg.generator.max_recursion
        self.individual_pool_size = cfg.generator.individual_pool_size
        self.individual_reuse_prob = cfg.generator.individual_reuse_prob
        self.rule_sampling_mode = str(cfg.generator.get("rule_sampling_mode", "random")).lower()
        self.enforce_goal_predicate_coverage = bool(cfg.generator.get("enforce_goal_predicate_coverage", False))
        self.proof_selection_strategy = str(cfg.generator.get("proof_selection_strategy", "random")).lower()
        weights_cfg = cfg.generator.get("proof_selection_weights", {})
        self.proof_selection_weights = {
            "easy": float(weights_cfg.get("easy", 1.0)),
            "medium": float(weights_cfg.get("medium", 1.0)),
            "hard": float(weights_cfg.get("hard", 1.0)),
        }
        if sum(max(w, 0.0) for w in self.proof_selection_weights.values()) <= 0:
            self.proof_selection_weights = {"easy": 1.0, "medium": 1.0, "hard": 1.0}
        self.export_proofs = cfg.export_proofs
        self.output_dir = cfg.dataset.output_dir
        self.proof_output_dir = (
            cfg.get("proof_output_path", os.path.join(self.output_dir, "proofs")) if self.export_proofs else None
        )

        fixed_proof_roots = cfg.generator.get("proof_roots_per_rule", None)
        if fixed_proof_roots is None:
            self.min_proof_roots = cfg.generator.get("min_proof_roots", 5)
            self.max_proof_roots = cfg.generator.get("max_proof_roots", 15)
        else:
            self.min_proof_roots = int(fixed_proof_roots)
            self.max_proof_roots = int(fixed_proof_roots)

        # --- Ontology Parsing Phase ---
        parsing_end = time.perf_counter()
        self._timing["parsing_seconds"] = parsing_end - parsing_start

        # Initialize KGenerator
        gen_start = time.perf_counter()
        self.generator = KGenerator(
            cfg=cfg,
            verbose=False,  # Keep generator quiet during batch generation
        )
        gen_end = time.perf_counter()
        self._timing["generator_init_seconds"] = gen_end - gen_start

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
            min_lcc_ratio=cfg.generator.get("min_lcc_ratio", 0.8),
        )

        # Negative sampling config
        self.neg_strategy = cfg.neg_sampling.strategy
        self.neg_ratio = cfg.neg_sampling.ratio
        self.neg_corrupt_base_facts = cfg.neg_sampling.corrupt_base_facts

        # Track rule usage for coverage analysis
        self.train_rule_usage: Dict[str, int] = defaultdict(int)
        self.test_rule_usage: Dict[str, int] = defaultdict(int)

        # Validation stats
        self.validation_errors: Dict[str, List[str]] = defaultdict(list)
        self.discarded_samples: Dict[str, List[str]] = defaultdict(list)

        # Detailed tracking for debugging
        self.rule_selection_count: Dict[str, int] = defaultdict(int)
        self.rule_success_count: Dict[str, int] = defaultdict(int)

        self.train_goal_predicates_seen = set()
        self.test_goal_predicates_seen = set()
        self.all_goal_predicates = set()
        if self.rules:
            for rule in self.rules:
                if hasattr(rule, "conclusion") and hasattr(rule.conclusion, "predicate"):
                    pred = rule.conclusion.predicate
                    self.all_goal_predicates.add(str(getattr(pred, "name", pred)))

        if self.verbose:
            logger.info(f"Loaded {len(self.rules)} rules from ontology")
            logger.info(
                f"Schema: {len(self.schema_classes)} classes, "
                f"{len(self.schema_relations)} relations, "
                f"{len(self.schema_attributes)} attributes"
            )
            logger.info(f"Constraints: {len(self.generator.parser.constraints)}")

        # Classify rules by complexity
        self.simple_rules = [r for r in self.rules if len(r.premises) == 1]
        self.complex_rules = [r for r in self.rules if len(r.premises) > 1]
        if self.verbose:
            logger.info(f"Rule Complexity: {len(self.simple_rules)} simple, {len(self.complex_rules)} complex")

    def generate_dataset(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        min_individuals: int,
        max_individuals: int,
        min_rules: int = 1,
        max_rules: int = 5,
        target_min_proofs_rule: int = 5,
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
            target_min_proofs_rule:    Minimum proofs to select per rule

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        phase_start = time.perf_counter()

        logger.info("GENERATING KGE DATASET")
        logger.info(f"\tTarget: {n_train} train, {n_val} val, {n_test} test samples")
        logger.info(f"\tIndividual range: {min_individuals}-{max_individuals}")
        logger.info(f"\tRules per sample: {min_rules}-{max_rules}")
        logger.info(f"\tMin proofs per rule: {target_min_proofs_rule}")

        # --- Generation Phase ---
        gen_samples_start = time.perf_counter()
        split_start = time.perf_counter()
        train_samples = self._generate_samples(
            n_samples=n_train,
            min_individuals=min_individuals,
            max_individuals=max_individuals,
            min_rules=min_rules,
            max_rules=max_rules,
            target_min_proofs_rule=target_min_proofs_rule,
            sample_type="TRAIN",
        )
        train_runtime = time.perf_counter() - split_start

        split_start = time.perf_counter()
        val_samples = self._generate_samples(
            n_samples=n_val,
            min_individuals=min_individuals,
            max_individuals=max_individuals,
            min_rules=min_rules,
            max_rules=max_rules,
            target_min_proofs_rule=target_min_proofs_rule,
            sample_type="VAL",
        )
        val_runtime = time.perf_counter() - split_start

        split_start = time.perf_counter()
        test_samples = self._generate_samples(
            n_samples=n_test,
            min_individuals=min_individuals,
            max_individuals=max_individuals,
            min_rules=min_rules,
            max_rules=max_rules,
            target_min_proofs_rule=target_min_proofs_rule,
            sample_type="TEST",
        )
        test_runtime = time.perf_counter() - split_start
        gen_samples_end = time.perf_counter()
        self._timing["generation_seconds"] = gen_samples_end - gen_samples_start
        self._timing["train_split_seconds"] = train_runtime
        self._timing["val_split_seconds"] = val_runtime
        self._timing["test_split_seconds"] = test_runtime

        # --- Validation/Negative Sampling Phase ---
        # (Already included in sample generation, but can be split if needed)

        # --- Balancing Phase (if any) ---
        # Placeholder: add balancing timing if balancing is performed
        # self._timing["balancing_seconds"] = ...

        # --- Reporting Phase ---
        reporting_start = time.perf_counter()
        summary_metrics = self._print_dataset_summary(train_samples, val_samples, test_samples)
        reporting_end = time.perf_counter()
        self._timing["reporting_seconds"] = reporting_end - reporting_start

        total_runtime = time.perf_counter() - phase_start
        self._timing["total_seconds"] = total_runtime

        metrics_report = {
            "timing": dict(self._timing),
            **summary_metrics,
        }

        os.makedirs(self.output_dir, exist_ok=True)
        metrics_path = os.path.join(self.output_dir, "generation_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_report, f, indent=2)
        logger.info(f"Saved generation metrics to: {metrics_path}")

        # Save in Standard Format with optional split-level fact caps.
        train_fact_cap = self.cfg.dataset.get("train_fact_cap", None)
        val_fact_cap = self.cfg.dataset.get("val_fact_cap", None)
        test_fact_cap = self.cfg.dataset.get("test_fact_cap", None)
        train_target_cap = self.cfg.dataset.get("train_target_cap", None)
        val_target_cap = self.cfg.dataset.get("val_target_cap", None)
        test_target_cap = self.cfg.dataset.get("test_target_cap", None)

        if n_train > 0:
            save_standard_dataset(
                train_samples,
                os.path.join(self.output_dir, "train"),
                fact_cap=train_fact_cap,
                target_cap=train_target_cap,
            )
        else:
            logger.info("Skipping train split export because n_train=0")

        if n_val > 0:
            save_standard_dataset(
                val_samples,
                os.path.join(self.output_dir, "val"),
                fact_cap=val_fact_cap,
                target_cap=val_target_cap,
            )
        else:
            logger.info("Skipping val split export because n_val=0")

        if n_test > 0:
            save_standard_dataset(
                test_samples,
                os.path.join(self.output_dir, "test"),
                fact_cap=test_fact_cap,
                target_cap=test_target_cap,
            )
        else:
            logger.info("Skipping test split export because n_test=0")

        return train_samples, val_samples, test_samples

    def _generate_samples(
        self,
        n_samples: int,
        min_individuals: int,
        max_individuals: int,
        min_rules: int,
        max_rules: int,
        target_min_proofs_rule: int,
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
            target_min_proofs_rule:    Minimum proofs to select per rule
            stratified_selection:   Whether to use stratified rule selection
            sample_type:            "TRAIN", "VAL", or "TEST" (for logging)

        Returns:
            List of generated KG samples
        """
        samples = []
        failed_attempts = 0
        max_failed_attempts = n_samples * 10 if n_samples > 0 else 0  # TODO: tune

        print("=" * 80)
        pbar = tqdm(total=n_samples, desc=f"Generating {sample_type} samples")

        while len(samples) < n_samples and failed_attempts < max_failed_attempts:
            if sample_type == "TEST":
                prefix = f"test_{len(samples)}_"
            elif sample_type == "VAL":
                prefix = f"val_{len(samples)}_"
            else:
                prefix = f"train_{len(samples)}_"

            # Reset individual pool for each sample
            self.generator.chainer.reset_individual_pool(name_prefix=prefix)

            # Generate one, individual, knowledge graph sample
            sample = self._generate_one_sample(
                min_individuals=min_individuals,
                max_individuals=max_individuals,
                min_rules=min_rules,
                max_rules=max_rules,
                target_min_proofs_rule=target_min_proofs_rule,
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

    def _proof_difficulty_bucket(self, proof) -> str:
        """Classifies a proof by depth for stratified sampling."""
        depth = proof.depth() if hasattr(proof, "depth") else 1
        if depth <= 1:
            return "easy"
        if 2 <= depth <= 4:
            return "medium"
        return "hard"

    def _select_proofs(self, proofs: List, n_select: int) -> List:
        """Selects proofs according to configured strategy."""
        if not proofs:
            return []

        n_select = max(0, min(n_select, len(proofs)))
        if n_select == 0:
            return []

        strategy = self.proof_selection_strategy
        if strategy == "random":
            return random.sample(proofs, n_select)

        if strategy != "stratified":
            logger.warning(f"Unknown proof_selection_strategy='{strategy}', falling back to random")
            return random.sample(proofs, n_select)

        buckets = {"easy": [], "medium": [], "hard": []}
        for proof in proofs:
            buckets[self._proof_difficulty_bucket(proof)].append(proof)

        available_groups = [g for g in ("easy", "medium", "hard") if buckets[g]]
        if not available_groups:
            return []

        group_weights = {g: max(self.proof_selection_weights.get(g, 0.0), 0.0) for g in available_groups}
        total_weight = sum(group_weights.values())
        if total_weight <= 0:
            group_weights = {g: 1.0 for g in available_groups}
            total_weight = float(len(available_groups))

        quotas = {g: int(n_select * (group_weights[g] / total_weight)) for g in available_groups}
        selected = []
        selected_ids = set()

        for group in available_groups:
            take = min(quotas[group], len(buckets[group]))
            if take <= 0:
                continue
            picks = random.sample(buckets[group], take)
            selected.extend(picks)
            selected_ids.update(id(p) for p in picks)

        while len(selected) < n_select:
            active_groups = [g for g in available_groups if any(id(p) not in selected_ids for p in buckets[g])]
            if not active_groups:
                break

            weights = [group_weights[g] for g in active_groups]
            if sum(weights) <= 0:
                weights = [1.0] * len(active_groups)
            chosen_group = random.choices(active_groups, weights=weights, k=1)[0]
            candidates = [p for p in buckets[chosen_group] if id(p) not in selected_ids]
            if not candidates:
                continue

            pick = random.choice(candidates)
            selected.append(pick)
            selected_ids.add(id(pick))

        return selected

    def _generate_one_sample(
        self,
        min_individuals: int,
        max_individuals: int,
        min_rules: int,
        max_rules: int,
        target_min_proofs_rule: int,
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

        def pick_low_usage(candidates: List):
            if not candidates:
                return None
            if sample_type == "TRAIN":
                usage_dict = self.train_rule_usage
            elif sample_type == "TEST":
                usage_dict = self.test_rule_usage
            else:
                usage_dict = self.rule_selection_count
            min_usage = min(usage_dict.get(r.name, 0) for r in candidates)
            pool = [r for r in candidates if usage_dict.get(r.name, 0) == min_usage]
            return random.choice(pool)

        # We use different variance strategies between each KG (sample)
        # to ensure diversity across the dataset.

        # VARIANCE STRATEGY 1: Vary recursion depth
        current_recursion = random.randint(1, self.max_recursion_cap)
        self.generator.chainer.max_recursion_depth = current_recursion

        # VARIANCE STRATEGY 2: Rule selection (Stratified)
        n_rules = random.randint(min_rules, min(max_rules, len(self.rules)))

        selected_rules = []

        remaining_pool = list(self.rules)

        # Optional coverage-first rule seeding (train split only).
        if self.enforce_goal_predicate_coverage and sample_type == "TRAIN" and self.all_goal_predicates:
            uncovered_goal_predicates = self.all_goal_predicates - self.train_goal_predicates_seen
            coverage_candidates = [
                r
                for r in remaining_pool
                if hasattr(r.conclusion.predicate, "name")
                and str(r.conclusion.predicate.name) in uncovered_goal_predicates
            ]
            if coverage_candidates:
                pick = pick_low_usage(coverage_candidates) if self.rule_sampling_mode == "balanced" else random.choice(
                    coverage_candidates
                )
                if pick is not None:
                    selected_rules.append(pick)
                    remaining_pool.remove(pick)

        # If possible, force at least one complex rule (for depth > 1)
        # but only if recursion depth allows it and we have complex rules
        if current_recursion > 1 and self.complex_rules:
            # Force 1 complex rule
            if not any(r in self.complex_rules for r in selected_rules):
                complex_candidates = [r for r in remaining_pool if r in self.complex_rules]
                if complex_candidates:
                    pick = pick_low_usage(complex_candidates) if self.rule_sampling_mode == "balanced" else random.choice(
                        complex_candidates
                    )
                    selected_rules.append(pick)
                    remaining_pool.remove(pick)
            # Fill rest with random mix
            remaining_count = n_rules - len(selected_rules)
            if remaining_count > 0:
                if self.rule_sampling_mode == "balanced":
                    while remaining_count > 0 and remaining_pool:
                        pick = pick_low_usage(remaining_pool)
                        selected_rules.append(pick)
                        remaining_pool.remove(pick)
                        remaining_count -= 1
                elif remaining_count <= len(remaining_pool):
                    selected_rules.extend(random.sample(remaining_pool, remaining_count))
                else:
                    selected_rules.extend(remaining_pool)
        else:
            # Pure random selection
            remaining_count = n_rules - len(selected_rules)
            if remaining_count > 0:
                if self.rule_sampling_mode == "balanced":
                    while remaining_count > 0 and remaining_pool:
                        pick = pick_low_usage(remaining_pool)
                        selected_rules.append(pick)
                        remaining_pool.remove(pick)
                        remaining_count -= 1
                else:
                    selected_rules.extend(random.sample(remaining_pool, min(remaining_count, len(remaining_pool))))

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
        sample_goal_predicates_generated: set[str] = set()

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
            n_select = random.randint(min(len(proofs), target_min_proofs_rule), min(len(proofs), MAX_PROOFS_CAP))
            selected = self._select_proofs(proofs, n_select)
            if selected:
                predicate = rule.conclusion.predicate
                predicate_name = getattr(predicate, "name", predicate)
                sample_goal_predicates_generated.add(str(predicate_name))

            for proof in selected:
                extracted_map = extract_proof_map(proof)
                for atom, proof_list in extracted_map.items():
                    sample_proof_map[atom].extend(proof_list)
                atoms_found = True

                # Export positive proofs if required
                if self.export_proofs and self.proof_output_dir:
                    pos_dir = os.path.join(self.proof_output_dir, "positive")
                    os.makedirs(pos_dir, exist_ok=True)
                    # Limit the number of exported proofs to prevent generating thousands of files
                    # We can use a simple cap based on existing files or random sampling.
                    # Here we just generate a unique filename
                    import uuid

                    uid = str(uuid.uuid4())[:8]
                    filename = f"positive_proof_{rule.name}_{uid}"
                    full_path = os.path.join(pos_dir, filename)
                    # We will rely on MAX_EXPORTS later to cap negatives, so just save positives freely here
                    # To prevent infinite growth, we might only save a fraction:
                    if random.random() < 0.2:  # Save 20% of positive proofs to avoid cluttering
                        proof.save_visualization(
                            full_path,
                            format="pdf",
                            title=f"Positive Proof ({rule.name})",
                            root_label="TRUE CONCLUSION",
                            fact_type="inferred",
                        )

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

        # VALIDATION: Check for complexity (inferred facts and depth)

        # Count inferred facts (triples + memberships)
        n_inferred = 0
        max_depth = 0

        for t in kg.triples:
            if not t.is_base_fact:
                n_inferred += 1
                max_depth = max(max_depth, t.get_hops())

        for m in kg.memberships:
            if not m.is_base_fact:
                n_inferred += 1
                max_depth = max(max_depth, m.get_hops())

        # Reject if no inferred facts (unless we explicitly allowed only base facts, which is rare for this task)
        if n_inferred == 0:
            if self.verbose:
                logger.debug("  [DEBUG] Sample rejected: No inferred facts found.")
            return None

        # Reject if depth is too shallow (only if configured for deep recursion)
        # If max_recursion > 1, we expect at least depth 2 for better training data
        if self.generator.chainer.max_recursion_depth > 1 and max_depth < 2:
            # Allow some shallow samples (e.g. 20%) to keep variety, but prefer deep ones
            if random.random() > 0.2:
                if self.verbose:
                    logger.debug(f"  [DEBUG] Sample rejected: Max depth {max_depth} too low (expected >= 2).")
                return None

        # Add negatives via NegativeSampler
        kg = self.negative_sampler.add_negative_samples(
            kg,
            strategy=self.neg_strategy,
            ratio=self.neg_ratio,
            corrupt_base_facts=self.neg_corrupt_base_facts,
            export_proofs=self.export_proofs,
            output_dir=self.proof_output_dir,
        )

        if sample_type == "TRAIN":
            self.train_goal_predicates_seen.update(sample_goal_predicates_generated)
        elif sample_type == "TEST":
            self.test_goal_predicates_seen.update(sample_goal_predicates_generated)

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
    ) -> Dict[str, Dict[str, float]]:
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

        if self.all_goal_predicates:
            train_goal_cov = (len(self.train_goal_predicates_seen) / len(self.all_goal_predicates)) * 100
            test_goal_cov = (len(self.test_goal_predicates_seen) / len(self.all_goal_predicates)) * 100
            logger.info(
                "Goal predicate coverage (train): {:.2f}% ({}/{})",
                train_goal_cov,
                len(self.train_goal_predicates_seen),
                len(self.all_goal_predicates),
            )
            logger.info(
                "Goal predicate coverage (test): {:.2f}% ({}/{})",
                test_goal_cov,
                len(self.test_goal_predicates_seen),
                len(self.all_goal_predicates),
            )
            uncovered_train_goals = sorted(self.all_goal_predicates - self.train_goal_predicates_seen)
            if uncovered_train_goals:
                logger.warning("Uncovered train goal predicates ({}): {}", len(uncovered_train_goals), uncovered_train_goals)

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
        total_rules = len(self.rules)
        overall_triggered = set(self.train_rule_usage.keys()) | set(self.test_rule_usage.keys())
        if total_rules > 0:
            train_coverage_pct = (len(self.train_rule_usage) / total_rules) * 100
            test_coverage_pct = (len(self.test_rule_usage) / total_rules) * 100
            overall_coverage_pct = (len(overall_triggered) / total_rules) * 100
        else:
            train_coverage_pct = 0.0
            test_coverage_pct = 0.0
            overall_coverage_pct = 0.0

        logger.info("=" * 80)
        logger.info("ONTOLOGY COVERAGE")
        logger.info("=" * 80)
        logger.info(f"Train coverage:   {train_coverage_pct:.2f}% ({len(self.train_rule_usage)}/{total_rules})")
        logger.info(f"Test coverage:    {test_coverage_pct:.2f}% ({len(self.test_rule_usage)}/{total_rules})")
        logger.info(f"Overall coverage: {overall_coverage_pct:.2f}% ({len(overall_triggered)}/{total_rules})")

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
            base_facts = stats.get("n_base_facts", 0)
            inferred_facts = stats.get("n_inferred_facts", 0)
            yield_base_to_inferred = (base_facts / inferred_facts) if inferred_facts > 0 else float("nan")
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
            print(f"  Yield Rate (Base/Inferred): {yield_base_to_inferred:.3f}")
            print("  Negative Logic (Avg per sample):")
            print(
                f"    - Base Fact Corruption:      {stats.get('avg_neg_base_source', 0):.1f} (Direct corruption of base facts)"
            )
            print(
                f"    - Inferred Fact (Shallow):   {stats.get('avg_neg_inferred_shallow', 0):.1f} (Direct corruption of inferred facts)"
            )
            print(
                f"    - Inferred Fact (Deep/Prop): {stats.get('avg_neg_inferred_deep', 0):.1f} (Propagated consequences)"
            )
            print(f"    - Unknown/Other:             {stats.get('avg_neg_unknown', 0):.1f}")

        print_split_stats("TRAINING", train_stats)
        print_split_stats("VAL", val_stats)
        print_split_stats("TEST", test_stats)

        print(f"{'=' * 80}\n")

        def safe_div(numerator: float, denominator: float) -> Optional[float]:
            return (numerator / denominator) if denominator > 0 else None

        return {
            "ontology_coverage_pct": {
                "train": train_coverage_pct,
                "test": test_coverage_pct,
                "overall": overall_coverage_pct,
            },
            "goal_predicate_coverage_pct": {
                "train": (len(self.train_goal_predicates_seen) / len(self.all_goal_predicates) * 100)
                if self.all_goal_predicates
                else 0.0,
                "test": (len(self.test_goal_predicates_seen) / len(self.all_goal_predicates) * 100)
                if self.all_goal_predicates
                else 0.0,
            },
            "yield_rate_base_to_inferred": {
                "train": safe_div(train_stats.get("n_base_facts", 0), train_stats.get("n_inferred_facts", 0)),
                "val": safe_div(val_stats.get("n_base_facts", 0), val_stats.get("n_inferred_facts", 0)),
                "test": safe_div(test_stats.get("n_base_facts", 0), test_stats.get("n_inferred_facts", 0)),
                "overall": safe_div(
                    train_stats.get("n_base_facts", 0)
                    + val_stats.get("n_base_facts", 0)
                    + test_stats.get("n_base_facts", 0),
                    train_stats.get("n_inferred_facts", 0)
                    + val_stats.get("n_inferred_facts", 0)
                    + test_stats.get("n_inferred_facts", 0),
                ),
            },
        }


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


def save_standard_dataset(
    samples: List[KnowledgeGraph],
    output_base_dir: str,
    fact_cap: Optional[int] = None,
    target_cap: Optional[int] = None,
) -> None:
    """
    Saves the dataset in the Standard Format:
    - facts.csv: Base facts (positives, hops=0)
    - targets.csv: Everything else (queries, negatives, inferred)

    Args:
        samples (List[KnowledgeGraph]): List of samples to save.
        output_base_dir (str): output directory (e.g. data/train).
        fact_cap (Optional[int]): Maximum number of rows in facts.csv.
        target_cap (Optional[int]): Maximum number of rows in targets.csv.
    """
    path = Path(output_base_dir)
    path.mkdir(parents=True, exist_ok=True)

    facts_rows = []
    targets_rows = []

    retained_samples = 0

    for idx, kg in enumerate(samples):
        # Sample ID is 1000 + idx to avoid 0-indexing issues if any
        sample_id = str(1000 + idx)

        all_rows = kg.to_standard_rows(sample_id)

        sample_facts_rows = []
        sample_targets_rows = []

        for row in all_rows:
            # FACTS: Only Positive Base Facts
            if row["type"] == "base_fact" and row["label"] == 1:
                # Minimal columns for facts.csv: sample_id, subject, predicate, object
                sample_facts_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                    }
                )
                # Base facts are ALSO targets (trivial logic)
                sample_targets_rows.append(row)
            else:
                # Everything else is a target
                sample_targets_rows.append(row)

        # Keep complete samples when applying caps so facts/targets stay aligned.
        if fact_cap is not None and retained_samples > 0 and (len(facts_rows) + len(sample_facts_rows)) > fact_cap:
            break
        if (
            target_cap is not None
            and retained_samples > 0
            and (len(targets_rows) + len(sample_targets_rows)) > target_cap
        ):
            break

        facts_rows.extend(sample_facts_rows)
        targets_rows.extend(sample_targets_rows)
        retained_samples += 1

    if fact_cap is not None and len(facts_rows) > fact_cap:
        facts_rows = facts_rows[:fact_cap]

    if target_cap is not None and len(targets_rows) > target_cap:
        targets_rows = targets_rows[:target_cap]

    if fact_cap is not None or target_cap is not None:
        fact_sample_ids = {row["sample_id"] for row in facts_rows}
        target_sample_ids = {row["sample_id"] for row in targets_rows}
        retained_sample_ids = fact_sample_ids & target_sample_ids
        facts_rows = [row for row in facts_rows if row.get("sample_id") in retained_sample_ids]
        targets_rows = [row for row in targets_rows if row.get("sample_id") in retained_sample_ids]

        logger.info(
            "Applied caps for %s: fact_cap=%s, target_cap=%s, kept_samples=%s, kept_facts=%s, kept_targets=%s"
            % (
                output_base_dir,
                fact_cap,
                target_cap,
                len(retained_sample_ids),
                len(facts_rows),
                len(targets_rows),
            )
        )

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

    logger.success(
        f"Saved standard dataset to {output_base_dir} ({len(facts_rows)} facts, {len(targets_rows)} targets)"
    )


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
    logger.add(
        log_path, mode="w", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

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
        min_individuals=cfg.generator.min_individuals,
        max_individuals=cfg.generator.max_individuals,
        min_rules=cfg.generator.min_rules,
        max_rules=cfg.generator.max_rules,
        target_min_proofs_rule=cfg.generator.target_min_proofs_rule,
    )

    # Save to CSV (Individual Samples) - Optional
    output_dir = cfg.dataset.output_dir

    if cfg.generator.get("save_individual_samples", False):
        logger.info("Saving individual sample CSVs...")
        if cfg.dataset.n_train > 0:
            save_dataset_to_csv(train_samples, f"{output_dir}/train", prefix="train_sample")
        if cfg.dataset.n_val > 0:
            save_dataset_to_csv(val_samples, f"{output_dir}/val", prefix="val_sample")
        if cfg.dataset.n_test > 0:
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
