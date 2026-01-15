"""
DESCRIPTION

    Advanced negative sampling strategies for Knowledge Graph data generation.

    Strategies:
    1. Random corruption (baseline)
    2. Constrained corruption (respects domain/range)
    3. Proof-based corruption (corrupts base facts to falsify goals)
    4. Type-aware corruption (respects class memberships)

AUTHOR

    Vincent Van Schependom
"""

import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set

from omegaconf import DictConfig

from synthology.data_structures import (
    RDF,
    Atom,
    Class,
    Individual,
    KnowledgeGraph,
    Membership,
    Proof,
    Relation,
    Term,
    Triple,
    Var,
)

logger = logging.getLogger(__name__)


class NegativeSampler:
    """
    Handles generation of negative samples using multiple strategies.
    """

    def __init__(
        self,
        schema_classes: Dict[str, Class],
        schema_relations: Dict[str, Relation],
        cfg: DictConfig,
        domains: Optional[Dict[str, Set[str]]] = None,
        ranges: Optional[Dict[str, Set[str]]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the negative sampler.

        Args:
            schema_classes: Dict of class name -> Class object
            schema_relations: Dict of relation name -> Relation object
            cfg: Hydra configuration object.
            domains: Dict of relation name -> set of domain class names
            ranges: Dict of relation name -> set of range class names
            verbose: Enable debug output
        """
        self.schema_classes = schema_classes
        self.schema_relations = schema_relations
        self.cfg = cfg
        self.domains = domains or {}
        self.ranges = ranges or {}
        self.verbose = verbose

        # Optimization: Sets for fast lookup of existing facts
        self.existing_triples: Set[str] = set()
        self.existing_memberships: Set[str] = set()

        # Track strategy usage
        self.strategy_usage: Dict[str, int] = defaultdict(int)

    def _index_existing_facts(self, kg: KnowledgeGraph):
        """Build sets of existing facts for O(1) lookup."""
        self.existing_triples.clear()
        self.existing_memberships.clear()

        for t in kg.triples:
            if t.positive:
                # Key: (subject, predicate, object)
                key = f"{t.subject.name}|{t.predicate.name}|{t.object.name}"
                self.existing_triples.add(key)

        for m in kg.memberships:
            if m.is_member:
                # Key: (individual, class)
                key = f"{m.individual.name}|{m.cls.name}"
                self.existing_memberships.add(key)

    def add_negative_samples(
        self,
        kg: KnowledgeGraph,
        strategy: str = "constrained",
        ratio: float = 1.0,
        corrupt_base_facts: bool = False,
        export_proofs: bool = False,
        output_dir: Optional[str] = None,
    ) -> KnowledgeGraph:
        """
        Add negative samples to a knowledge graph.

        Args:
            kg: Knowledge graph to add negatives to
            strategy: Negative sampling strategy
                - "random": Random corruption
                - "constrained": Respects domain/range constraints
                - "proof_based": Corrupts base facts in proof trees
                - "type_aware": Considers class memberships
                - "mixed": Randomly selects one of the above for each sample
            ratio: Ratio of negative to positive samples (1.0 = balanced)
            corrupt_base_facts: If True, also corrupt base facts in proofs
            export_proofs: Whether to export visualizations of corrupted proofs
            output_dir: Directory to save visualizations

        Returns:
            Knowledge graph with negative samples added
        """
        # Index existing facts for fast lookup
        self._index_existing_facts(kg)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if strategy != "mixed":
            # For single strategies, we can just count the number of negatives added
            # But we need to know how many were actually added.
            # The methods return the KG, so we can check the difference in length.
            initial_len = len(kg.triples)

        if strategy == "random":
            kg = self._random_corruption(kg, ratio)
            self.strategy_usage["random"] += len(kg.triples) - initial_len
            return kg
        elif strategy == "constrained":
            kg = self._constrained_corruption(kg, ratio)
            self.strategy_usage["constrained"] += len(kg.triples) - initial_len
            return kg
        elif strategy == "proof_based":
            kg = self._proof_based_corruption(kg, ratio, corrupt_base_facts, export_proofs, output_dir)
            self.strategy_usage["proof_based"] += len(kg.triples) - initial_len
            return kg
        elif strategy == "type_aware":
            kg = self._type_aware_corruption(kg, ratio)
            self.strategy_usage["type_aware"] += len(kg.triples) - initial_len
            return kg
        elif strategy == "mixed":
            return self._mixed_corruption(kg, ratio, corrupt_base_facts)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _random_corruption(self, kg: KnowledgeGraph, ratio: float) -> KnowledgeGraph:
        """
        Strategy 1: Random corruption (baseline).

        For each positive triple, randomly corrupt subject OR object with any individual.
        """
        positive_triples = [t for t in kg.triples if t.positive]
        n_negatives = int(len(positive_triples) * ratio)
        negative_triples = []

        for _ in range(n_negatives):
            if not positive_triples:
                break

            pos_triple = random.choice(positive_triples)
            neg_triple = self._corrupt_triple_random(pos_triple, kg.individuals, original_triple=pos_triple)

            if neg_triple and self.is_valid_negative(neg_triple):
                negative_triples.append(neg_triple)

        kg.triples.extend(negative_triples)

        # Also add negative memberships
        positive_memberships = [m for m in kg.memberships if m.is_member]
        n_neg_memberships = int(len(positive_memberships) * ratio)
        negative_memberships = []

        for _ in range(n_neg_memberships):
            if not positive_memberships:
                break

            pos_mem = random.choice(positive_memberships)
            neg_mem = self._corrupt_membership_random(
                pos_mem, list(self.schema_classes.values()), original_membership=pos_mem
            )

            if neg_mem and not self._is_positive_membership(neg_mem, kg):
                negative_memberships.append(neg_mem)

        kg.memberships.extend(negative_memberships)

        return kg

    def _constrained_corruption(self, kg: KnowledgeGraph, ratio: float) -> KnowledgeGraph:
        """
        Strategy 2: Constrained corruption.

        Respects domain/range constraints when corrupting.
        Only substitutes individuals that satisfy type constraints.
        """
        positive_triples = [t for t in kg.triples if t.positive]
        n_negatives = int(len(positive_triples) * ratio)
        negative_triples = []

        # Build individual -> classes mapping
        ind_classes = self._build_individual_classes_map(kg)

        for _ in range(n_negatives):
            if not positive_triples:
                break

            pos_triple = random.choice(positive_triples)

            # Get valid candidates based on domain/range
            if random.random() < 0.5:
                # Corrupt subject (check domain)
                candidates = self._get_domain_candidates(pos_triple.predicate, kg.individuals, ind_classes)
                candidates = [c for c in candidates if c != pos_triple.subject]
                if candidates:
                    new_subj = random.choice(candidates)
                    neg_triple = Triple(
                        new_subj,
                        pos_triple.predicate,
                        pos_triple.object,
                        positive=False,
                        proofs=[],
                        metadata={"source_type": "base" if pos_triple.is_base_fact else "inferred"},
                    )
                else:
                    continue
            else:
                # Corrupt object (check range)
                candidates = self._get_range_candidates(pos_triple.predicate, kg.individuals, ind_classes)
                candidates = [c for c in candidates if c != pos_triple.object]
                if candidates:
                    new_obj = random.choice(candidates)
                    neg_triple = Triple(
                        pos_triple.subject,
                        pos_triple.predicate,
                        new_obj,
                        positive=False,
                        proofs=[],
                        metadata={"source_type": "base" if pos_triple.is_base_fact else "inferred"},
                    )
                else:
                    continue

            if self.is_valid_negative(neg_triple):
                negative_triples.append(neg_triple)

        kg.triples.extend(negative_triples)

        # Constrained membership negatives
        positive_memberships = [m for m in kg.memberships if m.is_member]
        n_neg_memberships = int(len(positive_memberships) * ratio)
        negative_memberships = []

        for _ in range(n_neg_memberships):
            if not positive_memberships:
                break

            pos_mem = random.choice(positive_memberships)

            # Get classes the individual is NOT in
            current_classes = {m.cls.name for m in kg.memberships if m.individual == pos_mem.individual and m.is_member}

            candidate_classes = [c for c in self.schema_classes.values() if c.name not in current_classes]

            if candidate_classes:
                neg_cls = random.choice(candidate_classes)
                neg_mem = Membership(
                    pos_mem.individual,
                    neg_cls,
                    is_member=False,
                    proofs=[],
                    metadata={"source_type": "base" if pos_mem.is_base_fact else "inferred"},
                )
                negative_memberships.append(neg_mem)

        kg.memberships.extend(negative_memberships)

        return kg

    def _proof_based_corruption(
        self,
        kg: KnowledgeGraph,
        ratio: float,
        corrupt_base_facts: bool,
        export_proofs: bool = False,
        output_dir: Optional[str] = None,
    ) -> KnowledgeGraph:
        """
        Strategy 3: Proof-based corruption.

        Corrupts facts in proof trees to create negatives that would falsify inferences.
        This creates harder negatives that test reasoning capabilities.

        Example:
            If hasGrandparent(A, C) is inferred from hasParent(A, B) ∧ hasParent(B, C),
            then corrupting hasParent(A, B) to hasParent(A, D) falsifies the inference.
        """
        positive_triples = [t for t in kg.triples if t.positive]
        n_negatives = int(len(positive_triples) * ratio)
        negative_triples = []

        # Collect triples with proofs
        # Prioritize inferred facts (those with non-trivial proofs)
        inferred_triples_with_proofs = []
        base_triples_with_proofs = []

        # Pre-calc ind_classes for constrained corruption
        ind_classes = self._build_individual_classes_map(kg)

        for t in positive_triples:
            if t.proofs:
                # Check if any proof is non-trivial (has a rule)
                inferred_proofs = [p for p in t.proofs if p.rule is not None]
                if inferred_proofs:
                    inferred_triples_with_proofs.append((t, inferred_proofs))
                else:
                    base_triples_with_proofs.append((t, t.proofs))

        # Use inferred triples if available, otherwise fallback to base
        # We want to prioritize breaking reasoning chains
        if inferred_triples_with_proofs:
            candidate_pool = inferred_triples_with_proofs
            if self.verbose:
                logger.debug(f"Prioritizing {len(candidate_pool)} inferred facts for corruption")
        else:
            candidate_pool = base_triples_with_proofs
            if self.verbose:
                logger.debug(f"No inferred facts found, falling back to {len(candidate_pool)} base facts")

        if not candidate_pool:
            # Fallback to random if no proofs available
            return self._random_corruption(kg, ratio)

        # Limit exported visualizations to prevent freezing
        exported_corrupted_count = 0
        exported_propagated_count = 0
        MAX_EXPORTS = 200

        attempts = 0
        max_attempts = n_negatives * 10  # Safety limit to prevent infinite loops

        while len(negative_triples) < n_negatives and attempts < max_attempts:
            attempts += 1

            if not candidate_pool:
                logger.warning("No triples with proofs found")
                break

            # Pick a triple with proofs
            pos_triple, proofs = random.choice(candidate_pool)

            if not proofs:
                logger.warning(f"No proofs found for triple: {pos_triple}")
                continue

            # Pick a random proof
            proof = random.choice(proofs)

            # Get base facts from this proof
            base_facts = proof.get_base_facts()

            # Filter out RDF.type facts if we are in corrupt_base_facts mode
            if corrupt_base_facts:
                base_facts = {bf for bf in base_facts if bf.predicate != RDF.type}

            neg_triple = None

            if not base_facts or not corrupt_base_facts:
                # Corrupt the goal instead
                # Use constrained corruption if possible, or fallback to random
                # Ideally we want type-safe corruption even for leaf goals?
                neg_triple = self._corrupt_triple_constrained(pos_triple, kg.individuals, ind_classes)
                if not neg_triple:
                    neg_triple = self._corrupt_triple_random(pos_triple, kg.individuals, original_triple=pos_triple)

                if neg_triple:
                    neg_triple.metadata["source_type"] = "inferred"  # Explicitly inferred since we target goal
                    neg_triple.metadata["explanation"] = self._generate_explanation(
                        pos_triple, neg_triple, "Corrupted Goal (Leaf)"
                    )
            else:
                # Corrupt a base fact from the proof
                # This creates a negative that would break the inference chain
                base_fact = random.choice(list(base_facts))

                # Find corresponding triple in KG
                matching_triples = [
                    t
                    for t in kg.triples
                    if (
                        t.subject.name == base_fact.subject.name
                        and t.predicate.name == base_fact.predicate.name
                        and t.object.name == base_fact.object.name
                        and t.positive
                    )
                ]

                if matching_triples:
                    base_triple = matching_triples[0]
                    # Try constrained corruption first
                    neg_triple = self._corrupt_triple_constrained(base_triple, kg.individuals, ind_classes)
                    if not neg_triple:
                        neg_triple = self._corrupt_triple_random(
                            base_triple, kg.individuals, original_triple=base_triple
                        )

                    if neg_triple:
                        neg_triple.metadata["source_type"] = "base"  # Explicitly base since we target base fact
                        neg_triple.metadata["explanation"] = self._generate_explanation(
                            base_triple, neg_triple, f"Corrupted Base Fact in proof for {pos_triple}"
                        )

                    propagated_exported = False

                    # PROPAGATION: Try to derive the falsified goal

                    if proof.rule:
                        # Find which variable mapped to the original object/subject that we changed
                        new_subst = proof.substitutions.copy()

                        # Check if we changed subject or object
                        changed_term = None
                        new_term = None

                        if neg_triple.subject != base_triple.subject:
                            changed_term = base_triple.subject
                            new_term = neg_triple.subject
                        elif neg_triple.object != base_triple.object:
                            changed_term = base_triple.object
                            new_term = neg_triple.object
                        else:
                            if self.verbose:
                                logger.debug(
                                    f"Propagation failed: Corruption did not change subject or object? {base_triple} => {neg_triple}"
                                )

                        if changed_term and new_term:
                            # Update substitution for all variables that mapped to the changed term
                            for var, term in proof.substitutions.items():
                                if term == changed_term:
                                    new_subst[var] = new_term

                            # Instantiate conclusion with new substitution
                            # PROBLEM: proof.substitutions has renamed vars (e.g. X_13), but proof.rule has original vars (e.g. X)
                            # SOLUTION: Create a normalized substitution mapping original vars to values

                            normalized_subst = {}
                            for var, term in new_subst.items():
                                # Extract base name (e.g. "X_13" -> "X")
                                # Assuming format Name_ID
                                base_name = var.name.split("_")[0]

                                # Find corresponding variable in original rule
                                for rule_var in proof.rule.conclusion.get_variables():
                                    if rule_var.name == base_name:
                                        normalized_subst[rule_var] = term

                            new_goal_atom = proof.rule.conclusion.substitute(normalized_subst)

                            if new_goal_atom.is_ground():
                                # Create Triple from Atom

                                # Check if it's a valid Triple structure
                                if isinstance(new_goal_atom.predicate, Relation):
                                    neg_goal = Triple(
                                        subject=new_goal_atom.subject,
                                        predicate=new_goal_atom.predicate,
                                        object=new_goal_atom.object,
                                        positive=False,
                                        proofs=[],
                                        metadata={
                                            "source_type": "propagated_inferred",
                                            "explanation": (
                                                f"Propagated from corrupted base fact "
                                                f"({base_triple.subject.name} {base_triple.predicate.name} {base_triple.object.name} "
                                                f"-> {neg_triple.subject.name} {neg_triple.predicate.name} {neg_triple.object.name}) "
                                                f"via Rule {proof.rule.name if proof.rule else 'Unknown'}"
                                            ),
                                        },
                                    )

                                    # Check if this new goal contradicts existing facts
                                    # Safe-guard: explicitly check against original positive triple
                                    if (
                                        neg_goal.subject.name == pos_triple.subject.name
                                        and neg_goal.predicate.name == pos_triple.predicate.name
                                        and neg_goal.object.name == pos_triple.object.name
                                    ):
                                        if self.verbose:
                                            logger.debug(
                                                "Propagation failed: Derived goal is identical to original positive triple."
                                            )
                                    elif self.is_valid_negative(neg_goal):
                                        # CRITICAL CHECK: Verify other premises hold
                                        # Only add if the rule chain is otherwise valid
                                        neg_base_atom = Atom(
                                            predicate=neg_triple.predicate,
                                            subject=neg_triple.subject,
                                            object=neg_triple.object,
                                        )

                                        if self._check_premises_satisfied(proof.rule, normalized_subst, neg_base_atom):
                                            negative_triples.append(neg_goal)

                                            # VISUALIZATION: Export the propagated proof
                                            if export_proofs and output_dir and exported_propagated_count < MAX_EXPORTS:
                                                # Create atom from negative goal
                                                neg_goal_atom = Atom(
                                                    predicate=neg_goal.predicate,
                                                    subject=neg_goal.subject,
                                                    object=neg_goal.object,
                                                )

                                                # Create atom from corrupted base fact
                                                neg_base_atom = Atom(
                                                    predicate=neg_triple.predicate,
                                                    subject=neg_triple.subject,
                                                    object=neg_triple.object,
                                                )

                                                term_mapping = {changed_term: new_term}
                                                
                                                # Reconstruct the proof tree with corrupted values
                                                propagated_proof = self._create_propagated_proof(
                                                    proof,
                                                    base_fact,
                                                    neg_base_atom,
                                                    term_mapping
                                                )

                                            if propagated_proof:
                                                filename = f"propagated_proof_{len(negative_triples)}_{neg_goal.subject.name}_{neg_goal.predicate.name}_{neg_goal.object.name}"
                                                full_path = os.path.join(output_dir, filename)
                                                propagated_proof.save_visualization(
                                                    full_path,
                                                    format="pdf",
                                                    title="Counterfactual Proof (False Pattern)",
                                                    root_label="FALSE CONCLUSION (DERIVED)",
                                                )
                                                exported_propagated_count += 1
                                                propagated_exported = True
                                    else:
                                        if self.verbose:
                                            logger.debug(
                                                f"Propagation failed: Derived goal {neg_goal} contradicts existing positive fact."
                                            )
                                else:
                                    if self.verbose:
                                        logger.debug(
                                            f"Propagation failed: Derived goal predicate {new_goal_atom.predicate} is not a Relation."
                                        )
                            else:
                                if self.verbose:
                                    logger.debug(f"Propagation failed: New goal atom is not ground: {new_goal_atom}")
                        else:
                            if self.verbose:
                                logger.debug("Propagation failed: Could not determine changed term.")
                    else:
                        # This happens if proof.rule is None.
                        # If we are corrupting a base fact that has no rule (i.e. it's a root fact itself),
                        # then there is no propagation to do. This is normal for base facts.
                        if self.verbose:
                            logger.debug(
                                f"  [INFO] No propagation: Proof for {pos_triple} has no rule (likely a base fact)."
                            )

                    # If we have a negative triple and we want to export proofs
                    # Only export corrupted proof if we didn't export a propagated one (avoid redundancy)
                    if (
                        neg_triple
                        and export_proofs
                        and output_dir
                        and exported_corrupted_count < MAX_EXPORTS
                        and not propagated_exported
                    ):
                        # Create atom from negative triple
                        new_atom = Atom(
                            predicate=neg_triple.predicate,
                            subject=neg_triple.subject,
                            object=neg_triple.object,
                        )

                        # Create corrupted proof
                        corrupted_proof = proof.corrupt_leaf(base_fact, new_atom)

                        # Save visualization
                        filename = f"corrupted_proof_{len(negative_triples)}_{pos_triple.subject.name}_{pos_triple.predicate.name}_{pos_triple.object.name}"
                        full_path = os.path.join(output_dir, filename)
                        corrupted_proof.save_visualization(
                            full_path, format="pdf", root_label="FALSE FACT (CORRUPTED)"
                        )
                        exported_corrupted_count += 1

            if neg_triple and self.is_valid_negative(neg_triple):
                negative_triples.append(neg_triple)

        kg.triples.extend(negative_triples)
        return kg

    def _type_aware_corruption(self, kg: KnowledgeGraph, ratio: float) -> KnowledgeGraph:
        """
        Strategy 4: Type-aware corruption.

        Only corrupts with individuals of the same types to create semantically valid
        but factually incorrect negatives.

        Example:
            If hasParent(Person1, Person2), only corrupt with other Persons.
        """
        positive_triples = [t for t in kg.triples if t.positive]
        n_negatives = int(len(positive_triples) * ratio)
        negative_triples = []

        # Build individual -> classes mapping
        ind_classes = self._build_individual_classes_map(kg)

        # Group individuals by their class sets
        class_groups = defaultdict(list)
        for ind in kg.individuals:
            classes = frozenset(ind_classes.get(ind, set()))
            class_groups[classes].append(ind)

        for _ in range(n_negatives):
            if not positive_triples:
                break

            pos_triple = random.choice(positive_triples)

            if random.random() < 0.5:
                # Corrupt subject with same-type individual
                subj_classes = frozenset(ind_classes.get(pos_triple.subject, set()))
                candidates = [c for c in class_groups[subj_classes] if c != pos_triple.subject]

                if candidates:
                    new_subj = random.choice(candidates)
                    neg_triple = Triple(
                        new_subj,
                        pos_triple.predicate,
                        pos_triple.object,
                        positive=False,
                        proofs=[],
                        metadata={"source_type": "base" if pos_triple.is_base_fact else "inferred"},
                    )
                else:
                    continue
            else:
                # Corrupt object with same-type individual
                obj_classes = frozenset(ind_classes.get(pos_triple.object, set()))
                candidates = [c for c in class_groups[obj_classes] if c != pos_triple.object]

                if candidates:
                    new_obj = random.choice(candidates)
                    neg_triple = Triple(
                        pos_triple.subject,
                        pos_triple.predicate,
                        new_obj,
                        positive=False,
                        proofs=[],
                        metadata={"source_type": "base" if pos_triple.is_base_fact else "inferred"},
                    )
                else:
                    continue

            if self.is_valid_negative(neg_triple):
                negative_triples.append(neg_triple)

        kg.triples.extend(negative_triples)
        return kg

    def _mixed_corruption(
        self,
        kg: KnowledgeGraph,
        ratio: float,
        corrupt_base_facts: bool,
    ) -> KnowledgeGraph:
        """
        Strategy 5: Mixed corruption.

        Randomly selects a strategy for each negative sample to create a diverse dataset.
        """
        positive_triples = [t for t in kg.triples if t.positive]
        n_negatives = int(len(positive_triples) * ratio)
        negative_triples = []

        # Pre-calculate data structures needed for different strategies
        ind_classes = self._build_individual_classes_map(kg)

        # Group individuals by their class sets (for type_aware)
        class_groups = defaultdict(list)
        for ind in kg.individuals:
            classes = frozenset(ind_classes.get(ind, set()))
            class_groups[classes].append(ind)

        triples_with_proofs = [(t, t.proofs) for t in positive_triples if t.proofs]

        strategies = ["random", "constrained", "type_aware"]
        if triples_with_proofs:
            strategies.append("proof_based")

        for _ in range(n_negatives):
            if not positive_triples:
                break

            # Pick a random strategy for this sample
            strategy = random.choice(strategies)

            neg_triple = None

            if strategy == "random":
                pos_triple = random.choice(positive_triples)
                neg_triple = self._corrupt_triple_random(pos_triple, kg.individuals, original_triple=pos_triple)

            elif strategy == "constrained":
                pos_triple = random.choice(positive_triples)
                if random.random() < 0.5:
                    # Corrupt subject
                    candidates = self._get_domain_candidates(pos_triple.predicate, kg.individuals, ind_classes)
                    candidates = [c for c in candidates if c != pos_triple.subject]
                    if candidates:
                        new_subj = random.choice(candidates)
                        neg_triple = Triple(
                            new_subj,
                            pos_triple.predicate,
                            pos_triple.object,
                            positive=False,
                            proofs=[],
                            metadata={"source_type": "base" if pos_triple.is_base_fact else "inferred"},
                        )
                else:
                    # Corrupt object
                    candidates = self._get_range_candidates(pos_triple.predicate, kg.individuals, ind_classes)
                    candidates = [c for c in candidates if c != pos_triple.object]
                    if candidates:
                        new_obj = random.choice(candidates)
                        neg_triple = Triple(
                            pos_triple.subject,
                            pos_triple.predicate,
                            new_obj,
                            positive=False,
                            proofs=[],
                            metadata={"source_type": "base" if pos_triple.is_base_fact else "inferred"},
                        )

            elif strategy == "type_aware":
                pos_triple = random.choice(positive_triples)
                if random.random() < 0.5:
                    # Corrupt subject
                    subj_classes = frozenset(ind_classes.get(pos_triple.subject, set()))
                    candidates = [c for c in class_groups[subj_classes] if c != pos_triple.subject]
                    if candidates:
                        new_subj = random.choice(candidates)
                        neg_triple = Triple(
                            new_subj,
                            pos_triple.predicate,
                            pos_triple.object,
                            positive=False,
                            proofs=[],
                            metadata={"source_type": "base" if pos_triple.is_base_fact else "inferred"},
                        )
                else:
                    # Corrupt object
                    obj_classes = frozenset(ind_classes.get(pos_triple.object, set()))
                    candidates = [c for c in class_groups[obj_classes] if c != pos_triple.object]
                    if candidates:
                        new_obj = random.choice(candidates)
                        neg_triple = Triple(
                            pos_triple.subject,
                            pos_triple.predicate,
                            new_obj,
                            positive=False,
                            proofs=[],
                            metadata={"source_type": "base" if pos_triple.is_base_fact else "inferred"},
                        )

            elif strategy == "proof_based":
                # Simplified proof based logic for single sample
                if triples_with_proofs:
                    pos_triple, proofs = random.choice(triples_with_proofs)
                    if proofs:
                        proof = random.choice(proofs)
                        base_facts = proof.get_base_facts()

                        if corrupt_base_facts:
                            base_facts = {bf for bf in base_facts if bf.predicate != RDF.type}

                        if not base_facts or not corrupt_base_facts:
                            neg_triple = self._corrupt_triple_random(
                                pos_triple, kg.individuals, original_triple=pos_triple
                            )
                            if neg_triple:
                                neg_triple.metadata["source_type"] = "inferred"
                        else:
                            base_fact = random.choice(list(base_facts))
                            # Find matching triple
                            matching_triples = [
                                t
                                for t in kg.triples
                                if t.positive
                                and t.subject.name == base_fact.subject.name
                                and t.predicate.name == base_fact.predicate.name
                                and t.object.name == base_fact.object.name
                            ]
                            if matching_triples:
                                neg_triple = self._corrupt_triple_random(
                                    matching_triples[0], kg.individuals, original_triple=matching_triples[0]
                                )
                                if neg_triple:
                                    neg_triple.metadata["source_type"] = "base"

            if neg_triple and self.is_valid_negative(neg_triple):
                negative_triples.append(neg_triple)
                self.strategy_usage[strategy] += 1

        kg.triples.extend(negative_triples)

        # For memberships, we just use random corruption for now as it's less critical
        # TODO
        positive_memberships = [m for m in kg.memberships if m.is_member]
        n_neg_memberships = int(len(positive_memberships) * ratio)
        negative_memberships = []

        for _ in range(n_neg_memberships):
            if not positive_memberships:
                break
            pos_mem = random.choice(positive_memberships)
            neg_mem = self._corrupt_membership_random(
                pos_mem, list(self.schema_classes.values()), original_membership=pos_mem
            )
            if neg_mem and not self._is_positive_membership(neg_mem, kg):
                negative_memberships.append(neg_mem)

        kg.memberships.extend(negative_memberships)
        return kg

    # ==================== HELPER METHODS ==================== #

    def _check_premises_satisfied(self, rule, substitutions: Dict[Var, Term], corrupted_atom: Atom) -> bool:
        """
        Check if all premises of the rule are satisfied in the KG,
        EXCEPT the one corresponding to the corrupted atom.

        This ensures that the "negative goal" is derived from a chain
        where only one link is broken (the corruption), making it a
        consistent "hard negative".
        """
        for premise in rule.premises:
            # Substitute variables
            instantiated_premise = premise.substitute(substitutions)

            # Check if this premise matches the corrupted atom (the "broken link")
            if (
                instantiated_premise.predicate == corrupted_atom.predicate
                and instantiated_premise.subject == corrupted_atom.subject
                and instantiated_premise.object == corrupted_atom.object
            ):
                continue

            # Otherwise, this premise MUST exist in the KG
            key = f"{instantiated_premise.subject.name}|{instantiated_premise.predicate.name}|{instantiated_premise.object.name}"
            if key not in self.existing_triples:
                # One of the supportive facts is missing!
                # This means the rule logic falls apart not just because of corruption,
                # but because other preconditions aren't met.
                return False

        return True

    def _corrupt_triple_constrained(
        self,
        triple: Triple,
        individuals: List[Individual],
        ind_classes: Dict[Individual, Set[str]],
    ) -> Optional[Triple]:
        """Corrupts triple respecting domain/range constraints."""
        if random.random() < 0.5:
             # Corrupt subject
             candidates = self._get_domain_candidates(triple.predicate, individuals, ind_classes)
             candidates = [c for c in candidates if c != triple.subject]
             if candidates:
                 new_subj = random.choice(candidates)
                 return Triple(new_subj, triple.predicate, triple.object, positive=False, proofs=[])
        else:
             # Corrupt object
             candidates = self._get_range_candidates(triple.predicate, individuals, ind_classes)
             candidates = [c for c in candidates if c != triple.object]
             if candidates:
                 new_obj = random.choice(candidates)
                 return Triple(triple.subject, triple.predicate, new_obj, positive=False, proofs=[])
        return None

    def _corrupt_triple_random(
        self, triple: Triple, individuals: List[Individual], original_triple: Optional[Triple] = None
    ) -> Optional[Triple]:
        """Randomly corrupt subject or object of a triple."""
        if not individuals:
            return None

        if random.random() < 0.5:
            # Corrupt subject
            candidates = [i for i in individuals if i != triple.subject]
            if not candidates:
                return None
            new_subj = random.choice(candidates)
            metadata = {}
            if original_triple:
                metadata["source_type"] = "base" if original_triple.is_base_fact else "inferred"
            return Triple(new_subj, triple.predicate, triple.object, positive=False, proofs=[], metadata=metadata)
        else:
            # Corrupt object
            candidates = [i for i in individuals if i != triple.object]
            if not candidates:
                return None
            new_obj = random.choice(candidates)
            metadata = {}
            if original_triple:
                metadata["source_type"] = "base" if original_triple.is_base_fact else "inferred"
            return Triple(triple.subject, triple.predicate, new_obj, positive=False, proofs=[], metadata=metadata)

    def _generate_explanation(self, original: Triple, corrupted: Triple, context: str) -> str:
        """Helper to generate a human-readable explanation of the corruption."""

        def get_name(term):
            return getattr(term, "name", str(term))

        orig_s, orig_p, orig_o = get_name(original.subject), get_name(original.predicate), get_name(original.object)
        curr_s, curr_p, curr_o = get_name(corrupted.subject), get_name(corrupted.predicate), get_name(corrupted.object)

        diff = []
        if orig_s != curr_s:
            diff.append(f"Subject: {orig_s} -> {curr_s}")
        if orig_p != curr_p:
            diff.append(f"Predicate: {orig_p} -> {curr_p}")
        if orig_o != curr_o:
            diff.append(f"Object: {orig_o} -> {curr_o}")

        diff_str = ", ".join(diff)
        return f"{context}: {diff_str}"

    def _corrupt_membership_random(
        self, membership: Membership, classes: List[Class], original_membership: Optional[Membership] = None
    ) -> Optional[Membership]:
        """Randomly corrupt class membership."""
        if not classes:
            return None
        candidates = [c for c in classes if c != membership.cls]
        if not candidates:
            return None
        new_cls = random.choice(candidates)
        metadata = {}
        if original_membership:
            metadata["source_type"] = "base" if original_membership.is_base_fact else "inferred"
        return Membership(membership.individual, new_cls, is_member=False, proofs=[], metadata=metadata)

    def _create_propagated_proof(
        self,
        original_proof: Proof,
        original_base_fact: Atom,
        corrupted_base_fact: Atom,
        term_mapping: Dict[Term, Term],
    ) -> Optional[Proof]:
        """
        Recursively reconstructs a proof tree with corrupted values.
        """
        # Base case: this is the leaf we want to corrupt
        if original_proof.is_base_fact():
            if original_proof.goal == original_base_fact:
                return Proof(
                    goal=corrupted_base_fact,
                    rule=None,
                    sub_proofs=tuple(),
                    recursive_use_counts=original_proof.recursive_use_counts,
                    substitutions=original_proof.substitutions,
                    is_valid=False,
                    is_corrupted_leaf=True,
                    original_goal=original_base_fact,
                )
            # Other base facts remain unchanged
            return original_proof

        # Recursive step
        if original_proof.rule:
            new_sub_proofs = []
            changed = False

            # 1. Update Sub-proofs
            for sp in original_proof.sub_proofs:
                new_sp = self._create_propagated_proof(
                    sp, original_base_fact, corrupted_base_fact, term_mapping
                )
                new_sub_proofs.append(new_sp)
                if new_sp is not sp:
                    changed = True

            # 2. Update Substitutions
            new_substs = {}
            substs_changed = False
            for var, term in original_proof.substitutions.items():
                if term in term_mapping:
                    new_substs[var] = term_mapping[term]
                    substs_changed = True
                else:
                    new_substs[var] = term

            if changed or substs_changed:
                # 3. Recalculate Goal
                # We need to apply the new substitutions to the rule conclusion.
                # Since proof.substitutions uses renamed variables (e.g. X_12) and rule.conclusion uses 
                # original variables (e.g. X), we need to normalize the keys.
                
                normalized_new_substs = {}
                rule_vars = original_proof.rule.conclusion.get_variables()
                
                for var, term in new_substs.items():
                    # Heuristic: base name is part before first underscore
                    # This assumes standard naming: Name_ID
                    base_name = var.name.split("_")[0]
                    for rule_var in rule_vars:
                         if rule_var.name == base_name:
                             normalized_new_substs[rule_var] = term
                
                try:
                    new_goal = original_proof.rule.conclusion.substitute(normalized_new_substs)
                except Exception as e:
                     logger.warning(f"Failed to substitute new goal in propagation: {e}")
                     # Fallback to old goal (incorrect but prevents crash)
                     new_goal = original_proof.goal

                return Proof(
                    goal=new_goal,
                    rule=original_proof.rule,
                    sub_proofs=tuple(new_sub_proofs),
                    recursive_use_counts=original_proof.recursive_use_counts,
                    substitutions=new_substs,
                    is_valid=False,  # Invalid because a child is invalid (or corrupted)
                )

        return original_proof

    def _build_individual_classes_map(self, kg: KnowledgeGraph) -> Dict[Individual, Set[str]]:
        """Build mapping from individuals to their classes."""
        ind_classes = defaultdict(set)
        for mem in kg.memberships:
            if mem.is_member:
                ind_classes[mem.individual].add(mem.cls.name)
        return ind_classes

    def _get_domain_candidates(
        self,
        relation: Relation,
        individuals: List[Individual],
        ind_classes: Dict[Individual, Set[str]],
    ) -> List[Individual]:
        """Get individuals that satisfy domain constraints for a relation."""
        required_classes = self.domains.get(relation.name, set())

        if not required_classes:
            return individuals

        candidates = []
        for ind in individuals:
            ind_cls = ind_classes.get(ind, set())
            if not required_classes.isdisjoint(ind_cls):
                candidates.append(ind)

        return candidates if candidates else individuals

    def _get_range_candidates(
        self,
        relation: Relation,
        individuals: List[Individual],
        ind_classes: Dict[Individual, Set[str]],
    ) -> List[Individual]:
        """Get individuals that satisfy range constraints for a relation."""
        required_classes = self.ranges.get(relation.name, set())

        if not required_classes:
            return individuals

        candidates = []
        for ind in individuals:
            ind_cls = ind_classes.get(ind, set())
            if not required_classes.isdisjoint(ind_cls):
                candidates.append(ind)

        return candidates if candidates else individuals

    def is_valid_negative(self, neg_triple: Triple) -> bool:
        """
        Check if a candidate negative triple is valid.

        A negative fact is valid ONLY if it does not appear in the positive deductive closure.
        Logic: is_valid_negative(candidate_triple) = candidate_triple NOT IN all_positive_triples
        """
        # Optimized O(1) lookup
        key = f"{neg_triple.subject.name}|{neg_triple.predicate.name}|{neg_triple.object.name}"
        # If it IS in existing triples, it's a positive fact, so it's NOT a valid negative.
        return key not in self.existing_triples

    def _is_positive_membership(self, neg_mem: Membership, kg: KnowledgeGraph) -> bool:
        """Check if a negative membership conflicts with a positive one."""
        # Optimized O(1) lookup
        key = f"{neg_mem.individual.name}|{neg_mem.cls.name}"
        return key in self.existing_memberships
