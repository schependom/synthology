"""
DESCRIPTION:
    Validator for Knowledge Graph datasets.

    Verifies generated KGE datasets against:
    1. Logical consistency (no contradictions)
    2. Ontology constraints (disjointness, functional properties, etc.)
    3. Schema adherence (domain/range)
    4. Triviality (minimum size, connectivity)

AUTHOR:
    Vincent Van Schependom
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Set

import networkx as nx
from rdflib.namespace import OWL

from synthology.data_structures import Constraint, KnowledgeGraph


class Validator:
    """
    Validates KnowledgeGraph objects against ontology constraints and structural requirements.

    Extensibility:
    New checks can be added by defining a method `check_custom_thing(self, kg) -> List[str]`
    and adding it to `self.checks` list in `__init__`.
    """

    def __init__(
        self,
        constraints: List[Constraint],
        domains: Dict[str, Set[str]],
        ranges: Dict[str, Set[str]],
        verbose: bool = False,
    ):
        self.constraints = constraints
        self.domains = domains
        self.ranges = ranges
        self.verbose = verbose

        # Registry of validation checks
        # Each check should return a list of error strings (empty if passed)
        self.checks: List[Callable[[KnowledgeGraph], List[str]]] = [
            self.check_consistency,
            self.check_constraints,
            self.check_schema_adherence,
            self.check_triviality,
        ]

    def validate(self, kg: KnowledgeGraph) -> Dict[str, Any]:
        """
        Run all validation checks on the Knowledge Graph.

        Returns:
            Dict containing:
            - 'valid': bool
            - 'errors': List[str]
            - 'stats': Dict of validation statistics
        """
        all_errors = []

        for check in self.checks:
            errors = check(kg)
            if errors:
                all_errors.extend(errors)

        return {
            "valid": len(all_errors) == 0,
            "errors": all_errors,
            "stats": {
                "n_checks": len(self.checks),
                "n_constraints": len(self.constraints),
            },
        }

    def check_consistency(self, kg: KnowledgeGraph) -> List[str]:
        """
        Check for logical contradictions.
        E.g., A fact cannot be both positive and negative.
        """
        errors = []

        # Check triples
        positive_triples = set()
        negative_triples = set()

        for t in kg.triples:
            key = (t.subject.name, t.predicate.name, t.object.name)
            if t.positive:
                if key in negative_triples:
                    errors.append(f"Contradiction: Triple {key} exists as both positive and negative")
                positive_triples.add(key)
            else:
                if key in positive_triples:
                    errors.append(f"Contradiction: Triple {key} exists as both positive and negative")
                negative_triples.add(key)

        # Check memberships
        positive_mems = set()
        negative_mems = set()

        for m in kg.memberships:
            key = (m.individual.name, m.cls.name)
            if m.is_member:
                if key in negative_mems:
                    errors.append(f"Contradiction: Membership {key} exists as both positive and negative")
                positive_mems.add(key)
            else:
                if key in positive_mems:
                    errors.append(f"Contradiction: Membership {key} exists as both positive and negative")
                negative_mems.add(key)

        return errors

    def check_constraints(self, kg: KnowledgeGraph) -> List[str]:
        """
        Check ontology constraints (disjointness, functional, irreflexive).
        """
        errors = []

        # Pre-index for performance
        ind_classes = defaultdict(set)
        for m in kg.memberships:
            if m.is_member:
                ind_classes[m.individual.name].add(m.cls.name)

        triples_by_pred = defaultdict(list)
        for t in kg.triples:
            if t.positive:
                triples_by_pred[t.predicate.name].append(t)

        for constraint in self.constraints:
            # 1. Disjoint Classes
            if constraint.constraint_type == OWL.disjointWith:
                # terms: [ClassA, ClassB, Var('X')]
                c1_name = constraint.terms[0].name
                c2_name = constraint.terms[1].name

                for ind_name, classes in ind_classes.items():
                    if c1_name in classes and c2_name in classes:
                        errors.append(
                            f"Constraint Violation: {ind_name} is member of disjoint classes {c1_name} and {c2_name}"
                        )

            # 2. Irreflexive Property
            elif constraint.constraint_type == OWL.IrreflexiveProperty:
                # terms: [Property, Var('X')]
                prop_name = constraint.terms[0].name

                for t in triples_by_pred[prop_name]:
                    if t.subject.name == t.object.name:
                        errors.append(
                            f"Constraint Violation: Irreflexive property {prop_name} relates {t.subject.name} to itself"
                        )

            # 3. Functional Property
            elif constraint.constraint_type == OWL.FunctionalProperty:
                # terms: [Property, Var('X'), Var('Y')]
                prop_name = constraint.terms[0].name

                # Check uniqueness: subject -> {objects}
                subj_objects = defaultdict(set)
                for t in triples_by_pred[prop_name]:
                    subj_objects[t.subject.name].add(t.object.name)

                for subj, objs in subj_objects.items():
                    if len(objs) > 1:
                        errors.append(
                            f"Constraint Violation: Functional property {prop_name} has multiple values for {subj}: {objs}"
                        )

        return errors

    def check_schema_adherence(self, kg: KnowledgeGraph) -> List[str]:
        """
        Check if positive triples respect domain and range constraints.
        """
        errors = []

        # Pre-index memberships
        ind_classes = defaultdict(set)
        for m in kg.memberships:
            if m.is_member:
                ind_classes[m.individual.name].add(m.cls.name)

        for t in kg.triples:
            if not t.positive:
                continue

            prop_name = t.predicate.name
            subj_name = t.subject.name
            obj_name = t.object.name

            # Check Domain
            if prop_name in self.domains:
                required_domains = self.domains[prop_name]
                # Subject must be instance of AT LEAST ONE domain class (union semantics usually, but here strict)
                # Actually, RDFS domain implies intersection if multiple domains are defined.
                # But usually we have one domain per property in simple ontologies.
                # Let's check if subject has ALL required domain classes.

                subj_classes = ind_classes[subj_name]
                for domain_cls in required_domains:
                    if domain_cls not in subj_classes:
                        errors.append(f"Schema Violation: Subject {subj_name} of {prop_name} is not type {domain_cls}")

            # Check Range
            if prop_name in self.ranges:
                required_ranges = self.ranges[prop_name]
                obj_classes = ind_classes[obj_name]
                for range_cls in required_ranges:
                    if range_cls not in obj_classes:
                        errors.append(f"Schema Violation: Object {obj_name} of {prop_name} is not type {range_cls}")

        return errors

    def check_triviality(self, kg: KnowledgeGraph) -> List[str]:
        """
        Check for trivial datasets (too small, disconnected).
        """
        errors = []

        # Minimum Size
        if len(kg.individuals) < 2:
            errors.append("Triviality: Fewer than 2 individuals")
        if len(kg.triples) == 0 and len(kg.memberships) == 0:
            errors.append("Triviality: Empty graph (no facts)")

        # Connectivity (using NetworkX)
        if len(kg.individuals) > 0:
            G = nx.Graph()
            for ind in kg.individuals:
                G.add_node(ind.name)
            for t in kg.triples:
                if t.positive:
                    G.add_edge(t.subject.name, t.object.name)

            # Check connected components
            if len(G.nodes) > 0:
                components = list(nx.connected_components(G))
                largest_cc_size = max(len(c) for c in components)
                ratio = largest_cc_size / len(G.nodes)

                # Check if LCC is too small (graph too fragmented)
                # We want at least 80% of nodes to be in the main component for a healthy KG
                if ratio < 0.8:
                    errors.append(
                        f"Fragmented Graph: Graph is fragmented (LCC covers {ratio:.1%} of {len(G.nodes)} nodes). Ideal > 80%"
                    )

        return errors
