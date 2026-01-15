"""
DESCRIPTION:

    Knowledge Graph Generator (KGenerator).

    Core class for generating synthetic knowledge graphs from ontologies using
    backward chaining. Handles proof generation and fact collection.

AUTHOR

    Vincent Van Schependom
"""

import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rdflib.namespace import RDF

from ont_generator.chainer import BackwardChainer
from ont_generator.negative_sampler import NegativeSampler
from ont_generator.parse import OntologyParser
from synthology.data_structures import (
    Atom,
    Attribute,
    AttributeTriple,
    Class,
    Individual,
    KnowledgeGraph,
    LiteralValue,
    Membership,
    Proof,
    Relation,
    Term,
    Triple,
)

# ============================================================================ #
#                         SHARED UTILITY FUNCTIONS                             #
# ============================================================================ #


def extract_proof_map(proof: Proof) -> Dict[Atom, List[Proof]]:
    """
    Recursively extracts atoms and maps them to the proofs that derive them.

    Unlike extract_all_atoms_from_proof, this preserves Proof objects,
    allowing determination of whether facts are base or derived.

    Args:
        proof: Root proof to extract from

    Returns:
        Mapping from atoms to all proofs that derive them
    """
    proof_map = defaultdict(list)

    # Map goal of this proof node
    proof_map[proof.goal].append(proof)

    # Recursively collect from sub-proofs
    for sub_proof in proof.sub_proofs:
        sub_map = extract_proof_map(sub_proof)
        for atom, proofs in sub_map.items():
            proof_map[atom].extend(proofs)

    return proof_map


def extract_all_atoms_from_proof(proof: Proof) -> Set[Atom]:
    """
    Recursively extracts ALL atoms from a proof tree via DFS traversal.

    Includes:
    - The goal atom (conclusion)
    - All atoms from sub-proofs (premises)
    - Atoms at all levels (base facts + intermediate inferences)

    Args:
        proof: The proof tree to extract atoms from

    Returns:
        Set of all ground atoms in the proof tree
    """
    atoms: Set[Atom] = set()
    atoms.add(proof.goal)

    for sub_proof in proof.sub_proofs:
        atoms.update(extract_all_atoms_from_proof(sub_proof))

    return atoms


def atoms_to_knowledge_graph(
    atoms: Set[Atom],
    schema_classes: Dict[str, Class],
    schema_relations: Dict[str, Relation],
    schema_attributes: Dict[str, Attribute],
    proof_map: Optional[Dict[Atom, List[Proof]]] = None,
) -> KnowledgeGraph:
    """
    Converts a set of ground atoms into a KnowledgeGraph.

    Organizes atoms into:
    - Memberships: (Individual, rdf:type, Class)
    - Triples: (Individual, Relation, Individual)
    - AttributeTriples: (Individual, Attribute, LiteralValue)

    All facts are positive at this stage. Negatives added separately.

    Args:
        atoms: Set of ground atoms to convert
        schema_classes: Schema classes from parser
        schema_relations: Schema relations from parser
        schema_attributes: Schema attributes from parser
        proof_map: Optional mapping from atoms to their proofs

    Returns:
        Organized knowledge graph structure
    """
    individuals: Dict[str, Individual] = {}
    triples: Dict[tuple, Triple] = {}
    memberships: Dict[tuple, Membership] = {}
    attr_triples: Dict[tuple, AttributeTriple] = {}

    def register_individual(term) -> None:
        """Register an individual if not seen before."""
        if isinstance(term, Individual):
            if term.name not in individuals:
                individuals[term.name] = term

    # Convert each atom to appropriate KG structure
    for atom in atoms:
        s, p, o = atom.subject, atom.predicate, atom.object

        # Get proofs for this atom if available
        current_proofs = []
        if proof_map and atom in proof_map:
            current_proofs = proof_map[atom]

        # CLASS MEMBERSHIPS
        if p == RDF.type and isinstance(o, Class):
            register_individual(s)
            key = (s.name, o.name)
            if key not in memberships:
                assert isinstance(s, Individual)
                memberships[key] = Membership(s, o, True, proofs=[])
            if current_proofs:
                memberships[key].proofs.extend(current_proofs)

        # RELATIONAL TRIPLES
        elif isinstance(o, Individual) and isinstance(p, Relation):
            register_individual(s)
            register_individual(o)
            key = (s.name, p.name, o.name)
            if key not in triples:
                assert isinstance(s, Individual)
                triples[key] = Triple(s, p, o, True, proofs=[])
            if current_proofs:
                triples[key].proofs.extend(current_proofs)

        # ATTRIBUTES
        elif isinstance(p, Attribute):
            register_individual(s)
            key = (s.name, p.name, o)
            if key not in attr_triples:
                assert isinstance(s, Individual)
                assert isinstance(o, LiteralValue)
                attr_triples[key] = AttributeTriple(s, p, o, proofs=[])
            if current_proofs:
                attr_triples[key].proofs.extend(current_proofs)

    return KnowledgeGraph(
        attributes=list(schema_attributes.values()),
        classes=list(schema_classes.values()),
        relations=list(schema_relations.values()),
        individuals=list(individuals.values()),
        triples=list(triples.values()),
        memberships=list(memberships.values()),
        attribute_triples=list(attr_triples.values()),
    )


# ============================================================================ #
#                           CORE GENERATOR CLASS                               #
# ============================================================================ #


class KGenerator:
    """
    Knowledge Graph Generator.

    Orchestrates proof generation via backward chaining for synthetic knowledge graph creation.
    Designed to be called by create_data.py for train/test split generation.
    """

    def __init__(
        self,
        cfg: DictConfig,
        verbose: bool = True,
    ):
        """
        Initializes the Knowledge Graph Generator.

        Args:
            cfg (DictConfig):   Hydra configuration object.
            verbose (bool):     Print debug info.
        """
        # Flags
        self.verbose = verbose
        self.export_proof_visualizations = cfg.export_proofs

        # Parser initialization
        self.parser = OntologyParser(cfg.ontology.path)

        # Initialize backward chainer with constraints
        self.chainer = BackwardChainer(
            all_rules=self.parser.rules,
            cfg=cfg,
            constraints=self.parser.constraints,
            inverse_properties=self.parser.inverse_properties,
            domains=self.parser.domains,
            ranges=self.parser.ranges,
            verbose=verbose,
        )

        # Store schema references
        self.schema_classes = self.parser.classes  # Dict[str, Class], e.g. "Person" -> Class(...)
        self.schema_relations = self.parser.relations  # Dict[str, Relation], e.g. "knows" -> Relation(...)
        self.schema_attributes = self.parser.attributes  # Dict[str, Attribute], e.g. "age" -> Attribute(...)

    def generate_proofs_for_rule(
        self,
        rule_name: str,
        n_proof_roots: int = 1,
        max_proofs: Optional[int] = None,
    ) -> List[Proof]:
        """
        Generates proofs for a specific rule by name.

        Args:
            rule_name (str):    The name of the rule to generate proofs for.
            n_proof_roots (int):  Number of independent generation cycles (loops) to run for this rule.
            max_proofs (int):   Maximum number of total proofs to return.

        Returns:
            List[Proof]: A list of generated proof trees.
        """
        if rule_name not in self.chainer.all_rules:
            if self.verbose:
                logger.warning(f"Rule '{rule_name}' not found.")
            return []

        all_proofs = []

        # Outer loop: Restarts the search (creating new root individuals)
        for _ in range(n_proof_roots):
            # Stop completely if we hit the global cap
            if max_proofs and len(all_proofs) >= max_proofs:
                break

            # Consume generator ONE item at a time
            proof_generator = self.chainer.generate_proof_trees(rule_name)

            for proof in proof_generator:
                self.chainer.register_proof(proof)
                all_proofs.append(proof)

                # Check limit inside the inner loop
                if max_proofs and len(all_proofs) >= max_proofs:
                    break

        return all_proofs

    def generate_full_graph(self) -> KnowledgeGraph:
        """
        Generate complete knowledge graph with ALL derivable facts.

        This method is primarily for testing and verification. For train/test
        generation, use create_data.py which calls generate_proofs_for_rule()
        for controlled sample generation.

        NOTE: Output can be extremely large depending on ontology complexity.

        Returns:
            Complete knowledge graph with all positive facts
        """
        logger.info("Generating complete knowledge graph from all rules...")
        logger.info(f"Rules: {len(self.parser.rules)}, Constraints: {len(self.parser.constraints)}")

        all_rules = self.parser.rules
        if not all_rules:
            logger.warning("Warning: No rules found in ontology")
            return self._build_empty_kg()

        # Storage for facts with proof tracking
        individuals: Dict[str, Individual] = {}
        triples: Dict[tuple, Triple] = {}
        memberships: Dict[tuple, Membership] = {}
        attr_triples: Dict[tuple, AttributeTriple] = {}
        processed_proofs: Set[Proof] = set()

        # Track statistics
        stats = {"proofs_accepted": 0, "rules_without_proofs": 0}

        # Generate proofs from each rule as starting point
        for i, rule in enumerate(all_rules):
            if self.verbose:
                logger.info(f"\n[{i + 1}/{len(all_rules)}] Processing rule: {rule.name}")

            proof_generator = self.chainer.generate_proof_trees(rule.name)
            top_level_proofs = 0

            for top_level_proof in proof_generator:
                self.chainer.register_proof(top_level_proof)
                top_level_proofs += 1
                stats["proofs_accepted"] += 1

                # DFS traversal of proof tree (DAG)
                stack = [top_level_proof]
                while stack:
                    current_proof = stack.pop()

                    if current_proof in processed_proofs:
                        continue
                    processed_proofs.add(current_proof)

                    # Add goal to KG
                    self._add_atom_and_proof(
                        current_proof.goal,
                        current_proof,
                        individuals,
                        triples,
                        memberships,
                        attr_triples,
                    )

                    # Queue sub-proofs
                    stack.extend(current_proof.sub_proofs)

            if top_level_proofs == 0:
                stats["rules_without_proofs"] += 1
                if self.verbose:
                    logger.info(f"  No valid proofs for {rule.name}")

        logger.info("\nGeneration complete:")
        logger.info(f"  Proofs accepted: {stats['proofs_accepted']}")
        logger.info(f"  Rules without proofs: {stats['rules_without_proofs']}/{len(all_rules)}")

        return self._build_kg(individuals, triples, memberships, attr_triples)

    def _add_atom_and_proof(
        self,
        atom: Atom,
        proof: Proof,
        individuals: Dict[str, Individual],
        triples: Dict[tuple, Triple],
        memberships: Dict[tuple, Membership],
        attr_triples: Dict[tuple, AttributeTriple],
    ) -> None:
        """
        Convert a ground atom into a KG fact and add to storage.

        If fact exists, appends proof to track multiple derivation paths.

        Args:
            atom: Ground atom to add
            proof: Proof object that derived this atom
            individuals: Storage for individuals
            triples: Storage for relational triples
            memberships: Storage for class memberships
            attr_triples: Storage for attribute triples
        """
        if not atom.is_ground():
            if self.verbose:
                logger.warning(f"Warning: Skipping non-ground atom: {atom}")
            return

        s, p, o = atom.subject, atom.predicate, atom.object

        # CLASS MEMBERSHIPS
        if p == RDF.type and isinstance(o, Class):
            self._register_individual(s, individuals)
            key = (s.name, o.name)
            if key not in memberships:
                assert isinstance(s, Individual)
                memberships[key] = Membership(s, o, True, proofs=[])
            memberships[key].proofs.append(proof)

        # RELATIONAL TRIPLES
        elif isinstance(o, Individual) and isinstance(p, Relation):
            self._register_individual(s, individuals)
            self._register_individual(o, individuals)
            key = (s.name, p.name, o.name)
            if key not in triples:
                assert isinstance(s, Individual)
                triples[key] = Triple(s, p, o, True, proofs=[])
            triples[key].proofs.append(proof)

        # ATTRIBUTE TRIPLES
        elif isinstance(p, Attribute):
            self._register_individual(s, individuals)
            key = (s.name, p.name, o)
            if key not in attr_triples:
                assert isinstance(s, Individual)
                assert isinstance(o, LiteralValue)
                attr_triples[key] = AttributeTriple(s, p, o, proofs=[])
            attr_triples[key].proofs.append(proof)

    def _register_individual(self, term: Term, individuals: Dict[str, Individual]) -> None:
        """Add individual to storage if not present."""
        if isinstance(term, Individual) and term.name not in individuals:
            individuals[term.name] = term

    def _build_kg(
        self,
        individuals: Dict[str, Individual],
        triples: Dict[tuple, Triple],
        memberships: Dict[tuple, Membership],
        attr_triples: Dict[tuple, AttributeTriple],
    ) -> KnowledgeGraph:
        """Assemble final KnowledgeGraph from collected data."""
        return KnowledgeGraph(
            attributes=list(self.schema_attributes.values()),
            classes=list(self.schema_classes.values()),
            relations=list(self.schema_relations.values()),
            individuals=list(individuals.values()),
            triples=list(triples.values()),
            memberships=list(memberships.values()),
            attribute_triples=list(attr_triples.values()),
        )

    def _build_empty_kg(self) -> KnowledgeGraph:
        """Create empty knowledge graph with schema only."""
        return KnowledgeGraph(
            attributes=list(self.schema_attributes.values()),
            classes=list(self.schema_classes.values()),
            relations=list(self.schema_relations.values()),
            individuals=[],
            triples=[],
            memberships=[],
            attribute_triples=[],
        )


# ============================================================================ #
#                         COMMAND-LINE INTERFACE                               #
# ============================================================================ #


REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/ont_generator", config_name="config_single_graph")
def main(cfg: DictConfig):
    """
    Main entry point for full graph generation.

    For train/test split generation, use create_data.py instead.
    This script is for verification and testing on small ontologies.
    """
    logger.info(f"Running Ontology Knowledge Graph Generator with configuration:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.logging.level:
        logging.basicConfig(level=getattr(logging, cfg.logging.level))

    try:
        # Initialize generator
        # Initialize generator
        generator = KGenerator(
            cfg=cfg,
            verbose=(cfg.logging.level == "DEBUG"),
        )

        # Generate full graph
        kg = generator.generate_full_graph()

        # Add negative samples if requested
        # Using negative_sampling config from cfg
        neg_strategy = cfg.negative_sampling.strategy
        neg_ratio = cfg.negative_sampling.ratio
        corrupt_base_facts = cfg.negative_sampling.corrupt_base_facts

        if neg_strategy and neg_ratio > 0:
            logger.info(f"\nAdding negative samples (Strategy: {neg_strategy}, Ratio: {neg_ratio})...")

            # Initialize NegativeSampler
            # We need schema info which is in the generator
            sampler = NegativeSampler(
                schema_classes=generator.schema_classes,
                schema_relations=generator.schema_relations,
                cfg=cfg,
                domains=generator.parser.domains,
                ranges=generator.parser.ranges,
                verbose=(cfg.logging.level == "DEBUG"),
            )

            kg = sampler.add_negative_samples(
                kg,
                strategy=neg_strategy,
                ratio=neg_ratio,
                corrupt_base_facts=corrupt_base_facts,
                export_proofs=cfg.export_proofs,
                output_dir=cfg.get("proof_output_path", "proof-trees") if cfg.export_proofs else None,
            )

        # check if the kg is not too big
        if len(kg.triples) + len(kg.memberships) > 100:
            logger.warning("Warning: Generated knowledge graph is very large (>100 facts).")
            logger.warning("Not saving visualization to avoid performance issues.")
        else:
            kg.save_visualization(
                output_path=cfg.get("graph_output_path", "."),
                output_name="full_knowledge_graph",
                title="Complete Knowledge Graph",
                display_negatives=cfg.get("visualize_negatives", False),
            )

        # Print summary
        logger.info("\n--- Knowledge Graph Summary ---")
        logger.info("  Schema:")
        logger.info(f"    Classes:    {len(kg.classes)}")
        logger.info(f"    Relations:  {len(kg.relations)}")
        logger.info(f"    Attributes: {len(kg.attributes)}")
        logger.info("  Generated Data:")
        logger.info(f"    Individuals: {len(kg.individuals)}")
        logger.info(f"    Triples:     {len(kg.triples)}")
        logger.info(f"    Memberships: {len(kg.memberships)}")
        logger.info(f"    Attributes:  {len(kg.attribute_triples)}")
        logger.info("-------------------------------")

        if cfg.logging.level == "DEBUG":
            kg.print()

        if cfg.export_proofs:
            proof_out = cfg.get("proof_output_path", "proof-trees")
            logger.info(f"\nProof trees exported to: {proof_out}")

    except FileNotFoundError:
        logger.error(f"Error: Ontology file not found at '{cfg.ontology.path}'", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
