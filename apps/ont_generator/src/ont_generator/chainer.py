"""
DESCRIPTION

    Pure Python Backward-Chainer for proving goals based on rules and their premises.

AUTHOR

    Vincent Van Schependom
"""

import itertools
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple

from loguru import logger
from omegaconf import DictConfig
from rdflib.namespace import OWL, RDF

from synthology.data_structures import (
    Atom,
    Attribute,
    Class,
    Constraint,
    ExecutableRule,
    Individual,
    Proof,
    Relation,
    Term,
    Var,
)


class BackwardChainer:
    """
    Generates proof trees and synthetic data via backward chaining.

    This engine starts with a "goal rule" and tries to find all possible
    proof trees for its conclusion, generating new individuals along the way.
    """

    def __init__(
        self,
        all_rules: List[ExecutableRule],
        cfg: DictConfig,
        constraints: Optional[List[Constraint]] = None,
        inverse_properties: Optional[Dict[str, Set[str]]] = None,
        domains: Optional[Dict[str, Set[str]]] = None,
        ranges: Optional[Dict[str, Set[str]]] = None,
        verbose: bool = False,
        entity_prefix: str = "Ind_",
    ):
        """
        Initializes the chainer.

        Args:
            all_rules (List[ExecutableRule]):           All rules from the ontology parser.
            cfg (DictConfig):                           Hydra configuration object.
            constraints (List[Constraint]):             All constraints from the ontology parser.
            inverse_properties (Dict[str, Set[str]]):   Mapping of inverse properties.
            domains (Dict[str, Set[str]]):              Mapping of property domains.
            ranges (Dict[str, Set[str]]):               Mapping of property ranges.
            verbose (bool):                             Enable detailed debug output.
            entity_prefix (str):                        Prefix for generated individuals.
        """
        # Store rules, constraints and schema info
        self.all_rules = {rule.name: rule for rule in all_rules}
        self.constraints = constraints if constraints else []
        self.inverse_properties = inverse_properties if inverse_properties else {}
        self.domains = domains if domains else {}
        self.ranges = ranges if ranges else {}

        # Extract from config
        self.max_recursion_depth = cfg.generator.max_recursion
        self.global_max_depth = cfg.generator.global_max_depth
        self.max_proofs_per_atom = cfg.generator.max_proofs_per_atom
        self.use_signature_sampling = cfg.generator.use_signature_sampling
        self.export_proof_visualizations = cfg.export_proofs

        self.verbose = verbose

        # Individual pooling parameters
        self.individual_pool_size = cfg.generator.individual_pool_size
        self.individual_reuse_prob = cfg.generator.individual_reuse_prob
        self.always_generate_base_facts = cfg.generator.get("always_generate_base_facts", False)
        self.individual_pool: List[Individual] = []
        self._individual_counter = 0
        self.individual_name_prefix = entity_prefix

        # Track types of individuals across proofs (used for constraint checking)
        # E.g. "Ind_0" -> {"Person", "Employee"}
        self.committed_individual_types: Dict[Individual, Set[str]] = defaultdict(set)

        # Index rules and find recursive ones
        self.rules_by_head = self._index_rules(all_rules)
        self.recursive_rules = self._find_recursive_rules(all_rules)

        if self.recursive_rules and self.verbose:
            logger.debug(f"Identified {len(self.recursive_rules)} recursive rules")

        # Index constraints
        self._index_constraints()

        # Counter
        self._var_rename_counter = 0

        # Track functional property values (ensure uniqueness)
        # A functional property must have at most one value per subject
        self._functional_property_values: Dict[Tuple[Individual, Relation], Term] = {}

    def _index_constraints(self) -> None:
        """
        Index constraints by type for efficient lookup during constraint checking.

        Creates mappings:
        - disjoint_classes: Dict[Class, Set[Class]] - maps each class to classes it's disjoint with
        - irreflexive_properties: Set[Relation] - set of properties that cannot be reflexive
        - asymmetric_properties: Set[Relation] - set of properties where P(X,Y) forbids P(Y,X)
        - functional_properties: Set[Relation] - set of properties that must have unique values
        """
        self.disjoint_classes: Dict[Class, Set[Class]] = defaultdict(set)
        self.disjoint_class_names: Dict[str, Set[str]] = defaultdict(set)
        self.irreflexive_properties: Set[Relation] = set()
        self.asymmetric_properties: Set[Relation] = set()
        self.functional_properties: Set[Relation] = set()

        for constraint in self.constraints:
            if constraint.constraint_type == OWL.disjointWith:
                # constraint.terms = [Class1, Class2, Var('X')]
                # Represents: X cannot be both Class1 and Class2
                if len(constraint.terms) >= 2:
                    c1, c2 = constraint.terms[0], constraint.terms[1]
                    if isinstance(c1, Class) and isinstance(c2, Class):
                        self.disjoint_classes[c1].add(c2)
                        self.disjoint_classes[c2].add(c1)  # Symmetric
                        self.disjoint_class_names[c1.name].add(c2.name)
                        self.disjoint_class_names[c2.name].add(c1.name)

            elif constraint.constraint_type == OWL.IrreflexiveProperty:
                # constraint.terms = [Property, Var('X')]
                # Represents: (X, Property, X) is forbidden
                if len(constraint.terms) >= 1:
                    prop = constraint.terms[0]
                    if isinstance(prop, Relation):
                        self.irreflexive_properties.add(prop)

            elif constraint.constraint_type == OWL.AsymmetricProperty:
                # constraint.terms = [Property, Var('X'), Var('Y')]
                # Represents: (X P Y) and (Y P X) cannot both hold
                if len(constraint.terms) >= 1:
                    prop = constraint.terms[0]
                    if isinstance(prop, Relation):
                        self.asymmetric_properties.add(prop)

            elif constraint.constraint_type == OWL.FunctionalProperty:
                # constraint.terms = [Property, Var('X')]
                # Represents: X can have at most one value for Property
                if len(constraint.terms) >= 1:
                    prop = constraint.terms[0]
                    if isinstance(prop, Relation):
                        self.functional_properties.add(prop)

        if self.verbose:
            logger.debug("Constraint indexing complete:")
            logger.debug(f"  Disjoint class pairs: {sum(len(v) for v in self.disjoint_classes.values()) // 2}")
            logger.debug(f"  Irreflexive properties: {len(self.irreflexive_properties)}")
            logger.debug(f"  Asymmetric properties: {len(self.asymmetric_properties)}")
            logger.debug(f"  Functional properties: {len(self.functional_properties)}")

    def _get_atom_key(self, atom: Atom) -> Optional[Tuple]:
        """
        Creates a hashable key for an atom to index rules by their conclusions.

        Examples:
            Classes:    Atom(X, rdf:type, Person)     -> (rdf:type, 'Person')
            Relations:  Atom(X, hasParent, Y)         -> ('hasParent', None)
            Attributes: Atom(X, age, 25)              -> ('age', None)

        Args:
            atom (Atom): The atom to create a key for.

        Returns:
            Optional[Tuple]: A hashable key, or None if the predicate is a variable.
        """
        pred = atom.predicate
        obj = atom.object

        # Cannot index on a variable predicate
        if isinstance(pred, Var):
            return None

        # For class membership (rdf:type), key on (rdf:type, ClassName)
        if pred == RDF.type and not isinstance(obj, Var):
            if isinstance(obj, Class):
                return (RDF.type, obj.name)
            else:
                logger.warning(f"Unexpected object for rdf:type: {obj}")
                return (RDF.type, str(obj))

        # For relations and attributes, key on (PredicateName, None)
        else:
            if isinstance(pred, (Relation, Attribute)):
                return (pred.name, None)
            else:
                return (str(pred), None)

    def _index_rules(self, rules: List[ExecutableRule]) -> Dict[Tuple, List[ExecutableRule]]:
        """
        Indexes rules by their conclusion (head atom) for O(1) lookup.
        This allows us to quickly find all rules that could prove a given atom.

        Args:
            rules (List[ExecutableRule]):       List of all rules from the ontology.

        Returns:
            Dict[Tuple, List[ExecutableRule]]:  Mapping from atom keys to rules that have that atom as conclusion.

        E.g.:
            [ExecutableRule(conclusion=Atom(X, hasGrandparent, Y),
                            premises=[Atom(X, hasParent, Y), Atom(Y, hasParent, Z)]),
             ExecutableRule(...),] -> {('hasGrandparent', None): [ExecutableRule(...), ...], ...}
        """
        index: Dict[Tuple, List[ExecutableRule]] = defaultdict(list)

        for rule in rules:
            assert rule.conclusion is not None, "Rule conclusion cannot be None"
            key = self._get_atom_key(rule.conclusion)
            # -> E.g. ('hasGrandparent', None), ('rdf:type', 'Person'), etc.

            # key=None for variable predicate
            if key is not None:
                index[key].append(rule)
            else:
                logger.warning(f"Cannot index rule with variable predicate: {rule}")

        return index

    def _find_recursive_rules(self, all_rules: List[ExecutableRule]) -> Set[str]:
        """
        Finds all rules that are part of any recursive cycle.
        This includes direct recursion (A -> A) and mutual recursion (A -> B, B -> A).
        Uses DFS to detect cycles in the dependency graph.

        Args:
            all_rules (List[ExecutableRule]): All rules from the ontology.

        Returns:
            Set[str]: Set of rule names that are part of recursive cycles.
        """
        # Build a dependency graph: atom_key -> {atom_keys it depends on}
        graph: Dict[Tuple, Set[Tuple]] = defaultdict(set)

        for rule in all_rules:
            assert rule.conclusion is not None, "Rule conclusion cannot be None"
            head_key = self._get_atom_key(rule.conclusion)

            # Head can't be a Var
            if head_key is None:
                continue

            # Add edges to premises
            for premise in rule.premises:
                premise_key = self._get_atom_key(premise)
                if premise_key is not None:
                    graph[head_key].add(premise_key)

        # Use DFS to find all nodes that are part of a cycle
        recursive_keys: Set[Tuple] = set()
        visiting: Set[Tuple] = set()  # Currently in recursion stack
        visited: Set[Tuple] = set()  # Fully explored

        def dfs(key: Tuple):
            """DFS to detect cycles and mark all nodes in cycles."""
            visiting.add(key)

            for neighbor_key in graph.get(key, set()):
                if neighbor_key in visiting:
                    # Back edge found - cycle detected!
                    recursive_keys.add(key)
                elif neighbor_key not in visited:
                    dfs(neighbor_key)
                    # Propagate "recursive" status up the call stack
                    if neighbor_key in recursive_keys:
                        recursive_keys.add(key)

            visiting.remove(key)
            visited.add(key)

        # Run DFS from every unvisited node
        for key in list(graph.keys()):
            if key not in visited:
                dfs(key)

        # Map recursive atom keys to rule names
        recursive_rule_names: Set[str] = set()
        for rule in all_rules:
            assert rule.conclusion is not None, "Rule conclusion cannot be None"
            head_key = self._get_atom_key(rule.conclusion)
            if head_key in recursive_keys:
                recursive_rule_names.add(rule.name)

        return recursive_rule_names

    def _get_individual(self, reuse: bool = False) -> Individual:
        """
        Gets an individual from the pool or creates a new one.

        Args:
            reuse: Whether to attempt reuse from pool

        Returns:
            Individual: Either from pool or newly created
        """
        # Decide whether to reuse based on probability and pool availability
        if reuse and self.individual_pool and random.random() < self.individual_reuse_prob:
            return random.choice(self.individual_pool)

        # Create new individual
        idx = self._individual_counter
        self._individual_counter += 1
        ind = Individual(index=idx, name=f"{self.individual_name_prefix}{idx}")

        # Add to pool if not full
        if len(self.individual_pool) < self.individual_pool_size:
            self.individual_pool.append(ind)

        return ind

    def reset_individual_pool(self, name_prefix: str = "Ind_"):
        """Resets the individual pool (call between samples)."""
        self.individual_pool = []
        self._individual_counter = 0
        self._functional_property_values = {}
        self.committed_individual_types = defaultdict(set)
        self.individual_name_prefix = name_prefix

    def _rename_rule_vars(self, rule: ExecutableRule) -> ExecutableRule:
        """
        Creates a new rule with all variables renamed to be unique.
        This prevents variable name collisions when using the same rule multiple times.

        Example:
            Input:  (X, parent, Y) -> (X, grandparent, Z)
            Output: (X_1, parent, Y_1) -> (X_1, grandparent, Z_1)

        Args:
            rule (ExecutableRule): The rule to rename variables for.

        Returns:
            ExecutableRule: A new rule instance with renamed variables.
        """
        self._var_rename_counter += 1
        suffix = f"_{self._var_rename_counter}"
        var_map: Dict[Var, Var] = {}

        def get_renamed_var(v: Var) -> Var:
            """Get or create a renamed version of a variable."""
            if v not in var_map:
                var_map[v] = Var(v.name + suffix)
            return var_map[v]

        def rename_term(t: Term) -> Term:
            """Rename a term if it's a variable, otherwise return as-is."""
            return get_renamed_var(t) if isinstance(t, Var) else t

        # Rename conclusion
        renamed_conclusion = Atom(
            subject=rename_term(rule.conclusion.subject),
            predicate=rename_term(rule.conclusion.predicate),
            object=rename_term(rule.conclusion.object),
        )

        # Rename all premises
        renamed_premises = [
            Atom(
                subject=rename_term(p.subject),
                predicate=rename_term(p.predicate),
                object=rename_term(p.object),
            )
            for p in rule.premises
        ]

        return ExecutableRule(name=rule.name, conclusion=renamed_conclusion, premises=renamed_premises)

    def _unify(self, goal: Atom, pattern: Atom) -> Optional[Dict[Var, Term]]:
        """
        Attempts to unify a GROUND goal atom with a rule's conclusion pattern.
        Unification finds a substitution that makes the pattern match the goal.

        Example:
            goal    = Atom(Ind_A, hasParent, Ind_C)  [ground]
            pattern = Atom(X_1, hasParent, Y_1)      [has variables]
            result  = {X_1: Ind_A, Y_1: Ind_C}       [substitution]

        Args:
            goal (Atom):    A ground atom we want to prove.
            pattern (Atom): A rule conclusion with variables.

        Returns:
            Optional[Dict[Var, Term]]: A substitution mapping from variables to
                                       ground terms, or None if unification fails.

                                       None if unification fails.
        """
        subst: Dict[Var, Term] = {}

        def unify_terms(t1: Term, t2: Term) -> bool:
            """
            Unify two terms, updating the substitution dictionary.

            Args:
                t1 (Term): Ground term from goal.
                t2 (Term): Possibly variable term from pattern.

            Returns:
                bool: True if unification succeeds, False otherwise.
            """
            if isinstance(t2, Var):
                # t2 is a variable - try to bind it to t1
                if t2 in subst and subst[t2] != t1:
                    # Variable already bound to different value - unification fails
                    return False
                # Bind or confirm existing binding
                subst[t2] = t1
                return True
            else:
                # t2 is ground - must match t1 exactly
                return t1 == t2

        # Unify subject, predicate, and object
        if not unify_terms(goal.subject, pattern.subject):
            return None
        if not unify_terms(goal.predicate, pattern.predicate):
            return None
        if not unify_terms(goal.object, pattern.object):
            return None

        return subst

    def _check_constraints(self, collected_atoms: Set[Atom]) -> bool:
        """
        Checks if the collected atoms violate any constraints.

        This method verifies:
        1. DisjointWith: No individual belongs to two disjoint classes
        2. IrreflexiveProperty: No reflexive triples for irreflexive properties
        3. AsymmetricProperty: No P(X,Y) and P(Y,X) for asymmetric properties
        4. FunctionalProperty: Each subject has at most one value for functional properties

        ----------------
        Consider a proof tree that derives:
            - Atom(Ind_0, rdf:type, Person)
            - Atom(Ind_0, rdf:type, Building)

        If Person and Building are declared disjoint:
            :Person owl:disjointWith :Building .

        Then this proof violates the disjointWith constraint and should be rejected.
        ----------------

        Args:
            collected_atoms (Set[Atom]): All atoms derived in this proof tree.

        Returns:
            bool: True if proof is valid (no violations), False if constraints violated.
        """
        # Collect class memberships by individual
        # Dict[Individual, Set[Class]]: individual -> {classes it belongs to}
        individual_classes: Dict[Individual, Set[Class]] = defaultdict(set)

        # Collect relation triples by (subject, predicate)
        # Dict[Tuple[Individual, Relation], Set[Term]]: (subj, pred) -> {objects}
        property_values: Dict[Tuple[Individual, Relation], Set[Term]] = defaultdict(set)

        for atom in collected_atoms:
            # Collect class membership triples
            if atom.predicate == RDF.type and isinstance(atom.object, Class):
                if isinstance(atom.subject, Individual):
                    individual_classes[atom.subject].add(atom.object)

            # Collect relation triples
            elif isinstance(atom.predicate, Relation) and isinstance(atom.subject, Individual):
                property_values[(atom.subject, atom.predicate)].add(atom.object)
                
                # Inferred types are now handled by current_proof_types loop below

        # Helper to get all types for an individual from the current proof
        # merging explicit memberships and inferred domain/range types
        current_proof_types: Dict[Individual, Set[str]] = defaultdict(set)
        
        for atom in collected_atoms:
            if atom.predicate == RDF.type and isinstance(atom.object, Class):
                if isinstance(atom.subject, Individual):
                     current_proof_types[atom.subject].add(atom.object.name)
            
            elif isinstance(atom.predicate, Relation):
                if isinstance(atom.subject, Individual):
                     domains = self.domains.get(atom.predicate.name, set())
                     current_proof_types[atom.subject].update(domains)
                
                if isinstance(atom.object, Individual):
                     ranges = self.ranges.get(atom.predicate.name, set())
                     current_proof_types[atom.object].update(ranges)

        # ------------------------- CHECK DISJOINT CLASSES ------------------------- #

        # ------------------------- CHECK DISJOINT CLASSES ------------------------- #

        # ------------------------- CHECK DISJOINT CLASSES ------------------------- #

        for individual, proof_types in current_proof_types.items():
            # Get already committed types for this individual (if any)
            committed_types = self.committed_individual_types.get(individual, set())

            # Combine current proof types with committed types
            all_types = proof_types | committed_types

            for type_name in all_types:
                disjoint_with = self.disjoint_class_names.get(type_name, set())

                # Check if individual belongs to any disjoint class
                violation = all_types & disjoint_with
                if violation:
                    if self.verbose:
                        logger.debug(
                            f"  ✗ CONSTRAINT VIOLATION: {individual.name} cannot be both "
                            f"{type_name} and {next(iter(violation))} (disjoint classes)"
                        )
                    return False

        # ------------------------- CHECK IRREFLEXIVE PROPERTIES ------------------------- #
        for atom in collected_atoms:
            if isinstance(atom.predicate, Relation):
                if atom.predicate in self.irreflexive_properties:
                    # Check if subject equals object (reflexive triple)
                    if atom.subject == atom.object:
                        if self.verbose:
                            logger.debug(
                                f"CONSTRAINT VIOLATION: {atom} violates irreflexive property {atom.predicate.name}"
                            )
                        return False

        # ------------------------- CHECK ASYMMETRIC PROPERTIES ------------------------- #
        # For asymmetric properties, if P(X,Y) exists, P(Y,X) must not exist
        asymmetric_triples: Dict[Relation, Set[Tuple[Individual, Individual]]] = defaultdict(set)
        for atom in collected_atoms:
            if isinstance(atom.predicate, Relation) and atom.predicate in self.asymmetric_properties:
                if isinstance(atom.subject, Individual) and isinstance(atom.object, Individual):
                    # Check if the reverse triple already exists
                    if (atom.object, atom.subject) in asymmetric_triples[atom.predicate]:
                        if self.verbose:
                            logger.debug(
                                f"  ✗ CONSTRAINT VIOLATION: {atom.predicate.name}({atom.subject.name}, {atom.object.name}) "
                                f"and {atom.predicate.name}({atom.object.name}, {atom.subject.name}) "
                                f"both exist, violating asymmetry"
                            )
                        return False
                    asymmetric_triples[atom.predicate].add((atom.subject, atom.object))

        # ------------------------- CHECK FUNCTIONAL PROPERTIES ------------------------- #
        for (subject, predicate), objects in property_values.items():
            if predicate in self.functional_properties:
                # Functional property must have exactly one value per subject
                if len(objects) > 1:
                    if self.verbose:
                        obj_names = [o.name if hasattr(o, "name") else str(o) for o in objects]
                        logger.debug(
                            f"  ✗ CONSTRAINT VIOLATION: {subject.name} has multiple values "
                            f"for functional property {predicate.name}: {obj_names}"
                        )
                    return False

        # All constraints satisfied
        return True

    def _collect_all_atoms(self, proof: Proof) -> Set[Atom]:
        """Collects all atoms from a proof tree."""
        return proof.get_all_atoms()

    def generate_proof_trees(self, start_rule_name: str) -> Iterator[Proof]:
        """
        Main entry point to generate proof trees starting from a specific rule.

        This method generates all possible proof trees for the given rule's conclusion,
        creating new individuals as needed for variables, and CHECKING CONSTRAINTS
        to ensure only valid proofs are yielded.

        RUNNING EXAMPLE:
        ----------------
        Consider the rule for grandparent:
            hasParent(X, Y) ∧ hasParent(Y, Z) → hasGrandparent(X, Z)

        In OWL 2 RL:
            :hasGrandparent owl:propertyChainAxiom ( :hasParent :hasParent ) .

        Parsed as ExecutableRule:
            conclusion = Atom(Var('X'), hasGrandparent, Var('Z'))
            premises   = [Atom(Var('X'), hasParent, Var('Y')),
                         Atom(Var('Y'), hasParent, Var('Z'))]

        Args:
            start_rule_name (str): The name of the rule to use as starting point
                                   (e.g., "owl_chain_hasParent_hasParent_hasGrandparent").

        Yields:
            Proof: Complete, ground proof trees that satisfy all constraints.
        """
        if start_rule_name not in self.all_rules:
            logger.error(f"Rule '{start_rule_name}' not found.")
            return

        # Get the starting rule (unrenamed)
        start_rule = self.all_rules[start_rule_name]
        # Example: ExecutableRule(
        #   conclusion=Atom(Var('X'), hasGrandparent, Var('Z')),
        #   premises=[Atom(Var('X'), hasParent, Var('Y')),
        #             Atom(Var('Y'), hasParent, Var('Z'))]
        # )

        # Rename variables to avoid collisions
        rule = self._rename_rule_vars(start_rule)
        # Example after renaming: ExecutableRule(
        #   conclusion=Atom(Var('X_1'), hasGrandparent, Var('Z_1')),
        #   premises=[Atom(Var('X_1'), hasParent, Var('Y_1')),
        #             Atom(Var('Y_1'), hasParent, Var('Z_1'))]
        # )

        # Extract variables from the conclusion
        conclusion_vars = rule.conclusion.get_variables()
        # Example: {Var('X_1'), Var('Z_1')}

        if not conclusion_vars:
            # If the conclusion has no variables, it's already a ground fact
            logger.error(f"Rule '{start_rule_name}' conclusion has no variables. It is a fact.")
            return

        # Generate fresh individuals for all conclusion variables
        subst: Dict[Var, Term] = {}
        for var in conclusion_vars:
            subst[var] = self._get_valid_individual(var, subst, rule)
        # Example: subst = {
        #   Var('X_1'): Individual('Ind_0'),
        #   Var('Z_1'): Individual('Ind_1')
        # }

        # Create the ground goal we intend to prove
        ground_goal = rule.conclusion.substitute(subst)
        # Example: Atom(Individual('Ind_0'), hasGrandparent, Individual('Ind_1'))

        # Track recursive rule usage
        recursive_use_counts = frozenset()
        if rule.name in self.recursive_rules:
            # If the starting rule is recursive, count its first usage
            recursive_use_counts = frozenset([(rule.name, 1)])

        # Substitute known variables into premises
        premises_with_bound_vars = [p.substitute(subst) for p in rule.premises]
        # Example: [Atom(Individual('Ind_0'), hasParent, Var('Y_1')),
        #           Atom(Var('Y_1'), hasParent, Individual('Ind_1'))]

        # Find any remaining unbound variables in premises
        unbound_vars: Set[Var] = set()
        for p in premises_with_bound_vars:
            unbound_vars.update(p.get_variables())
        # Example: {Var('Y_1')}

        # Generate individuals for unbound variables
        for var in unbound_vars:
            if var not in subst:
                subst[var] = self._get_valid_individual(var, subst, rule)
        # Example: subst = {
        #   Var('X_1'): Individual('Ind_0'),
        #   Var('Z_1'): Individual('Ind_1'),
        #   Var('Y_1'): Individual('Ind_2')  # newly generated
        # }

        # Final grounding of premises
        ground_premises = [p.substitute(subst) for p in premises_with_bound_vars]
        # Example: [Atom(Individual('Ind_0'), hasParent, Individual('Ind_2')),
        #           Atom(Individual('Ind_2'), hasParent, Individual('Ind_1'))]

        # Handle zero-premise rules (axioms)
        if not ground_premises:
            logger.warning("Rule with no premises encountered.")
            proof = Proof.create_base_proof(ground_goal)
            # Check constraints before yielding
            atoms = self._collect_all_atoms(proof)
            if self._check_constraints(atoms):
                yield proof
            return

        # Track atoms in the current proof path to prevent circular reasoning
        atoms_in_path: frozenset[Atom] = frozenset([ground_goal])

        # Find proofs for all premises
        premise_sub_proof_iters = []
        failed_to_prove_a_premise = False

        for premise in ground_premises:
            proof_list = list(
                self._find_proofs_recursive(
                    premise, recursive_use_counts, atoms_in_path, depth=1, parent_predicate=ground_goal.predicate
                )
            )

            if not proof_list:
                # No proofs found for this premise
                failed_to_prove_a_premise = True
                break

            premise_sub_proof_iters.append(proof_list)

        if failed_to_prove_a_premise:
            return

        if failed_to_prove_a_premise:
            return

        # Yield a random sample of combinations (Monte Carlo) instead of exhaustive Product
        # This avoids the "first few items" bias of itertools.product and avoids O(N^M) explosion

        # Calculate total possible combinations
        total_combinations = 1
        for sub_iter in premise_sub_proof_iters:
            total_combinations *= len(sub_iter)

        # Determine strategy based on total size
        # If small, we can be exhaustive (or just use product)
        # If large, we MUST use sampling
        IS_HUGE = total_combinations > 10000

        proof_count = 0
        valid_proof_count = 0

        # If using signature sampling, we want DIVERSITY first, so we collect a buffer
        # If not, we just yield valid ones until we hit the limit

        # Generator for combinations
        def combination_generator():
            if not IS_HUGE:
                # Small enough to iterate deterministically (or if we really want all)
                for combo in itertools.product(*premise_sub_proof_iters):
                    yield list(combo)
            else:
                # Random sampling from the Cartesian space
                # We try to yield unique combinations if possible, but for huge spaces collisions are rare
                # We limit attempts to avoid infinite loops
                max_attempts = min(total_combinations, 5000)
                seen_indices = set()

                for _ in range(max_attempts):
                    # Pick random index for each premise
                    indices = tuple(random.randrange(len(sub_iter)) for sub_iter in premise_sub_proof_iters)
                    if indices in seen_indices:
                        continue
                    seen_indices.add(indices)

                    combo = [premise_sub_proof_iters[i][idx] for i, idx in enumerate(indices)]
                    yield combo

        # Use signature sampling (reservoir-ish)
        if self.use_signature_sampling:
            generated_proofs = []
            MAX_BUFFER = 500  # Collect this many valid candidate proofs before grouping

            for sub_proof_combination in combination_generator():
                complete_proof = Proof.create_derived_proof(
                    goal=ground_goal,
                    rule=start_rule,  # Use unrenamed rule
                    sub_proofs=sub_proof_combination,
                    substitutions=subst,
                )

                # Constraint check
                all_atoms = self._collect_all_atoms(complete_proof)
                if self._check_constraints(all_atoms):
                    generated_proofs.append(complete_proof)
                    if len(generated_proofs) >= MAX_BUFFER:
                        break

            # Group by signature
            signature_groups = defaultdict(list)
            for p in generated_proofs:
                sig = self._get_proof_signature(p)
                signature_groups[sig].append(p)

            # Yield one from each group
            keys = list(signature_groups.keys())
            random.shuffle(keys)  # Shuffle to yield random signatures

            for sig in keys:
                proofs_in_group = signature_groups[sig]
                yield random.choice(proofs_in_group)

        else:
            # Direct yielding logic
            for sub_proof_combination in combination_generator():
                complete_proof = Proof.create_derived_proof(
                    goal=ground_goal,
                    rule=start_rule,  # Use unrenamed rule
                    sub_proofs=sub_proof_combination,
                    substitutions=subst,
                )

                # Constraint check
                all_atoms = self._collect_all_atoms(complete_proof)
                if self._check_constraints(all_atoms):
                    valid_proof_count += 1

                    if self.export_proof_visualizations:
                        complete_proof.save_visualization(
                            root_label=start_rule_name,
                            filepath=f"proof_{valid_proof_count}",
                            format="pdf",
                            title=f"Proof #{valid_proof_count} for {ground_goal}",
                        )

                    yield complete_proof

                    # Stop if we have enough (caller controls this usually via islice, but good to check)
                    # Note: generate_proofs_for_rule handles the main count limit, so we just yield endlessly here
                    # unless we want to curb it per rule-instantiation.
                    # We rely on extraction logic to stop consuming.

    def register_proof(self, proof: Proof) -> None:
        """
        Registers a valid proof and updates individual type tracking.
        Should be called when a proof is accepted into the dataset.
        """
        atoms = self._collect_all_atoms(proof)
        for atom in atoms:
            # Track class memberships
            if atom.predicate == RDF.type and isinstance(atom.object, Class):
                if isinstance(atom.subject, Individual):
                    self.committed_individual_types[atom.subject].add(atom.object.name)

            # Track inferred types from domain/range
            elif isinstance(atom.predicate, Relation):
                if isinstance(atom.subject, Individual):
                    domains = self.domains.get(atom.predicate.name, set())
                    self.committed_individual_types[atom.subject].update(domains)

                if isinstance(atom.object, Individual):
                    ranges = self.ranges.get(atom.predicate.name, set())
                    self.committed_individual_types[atom.object].update(ranges)

    def check_proof(self, proof: Proof) -> bool:
        """
        Public method to check if a proof satisfies all constraints.
        Useful for re-validating proofs after committed types have changed.
        """
        atoms = self._collect_all_atoms(proof)
        return self._check_constraints(atoms)

    def _find_proofs_recursive(
        self,
        goal_atom: Atom,
        recursive_use_counts: frozenset[Tuple[str, int]],
        atoms_in_path: frozenset[Atom],
        depth: int = 0,
        parent_predicate: Optional[Term] = None,
    ) -> Iterator[Proof]:
        """
        Recursively finds all possible proof trees for a given ground atom.

        RUNNING EXAMPLE (continued):
        ----------------------------
        From the generate_proof_trees example, we now need to prove the premises:
            1. Atom(Individual('Ind_0'), hasParent, Individual('Ind_2'))
            2. Atom(Individual('Ind_2'), hasParent, Individual('Ind_1'))

        Let's trace the first premise:
            goal_atom = Atom(Individual('Ind_0'), hasParent, Individual('Ind_2'))

        This gets converted to a key for rule lookup:
            key = ('hasParent', None)

        We find matching rules, e.g.:
            ExecutableRule(conclusion = Atom(Var('X'), hasParent, Var('Y')),
                          premises   = [Atom(Var('Y'), hasChild, Var('X'))])

        After renaming:
            ExecutableRule(conclusion = Atom(Var('X_3'), hasParent, Var('Y_3')),
                          premises   = [Atom(Var('Y_3'), hasChild, Var('X_3'))])

        Unifying goal_atom with the renamed conclusion:
            subst = {Var('X_3'): Individual('Ind_0'),
                    Var('Y_3'): Individual('Ind_2')}

        Substituting into premises:
            [Atom(Individual('Ind_2'), hasChild, Individual('Ind_0'))]

        Then we recursively prove this new premise...

        Args:
            goal_atom (Atom):               Ground atom to prove.
            recursive_use_counts (frozenset): Tracks how many times each recursive
                                              rule has been used in this path.
            atoms_in_path (frozenset):      Atoms already in the current proof path
                                            (prevents circular reasoning).

        Yields:
            Proof: Valid proof trees for the goal_atom.
        """
        # Check global depth limit
        if depth > self.global_max_depth:
            if self.verbose:
                logger.debug(f"  ✗ Max global depth reached ({self.global_max_depth}) for goal: {goal_atom}")
            return

        # Check for circular reasoning
        if goal_atom in atoms_in_path:
            return  # Yield nothing - this would be circular

        yielded_count = 0

        # ------------------------- BASE CASE ------------------------- #
        # Allow this atom to be proven as a base fact
        # If always_generate_base_facts is False, we only yield base proof if no rules apply
        # (checked below) or if we want to allow hybrid (checked here).

        # We need to know if rules apply to decide if this is a "leaf" by necessity.
        key = self._get_atom_key(goal_atom)
        matching_rules = self.rules_by_head.get(key, []) if key is not None else []

        if self.always_generate_base_facts or not matching_rules:
            yield Proof.create_base_proof(goal_atom)
            yielded_count += 1
            if self.max_proofs_per_atom and yielded_count >= self.max_proofs_per_atom:
                return

        # ------------------------- RECURSIVE CASE ------------------------- #
        # Try to derive using rules

        # Add current atom to path BEFORE trying to derive it
        new_atoms_in_path = atoms_in_path | frozenset([goal_atom])

        # Find rules that could prove this atom
        key = self._get_atom_key(goal_atom)
        matching_rules = self.rules_by_head.get(key, []) if key is not None else []

        # Try each matching rule
        for original_rule in matching_rules:
            # Check recursion limits
            if original_rule.name in self.recursive_rules:
                current_recursive_uses = dict(recursive_use_counts).get(original_rule.name, 0)

                if current_recursive_uses >= self.max_recursion_depth:
                    if self.verbose:
                        logger.debug(f"Skipping {original_rule.name} (recursion limit)")
                    continue

                # Update recursion counter
                new_counts = dict(recursive_use_counts)
                new_counts[original_rule.name] = current_recursive_uses + 1
                new_recursive_use_counts = frozenset(new_counts.items())
            else:
                # Non-recursive rule - no change to counter
                new_recursive_use_counts = recursive_use_counts

            # INVERSE LOOP CHECK
            # If the rule's conclusion predicate is the inverse of the parent goal's predicate,
            # this is likely a trivial inverse loop step (e.g. hasParent -> hasChild -> hasParent).
            # We prune this branch to avoid redundant proofs.
            if parent_predicate and hasattr(original_rule.conclusion.predicate, "name"):
                rule_pred_name = original_rule.conclusion.predicate.name
                parent_pred_name = parent_predicate.name if hasattr(parent_predicate, "name") else str(parent_predicate)

                # Check if rule_pred is inverse of parent_pred
                if (
                    parent_pred_name in self.inverse_properties.get(rule_pred_name, set())  # type: ignore
                    if self.inverse_properties
                    else set()
                ):
                    if self.verbose:
                        logger.debug(
                            f"Skipping {original_rule.name} (inverse loop: {rule_pred_name} is inverse of {parent_pred_name})"
                        )
                    continue

            # Rename rule variables to avoid collisions
            rule = self._rename_rule_vars(original_rule)

            # Get all variables in this renamed rule for debug output
            rule_vars = set()
            rule_vars.update(rule.conclusion.get_variables())
            for premise in rule.premises:
                rule_vars.update(premise.get_variables())

            # Unify goal with rule conclusion
            assert rule.conclusion is not None, "Rule conclusion cannot be None"
            subst = self._unify(goal_atom, rule.conclusion)
            if subst is None:
                if self.verbose:
                    logger.debug(f"Unification between {goal_atom} and {rule.conclusion} failed for rule {rule.name}")
                continue

            # Check if the bound values satisfy the constraints of the new rule
            constraints_violated = False
            for var, term in subst.items():
                if isinstance(term, Individual):
                    required = self._get_required_classes(var, rule)
                    if not self._is_individual_compatible(term, required):
                        if self.verbose:
                            logger.debug(
                                f"Skipping {rule.name}: Bound individual {term.name} incompatible with requirements {required}"
                            )
                        constraints_violated = True
                        break

            if constraints_violated:
                continue

            # Create a new substitution dict for this rule's scope
            rule_subst = dict(subst)

            # Substitute into premises
            premises_with_bound_vars = [p.substitute(rule_subst) for p in rule.premises]

            # Find unbound variables
            unbound_vars: Set[Var] = set()
            for p in premises_with_bound_vars:
                unbound_vars.update(p.get_variables())

            # Handle functional properties and generate individuals
            for var in unbound_vars:
                if var not in rule_subst:
                    needs_functional_value = False
                    functional_key = None

                    for p in premises_with_bound_vars:
                        if (
                            isinstance(p.predicate, Relation)
                            and p.predicate in self.functional_properties
                            and p.object == var
                            and isinstance(p.subject, Individual)
                        ):
                            needs_functional_value = True
                            functional_key = (p.subject, p.predicate)
                            break

                    if needs_functional_value and functional_key in self._functional_property_values:
                        # REUSE: This subject already has a value for this property
                        rule_subst[var] = self._functional_property_values[functional_key]
                    else:
                        # CREATE NEW: Either not functional, or first time for this (subject, property)
                        new_ind = self._get_valid_individual(var, rule_subst, rule)
                        rule_subst[var] = new_ind

                        # STORE for future reuse if functional
                        if needs_functional_value and functional_key:
                            self._functional_property_values[functional_key] = new_ind

            # Final grounding of premises
            ground_premises = [p.substitute(rule_subst) for p in premises_with_bound_vars]

            # Handle zero-premise rules (axioms)
            if not ground_premises:
                yield Proof.create_derived_proof(
                    goal=goal_atom,
                    rule=original_rule,
                    sub_proofs=[],
                    substitutions=rule_subst,
                )
                continue

            # Try to prove all premises (with updated atoms_in_path)
            premise_sub_proof_iters = []
            failed_to_prove_a_premise = False

            for premise in ground_premises:
                proof_list = list(
                    self._find_proofs_recursive(
                        premise,
                        new_recursive_use_counts,
                        new_atoms_in_path,
                        depth=depth + 1,
                        parent_predicate=goal_atom.predicate,
                    )
                )

                if not proof_list:
                    # No proofs found for this premise
                    failed_to_prove_a_premise = True
                    break

                premise_sub_proof_iters.append(proof_list)

            if failed_to_prove_a_premise:
                continue  # Try next rule

            # Yield all combinations of sub-proofs (Cartesian product)
            # WITH PATH SIGNATURE SAMPLING (OPTIONAL)

            if self.use_signature_sampling:
                generated_proofs = []
                # BUFFER LIMIT: Prevent OOM if millions of combinations exist
                MAX_BUFFER_SIZE = 1000

                for idx, sub_proof_combination in enumerate(itertools.product(*premise_sub_proof_iters)):
                    if idx >= MAX_BUFFER_SIZE:
                        if self.verbose:
                            logger.warning(
                                f"  [WARN] Hit buffer limit ({MAX_BUFFER_SIZE}) for rule {original_rule.name}"
                            )
                        break

                    new_proof = Proof.create_derived_proof(
                        goal=goal_atom,
                        rule=original_rule,  # Use unrenamed rule
                        sub_proofs=list(sub_proof_combination),
                        substitutions=rule_subst,
                    )
                    generated_proofs.append(new_proof)

                # Group by signature
                signature_groups = defaultdict(list)
                for p in generated_proofs:
                    sig = self._get_proof_signature(p)
                    signature_groups[sig].append(p)

                # Sample one from each group
                for sig, proofs_in_group in signature_groups.items():
                    selected_proof = random.choice(proofs_in_group)
                    yield selected_proof
                    yielded_count += 1

                    if self.max_proofs_per_atom and yielded_count >= self.max_proofs_per_atom:
                        if self.verbose:
                            logger.debug(
                                f"Max proofs per atom reached ({self.max_proofs_per_atom}) for goal: {goal_atom}"
                            )
                        return
            else:
                # STANDARD GENERATION (No grouping, direct yield)
                for sub_proof_combination in itertools.product(*premise_sub_proof_iters):
                    yield Proof.create_derived_proof(
                        goal=goal_atom,
                        rule=original_rule,  # Use unrenamed rule
                        sub_proofs=list(sub_proof_combination),
                        substitutions=rule_subst,
                    )
                    yielded_count += 1

                    if self.max_proofs_per_atom and yielded_count >= self.max_proofs_per_atom:
                        if self.verbose:
                            logger.debug(
                                f"Max proofs per atom reached ({self.max_proofs_per_atom}) for goal: {goal_atom}"
                            )
                        return

    def _is_substitution_valid(self, subst: Dict[Var, Term], rule: ExecutableRule) -> bool:
        """
        Checks if the current partial substitution violates any constraints.
        Only checks fully ground atoms.
        """
        # Check all atoms in the rule
        for atom in [rule.conclusion] + rule.premises:
            # We need to substitute.
            # atom.substitute() returns a new Atom.
            ground_atom = atom.substitute(subst)

            # If the atom is fully ground (no variables), check constraints
            if ground_atom.is_ground():
                # Check Irreflexive
                if isinstance(ground_atom.predicate, Relation) and ground_atom.predicate in self.irreflexive_properties:
                    if ground_atom.subject == ground_atom.object:
                        # Violation!
                        return False

                # Check Asymmetric: look for reverse triple in the same rule
                if isinstance(ground_atom.predicate, Relation) and ground_atom.predicate in self.asymmetric_properties:
                    for other_atom in [rule.conclusion] + rule.premises:
                        other_ground = other_atom.substitute(subst)
                        if other_ground.is_ground() and other_ground != ground_atom:
                            if (
                                other_ground.predicate == ground_atom.predicate
                                and other_ground.subject == ground_atom.object
                                and other_ground.object == ground_atom.subject
                            ):
                                return False

                # Check FunctionalProperty (within the rule scope)
                if isinstance(ground_atom.predicate, Relation) and ground_atom.predicate in self.functional_properties:
                    # We need to check if we have multiple values for the same subject/predicate in this rule
                    # But we need to collect them first.
                    pass

        # Check FunctionalProperty consistency within the rule
        func_values = {}
        for atom in [rule.conclusion] + rule.premises:
            ground_atom = atom.substitute(subst)
            if (
                ground_atom.is_ground()
                and isinstance(ground_atom.predicate, Relation)
                and ground_atom.predicate in self.functional_properties
            ):
                key = (ground_atom.subject, ground_atom.predicate)
                if key in func_values:
                    if func_values[key] != ground_atom.object:
                        return False
                else:
                    func_values[key] = ground_atom.object

        return True

    def _get_valid_individual(self, var: Var, current_subst: Dict[Var, Term], rule: ExecutableRule) -> Term:
        """
        Tries to get an individual that doesn't violate constraints when added to substitution.

        Optimization: Uses forward-checking strategy.
        Instead of picking random individuals and hoping they work, we:
        1. Identify candidates from pool that satisfy basic type constraints (domain/range/disjointness).
        2. Pick from valid candidates.
        3. Verify complex rule-local constraints (Irreflexive, Functional).
        """
        required_classes = self._get_required_classes(var, rule)

        # Strategy 1: Attempt Reuse
        # Filter pool for candidates that satisfy "static" constraints (Types/Disjointness)
        # We only do this check if we typically reuse individuals
        if self.individual_pool and random.random() < self.individual_reuse_prob:
            # OPTIMIZATION: Filter first, then select
            # This avoids the wasteful "generate-and-fail" loop
            candidates = [
                ind
                for ind in self.individual_pool
                if self._is_individual_compatible(ind, required_classes)
            ]

            if candidates:
                # Shuffle to ensure randomness
                random.shuffle(candidates)

                # Try up to 10 candidates for the more expensive/complex checks
                for ind in candidates[:10]:
                    # Check 2: Rule-local constraints (Substitution consistency)
                    temp_subst = current_subst.copy()
                    temp_subst[var] = ind

                    # Check 3: Irreflexive property check (Linear scan of rule atoms)
                    # We can rely on _is_substitution_valid for this, but the separate check
                    # in the original code might have been for early exit.
                    # Let's use the robust _is_substitution_valid
                    if self._is_substitution_valid(temp_subst, rule):
                        return ind

        # Strategy 2: Create New Individual
        # If reuse failed or no candidates were compatible, create a new one.
        # A fresh individual has no committed types, so it (almost) satisfies everything
        # except potentially rule-local constraints (e.g. if rule says atom(X, prop, X) and prop is irreflexive)
        # But for a NEW individual X, it can't equal any other term in the rule unless we explicitly bind it so.
        return self._get_individual(reuse=False)

    def _get_required_classes(self, var: Var, rule: ExecutableRule) -> Set[str]:
        """Determine required classes for a variable based on domain/range usage in rule."""
        required = set()

        # Check usage in premises and conclusion
        # Atoms in premises imply requirements for the variable
        # Atoms in conclusion imply what we are asserting (if we are proving the conclusion, the subject/object MUST also satisfy domain/range)
        for atom in rule.premises + [rule.conclusion]:
            if isinstance(atom.predicate, Relation):
                if atom.subject == var:
                    required.update(self.domains.get(atom.predicate.name, set()))
                if atom.object == var:
                    required.update(self.ranges.get(atom.predicate.name, set()))
            elif atom.predicate == RDF.type and atom.subject == var:
                if isinstance(atom.object, Class):
                    required.add(atom.object.name)
        return required

    def _is_individual_compatible(self, ind: Individual, required_classes: Set[str]) -> bool:
        """Check if individual is compatible with required classes (no disjointness violation)."""
        existing_types = self.committed_individual_types.get(ind, set())

        # If no requirements or no existing types, compatible
        if not required_classes or not existing_types:
            return True

        for req_cls_name in required_classes:
            # Check if req_cls is disjoint with any existing type
            # disjoint_class_names maps class_name -> set of disjoint class names
            disjoint_with = self.disjoint_class_names.get(req_cls_name, set())

            # Intersection of existing types and disjoint set
            if not existing_types.isdisjoint(disjoint_with):
                if self.verbose:
                    conflict = existing_types.intersection(disjoint_with)
                    logger.warning(
                        f"[Constraint] Reuse rejected: {ind.name} is {existing_types}, required {req_cls_name} (disjoint with {conflict})"
                    )
                return False

        return True

    def _get_proof_signature(self, proof: Proof) -> Tuple[str, ...]:
        """
        Generates a signature for a proof based on the sequence of rules used.
        The signature is a sorted tuple of rule names involved in the proof.
        This ignores specific variable bindings, grouping structurally identical proofs.
        """
        rules = []
        if proof.rule:
            rules.append(proof.rule.name)

        for sub_proof in proof.sub_proofs:
            rules.extend(self._get_proof_signature(sub_proof))

        return tuple(sorted(rules))

    # def _format_atom(self, atom: Atom) -> str:
    #     """
    #     Helper to format atom for debug output.

    #     Args:
    #         atom (Atom): The atom to format.

    #     Returns:
    #         str: A human-readable representation like "(Ind_0, hasParent, Ind_1)".
    #     """
    #     s = atom.subject.name if hasattr(atom.subject, "name") else str(atom.subject)
    #     p = (
    #         atom.predicate.name
    #         if hasattr(atom.predicate, "name")
    #         else str(atom.predicate)
    #     )
    #     o = atom.object.name if hasattr(atom.object, "name") else str(atom.object)
    #     return f"({s}, {p}, {o})"
