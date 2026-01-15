"""
DESCRIPTION:

    Data structures for
        - Knowledge Graph representation
        - Rules and Constraints
        - Proofs and Backward Chaining

    This replaces the proprietary reldata format with standard Python classes (RRN KGE model)
    and facilitates backward chaining operations.

AUTHOR:

    Vincent Van Schependom
"""

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rdflib.namespace import RDF
from rdflib.term import Literal, URIRef

# ---------------------------------------------------------------------------- #
#                                     TYPES                                    #
# ---------------------------------------------------------------------------- #

LiteralValue = Union[str, int, float, bool, Literal]
Term = Union["Var", "Individual", "Class", "Relation", "Attribute", URIRef, LiteralValue]

# ---------------------------------------------------------------------------- #
#                                      KGE                                     #
# ---------------------------------------------------------------------------- #


@dataclass
class Class:
    """Represents a class in the knowledge graph."""

    index: int
    name: str

    def __eq__(self, other):
        if not isinstance(other, Class):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name


@dataclass
class Relation:
    """Represents a relation type in the knowledge graph."""

    index: int
    name: str

    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name


@dataclass
class Attribute:
    """Represents an attribute type (datatype property)."""

    index: int
    name: str

    def __eq__(self, other):
        if not isinstance(other, Attribute):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name


@dataclass
class Individual:
    """Represents an individual entity in the knowledge graph."""

    index: int
    name: str

    # We initialize classes as a list, but will store Membership objects here
    classes: List["Membership"] = field(default_factory=list)

    def get_class_memberships(self) -> Set[Class]:
        """Helper to get all classes this individual is a member of."""
        return {m.cls for m in self.classes if m.is_member}

    def __eq__(self, other):
        if not isinstance(other, Individual):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name


@dataclass
class Membership:
    """
    Represents class membership of an individual.
    This fact can be a base fact (proofs=[]) or an inferred fact (proofs=[...]).
    """

    individual: Individual
    cls: Class
    is_member: bool  # True if member, False if explicitly not a member

    # Keep track of all proofs leading to this membership fact
    proofs: List["Proof"] = field(default_factory=list)

    # Metadata for tracking origin (e.g., negative sampling source)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_inferred(self) -> bool:
        """A fact is inferred if it has at least one derived proof."""
        if not self.proofs:
            return False
        # Check if any proof is a derived proof (not a base fact leaf)
        return any(p.rule is not None for p in self.proofs)

    @property
    def is_base_fact(self) -> bool:
        """A fact is a base fact if it has no proofs or only a base proof."""
        return not self.is_inferred

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.individual.name, "rdf:type", self.cls.name))

    def to_atom(self) -> "Atom":
        """Converts this fact to a ground Atom (Atom)."""
        return Atom(self.individual, RDF.type, self.cls)

    def __repr__(self) -> str:
        status = "Inferred" if self.is_inferred else "Base"
        if self.is_member:
            return f"<{self.individual}, memberOf, {self.cls}> [{status}]"
        else:
            return f"<{self.individual}, ~memberOf, {self.cls}> [{status}]"


@dataclass
class Triple:
    """
    Represents a relational triple (subject, predicate, object).
    This fact can be a base fact (proofs=[]) or an inferred fact (proofs=[...]).
    """

    subject: Individual
    predicate: Relation
    object: Individual
    positive: bool  # True for positive predicate, False for negated predicate

    # Keep track of all proofs leading to this triple fact
    proofs: List["Proof"] = field(default_factory=list)

    # Metadata for tracking origin (e.g., negative sampling source)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_inferred(self) -> bool:
        """A fact is inferred if it has at least one derived proof."""
        if not self.proofs:
            return False
        return any(p.rule is not None for p in self.proofs)

    @property
    def is_base_fact(self) -> bool:
        """A fact is a base fact if it has no proofs or only a base proof."""
        return not self.is_inferred

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.subject.name, self.predicate.name, self.object.name))

    def to_atom(self) -> "Atom":
        """Converts this fact to a ground Atom (Atom)."""
        return Atom(self.subject, self.predicate, self.object)

    def __repr__(self) -> str:
        status = "Inferred" if self.is_inferred else "Base"
        if self.positive:
            return f"<{self.subject}, {self.predicate}, {self.object}> [{status}]"
        else:
            return f"<{self.subject}, ~{self.predicate}, {self.object}> [{status}]"


@dataclass
class AttributeTriple:
    """
    Represents an attribute triple (subject, predicate, value).
    This fact can be a base fact or an inferred fact.
    """

    subject: Individual
    predicate: Attribute  # E.g. age, height
    value: LiteralValue  # The literal value

    # Keep track of all proofs leading to this attribute triple fact
    proofs: List["Proof"] = field(default_factory=list)

    @property
    def is_inferred(self) -> bool:
        """A fact is inferred if it has at least one derived proof."""
        if not self.proofs:
            return False
        return any(p.rule is not None for p in self.proofs)

    @property
    def is_base_fact(self) -> bool:
        """A fact is a base fact if it has no proofs or only a base proof."""
        return not self.is_inferred

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.subject.name, self.predicate.name, self.value))

    def to_atom(self) -> "Atom":
        """Converts this fact to a ground Atom (Atom)."""
        return Atom(self.subject, self.predicate, self.value)

    def __repr__(self) -> str:
        status = "Inferred" if self.is_inferred else "Base"
        return f"<{self.subject}, {self.predicate}, {self.value}> [{status}]"


@dataclass
class KnowledgeGraph:
    """
    Complete knowledge graph.
    """

    attributes: List[Attribute]
    classes: List[Class]
    relations: List[Relation]
    individuals: List[Individual]
    triples: List[Triple]
    memberships: List[Membership]
    attribute_triples: List[AttributeTriple]

    def print(self) -> None:
        """Prints a summary of the knowledge graph."""
        print("Knowledge Graph Summary:")
        print(f"  Nb of Attributes: {len(self.attributes)}")
        print(f"  Nb of Classes: {len(self.classes)}")
        print(f"  Nb of Relations: {len(self.relations)}")
        print(f"  Nb of Individuals: {len(self.individuals)}")
        print(f"  Nb of Triples: {len(self.triples)}")
        print(f"  Nb of Memberships: {len(self.memberships)}")
        print(f"  Nb of Attribute Triples: {len(self.attribute_triples)}\n")
        print("Triples:")
        for triple in self.triples:
            print(f"    {triple}")
        print("Memberships:")
        for membership in self.memberships:
            print(f"    {membership}")
        print("Attribute Triples:")
        for attr_triple in self.attribute_triples:
            print(f"    {attr_triple}")

    def to_csv(self, file_path: str) -> None:
        """
        Saves the knowledge graph to a CSV file.

        Each row represents one fact (triple, membership, or attribute).
        This format is suitable for loading into RRN models.

        CSV FORMAT:
        ----------
        subject,predicate,object,label,fact_type

        EXAMPLES:
        --------
        Ind_0,hasParent,Ind_1,1,triple          # Positive relation
        Ind_0,hasParent,Ind_3,0,triple          # Negative relation
        Ind_0,rdf:type,Person,1,membership      # Class membership
        Ind_0,age,25,1,attribute                # Attribute value

        Args:
            file_path (str): Path to save CSV file.
        """
        rows = []

        # Convert memberships to CSV rows
        for membership in self.memberships:
            rows.append(
                {
                    "subject": membership.individual.name,
                    "predicate": "rdf:type",
                    "object": membership.cls.name,
                    "label": "1" if membership.is_member else "0",
                    "fact_type": "membership",
                    "is_inferred": "1" if membership.is_inferred else "0",
                }
            )

        # Convert triples to CSV rows
        for triple in self.triples:
            rows.append(
                {
                    "subject": triple.subject.name,
                    "predicate": triple.predicate.name,
                    "object": triple.object.name,
                    "label": "1" if triple.positive else "0",
                    "fact_type": "triple",
                    "is_inferred": "1" if triple.is_inferred else "0",
                }
            )

        # Convert attribute triples to CSV rows
        for attr_triple in self.attribute_triples:
            rows.append(
                {
                    "subject": attr_triple.subject.name,
                    "predicate": attr_triple.predicate.name,
                    "object": str(attr_triple.value),  # Convert literal to string
                    "label": "1",  # Attributes are always positive
                    "fact_type": "attribute",
                    "is_inferred": "1" if attr_triple.is_inferred else "0",
                }
            )

        # Write to CSV
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            if rows:
                fieldnames = ["subject", "predicate", "object", "label", "fact_type", "is_inferred"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    @classmethod
    def from_csv(cls, file_path: str) -> "KnowledgeGraph":
        """
        Loads a knowledge graph from a CSV file.

        This method reconstructs a KG from the CSV format created by to_csv().
        It rebuilds:
            - All individuals mentioned in the CSV
            - Schema elements (classes, relations, attributes)
            - All facts (triples, memberships, attributes)

        NOTE: The schema is inferred from the data in the CSV.
            For full schema with all possible classes/relations,
            look at ALL csv's.

        Args:
            file_path (str): Path to CSV file.

        Returns:
            KnowledgeGraph: Reconstructed knowledge graph.
        """
        # Storage for reconstructed elements
        individuals: Dict[str, Individual] = {}
        classes: Dict[str, Class] = {}
        relations: Dict[str, Relation] = {}
        attributes: Dict[str, Attribute] = {}

        triples: List[Triple] = []
        memberships: List[Membership] = []
        attribute_triples: List[AttributeTriple] = []

        # Index counters for creating new objects
        ind_counter = 0
        cls_counter = 0
        rel_counter = 0
        attr_counter = 0

        def get_or_create_individual(name: str) -> Individual:
            """Get or create an Individual by name."""
            nonlocal ind_counter
            if name not in individuals:
                individuals[name] = Individual(index=ind_counter, name=name)
                ind_counter += 1
            return individuals[name]

        def get_or_create_class(name: str) -> Class:
            """Get or create a Class by name."""
            nonlocal cls_counter
            if name not in classes:
                classes[name] = Class(index=cls_counter, name=name)
                cls_counter += 1
            return classes[name]

        def get_or_create_relation(name: str) -> Relation:
            """Get or create a Relation by name."""
            nonlocal rel_counter
            if name not in relations:
                relations[name] = Relation(index=rel_counter, name=name)
                rel_counter += 1
            return relations[name]

        def get_or_create_attribute(name: str) -> Attribute:
            """Get or create an Attribute by name."""
            nonlocal attr_counter
            if name not in attributes:
                attributes[name] = Attribute(index=attr_counter, name=name)
                attr_counter += 1
            return attributes[name]

        # Read CSV file
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                subject_name = row["subject"]
                predicate_name = row["predicate"]
                object_name = row["object"]
                label = row["label"] == "1"  # Convert to boolean
                fact_type = row["fact_type"]
                is_inferred = row.get("is_inferred", "0") == "1"

                # Helper to create dummy proof if inferred
                proofs = []
                if is_inferred:
                    # Create a dummy proof so that .is_inferred property returns True
                    # We need a dummy rule and goal
                    dummy_rule = ExecutableRule(name="LOADED_FROM_CSV", conclusion=None, premises=[])
                    # Goal is not strictly needed for is_inferred check, but good for completeness
                    # We'll just leave it None or minimal for now as we don't have the Atom easily available here without reconstruction
                    # The property only checks: any(p.rule is not None for p in self.proofs)
                    dummy_atom = Atom("dummy_s", "dummy_p", "dummy_o")
                    dummy_proof = Proof(goal=dummy_atom, rule=dummy_rule, sub_proofs=tuple())
                    proofs.append(dummy_proof)

                # Reconstruct based on fact type
                if fact_type == "membership":
                    # Class membership: (Individual, rdf:type, Class)
                    individual = get_or_create_individual(subject_name)
                    cls_obj = get_or_create_class(object_name)

                    membership = Membership(
                        individual=individual,
                        cls=cls_obj,
                        is_member=label,
                        proofs=list(proofs),
                    )
                    memberships.append(membership)
                    individual.classes.append(membership)

                elif fact_type == "triple":
                    # Relational triple: (Individual, Relation, Individual)
                    subject = get_or_create_individual(subject_name)
                    predicate = get_or_create_relation(predicate_name)
                    obj = get_or_create_individual(object_name)

                    triple = Triple(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        positive=label,
                        proofs=list(proofs),
                    )
                    triples.append(triple)

                elif fact_type == "attribute":
                    # Attribute triple: (Individual, Attribute, LiteralValue)
                    subject = get_or_create_individual(subject_name)
                    predicate = get_or_create_attribute(predicate_name)

                    # Try to infer literal type
                    value = cls._parse_literal_value(object_name)

                    attr_triple = AttributeTriple(
                        subject=subject,
                        predicate=predicate,
                        value=value,
                        proofs=list(proofs),
                    )
                    attribute_triples.append(attr_triple)

        # Create and return knowledge graph
        return cls(
            attributes=list(attributes.values()),
            classes=list(classes.values()),
            relations=list(relations.values()),
            individuals=list(individuals.values()),
            triples=triples,
            memberships=memberships,
            attribute_triples=attribute_triples,
        )

    @staticmethod
    def _parse_literal_value(value_str: str):
        """
        Attempts to parse a string literal value to its appropriate type.

        Tries in order: int, float, bool, string

        Args:
            value_str (str): String representation of the value.

        Returns:
            Parsed value (int, float, bool, or str).
        """
        # Try int
        try:
            return int(value_str)
        except ValueError:
            pass

        # Try float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Try bool
        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"

        # Default to string
        return value_str

    def to_csv_batch(
        samples: List["KnowledgeGraph"],
        output_dir: str,
        prefix: str = "sample",
    ) -> None:
        """
        Batch save multiple KG samples to CSV files.

        Convenience method for saving entire datasets.

        Args:
            samples (List[KnowledgeGraph]): List of KG samples.
            output_dir (str): Directory to save files.
            prefix (str): Prefix for file names.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for idx, kg in enumerate(samples):
            file_path = output_path / f"{prefix}_{idx:05d}.csv"
            kg.to_csv(str(file_path))

    @classmethod
    def from_csv_batch(
        cls,
        input_dir: str,
        prefix: str = "sample",
        n_samples: Optional[int] = None,
    ) -> List["KnowledgeGraph"]:
        """
        Batch load multiple KG samples from CSV files.

        Convenience method for loading entire datasets.

        Args:
            input_dir (str): Directory containing CSV files.
            prefix (str): Prefix of file names to load.
            n_samples (int): Max number to load (None = all).

        Returns:
            List[KnowledgeGraph]: Loaded KG samples.
        """
        input_path = Path(input_dir)
        pattern = f"{prefix}_*.csv"
        csv_files = sorted(input_path.glob(pattern))

        if n_samples is not None:
            csv_files = csv_files[:n_samples]

        samples = []
        for file_path in csv_files:
            kg = cls.from_csv(str(file_path))
            samples.append(kg)

        return samples

    def save_visualization(
        self,
        output_path: str,
        output_name: str,
        format: str = "pdf",
        title: Optional[str] = None,
        display_negatives: bool = False,
    ) -> None:
        """
        Save knowledge graph visualization to file.

        Args:
            output_path: Output file path (without extension)
            format: Output format ("pdf", "png", "svg")
            title: Optional title for the graph
        """
        try:
            from ont_generator.utils.graph_visualizer import GraphVisualizer  # type: ignore
        except ImportError:
            print("Warning: ont_generator.utils.graph_visualizer not available. Skipping visualization.")
            return

        if len(self.individuals) > 100:
            print(f"Graph too large to visualize ({len(self.individuals)} individuals). Skipping visualization.")
            return

        visualizer = GraphVisualizer(output_dir=output_path)
        visualizer.visualize(self, filename=output_name, title=title, display_negatives=display_negatives)


@dataclass
class DataType(Enum):
    """
    Specifies what type of data to use in the KGE model.
    This is for handling the data generated by the ASP solver in the original RRN paper.
    """

    INF = 1  # inferred facts
    SPEC = 2  # specific (base) facts
    ALL = 3  # all facts


# ---------------------------------------------------------------------------- #
#                               BACKWARD CHAINING                              #
# ---------------------------------------------------------------------------- #

"""
EXAMPLE

Rules:
parent(X,Y), parent(Y,Z) -> grandparent(X,Z)    (Rule1)
child(Y,X) -> parent(X,Y)                       (Rule2)

We select the first rule. We see that the head is grandparent(X,Z) and the body is parent(X,Y), parent(Y,Z).
-> goal = Atom(Var('A'), Relation('grandparent'), Var('B'))
-> premises = [Atom(Var('A'), Relation('parent'), Var('C')),
               Atom(Var('C'), Relation('parent'), Var('B'))]
-> rule = ExecutableRule('Rule1', goal, premises)

Now, we want to generate a proof for grandparent(X,Z) and we want to generate
individuals along the way.
"""


@dataclass(frozen=True)  # Variables are immutable and hashable
class Var:
    """Represents a variable in a rule, e.g., 'X' or 'Y'."""

    name: str

    def __repr__(self):
        return f"{self.name}"


@dataclass(frozen=True)  # Atoms are immutable and hashable
class Atom:
    """
    Represents a triple pattern, e.g. (Var('X'), rdf:type, Class('Person')).
    Can represent a ground atom (if no vars) or a pattern (with vars).
    """

    subject: Term
    predicate: Term
    object: Term

    def is_ground(self) -> bool:
        """
        Checks if the goal pattern is ground (no variables).
        """
        return not (isinstance(self.subject, Var) or isinstance(self.predicate, Var) or isinstance(self.object, Var))

    def substitute(self, substitution: Dict[Var, Term]) -> "Atom":
        """
        Applies a variable substitution to this pattern.
        """
        return Atom(
            subject=substitution.get(self.subject, self.subject) if isinstance(self.subject, Var) else self.subject,
            predicate=substitution.get(self.predicate, self.predicate)
            if isinstance(self.predicate, Var)
            else self.predicate,
            object=substitution.get(self.object, self.object) if isinstance(self.object, Var) else self.object,
        )

    def get_variables(self) -> Set[Var]:
        """Returns all variables in this atom."""
        vars = set()
        for term in [self.subject, self.predicate, self.object]:
            if isinstance(term, Var):
                vars.add(term)
        return vars

    def __repr__(self):
        s = self.subject.name if hasattr(self.subject, "name") else str(self.subject)

        if self.predicate == RDF.type:
            p = "rdf:type"
        else:
            p = self.predicate.name if hasattr(self.predicate, "name") else str(self.predicate)

        o = self.object.name if hasattr(self.object, "name") else str(self.object)
        return f"<{s}, {p}, {o}>"


@dataclass
class ExecutableRule:
    """
    Represents an executable rule derived from an ontology axiom.

    E.g., rdfs:subClassOf(ClassA, ClassB) becomes:
    Conclusion: (Var('X'), rdf:type, ClassB)
    Premises:   [(Var('X'), rdf:type, ClassA)]
    """

    name: str
    conclusion: Union[Atom, None]  # None for dummy rules
    premises: List[Atom]

    def __repr__(self):
        prem_str = ", ".join(map(str, self.premises))
        return f"{prem_str} ⇒ {self.conclusion}  ({self.name})"

    def __hash__(self):
        return hash(self.name)

    def is_recursive(self) -> bool:
        """
        A rule is recursive if any atom in its body (premises)
        shares the same "key" (e.g., predicate or class) as the head.
        """

        head = self.conclusion
        head_pred = head.predicate

        # Get the "type" term (class for rdf:type, or None)
        # This handles cases like A(x) -> B(x)
        head_type_term = None
        if head_pred == RDF.type and isinstance(head.object, (Class, Var)):
            head_type_term = head.object

        for premise in self.premises:
            # 1. Check predicate match
            if premise.predicate == head_pred:
                # 2. If predicate is rdf:type, check class match
                if head_type_term is not None:
                    # e.g., A(X) ... -> A(Y)
                    if premise.object == head_type_term:
                        return True
                # 3. If predicate is not rdf:type, just predicate match is enough
                else:
                    # e.g., P(X,Y) ... -> P(X,Z)
                    return True
        return False


@dataclass
class Constraint:
    """
    Represents a constraint that must not be violated.
    E.g., owl:disjointWith(ClassA, ClassB)
    """

    name: str
    constraint_type: URIRef  # e.g., OWL.disjointWith
    terms: List[Term]  # e.g., [Class('ClassA'), Class('ClassB')]

    def __repr__(self):
        return f"Constraint(name={self.name}, type={self.constraint_type}, terms={self.terms})"


@dataclass(frozen=True)  # Proofs are immutable and hashable
class Proof:
    """
    Represents a proof tree for a single GROUND goal (Atom).

    A proof is either for
        - a base fact
            -> if no rules can be applied to prove a goal
            -> leaf in the proof tree
            -> rule=None
        - a derived fact
            -> if a rule was applied to prove the goal
            -> node in the proof tree
            -> rule=ExecutableRule, sub_proofs=[Proof, ...]
    """

    # The ground atom this proof satisfies.
    goal: Atom

    # The rule (ExecutableRule) whose conclusion (ExecutableRule.conclusion)
    # was unified with the goal (grounded Atom).
    # If 'None', this proof represents a base fact (a leaf in the tree).
    rule: Optional[ExecutableRule] = None

    # The list of proofs for the premises of the rule.
    # Must be empty if rule is None.
    sub_proofs: Tuple["Proof", ...] = field(default_factory=tuple)
    # We use a Tuple instead of List to make Proof hashable

    # recursive_use_counts
    #   -> tracks {rule_name: count} for recursive rules.
    #   -> is an immutable, hashable set
    recursive_use_counts: frozenset[Tuple[str, int]] = field(default_factory=frozenset)
    # field() returns an empty frozenset

    # Maps variables in the rule to their ground terms
    substitutions: Dict[Var, Term] = field(default_factory=dict, hash=False, compare=False)

    # Validity flag (for negative sampling visualization)
    is_valid: bool = field(default=True, compare=False)

    # Flag if this specific node is the corrupted leaf
    is_corrupted_leaf: bool = field(default=False, compare=False)

    def __post_init__(self):
        # A base fact proof cannot have sub-proofs
        if self.rule is None and self.sub_proofs:
            raise ValueError("Base fact proof (rule=None) cannot have sub-proofs.")

        # A derived fact proof must have the same number of sub-proofs as premises
        # in the rule it tries to prove.
        if self.rule is not None and len(self.sub_proofs) != len(self.rule.premises):
            raise ValueError(
                f"Proof for rule '{self.rule.name}' must have "
                f"{len(self.rule.premises)} sub-proofs, but "
                f"{len(self.sub_proofs)} were given."
            )

        # The goal of a proof must be ground.
        if not self.goal.is_ground():
            raise ValueError(f"Proof goal '{self.goal}' must be a ground atom.")

    def is_base_fact(self) -> bool:
        """
        Checks if this proof represents a base fact (leaf).
        """
        return self.rule is None

    def get_base_facts(self) -> Set[Atom]:
        """
        Traverses the proof tree and returns the set of all base facts (leaves) this proof depends on.
        """

        # This is a base fact (a leaf)
        if self.is_base_fact():
            return {self.goal}

        # Derived fact: gather base facts from sub-proofs
        base_facts: Set[Atom] = set()
        for sp in self.sub_proofs:
            base_facts.update(sp.get_base_facts())
        return base_facts

    def get_all_atoms(self) -> Set[Atom]:
        """Returns all atoms (base + inferred) in this proof tree."""
        atoms = {self.goal}
        for sp in self.sub_proofs:
            atoms.update(sp.get_all_atoms())
        return atoms

    def get_depth(self) -> int:
        """Returns the maximum depth of this proof tree."""
        if self.is_base_fact():
            return 0
        return 1 + max((sp.get_depth() for sp in self.sub_proofs), default=0)

    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics about this proof tree."""
        all_atoms = self.get_all_atoms()
        base_facts = self.get_base_facts()

        individuals = set()
        for atom in all_atoms:
            for term in [atom.subject, atom.object]:
                if isinstance(term, Individual):
                    individuals.add(term.name)

        rules_used = set()

        def collect_rules(p: Proof):
            if p.rule:
                rules_used.add(p.rule.name)
            for sp in p.sub_proofs:
                collect_rules(sp)

        collect_rules(self)

        return {
            "total_atoms": len(all_atoms),
            "base_facts": len(base_facts),
            "inferred_facts": len(all_atoms) - len(base_facts),
            "depth": self.get_depth(),
            "individuals": len(individuals),
            "rules_used": sorted(rules_used),
            "max_recursion": max((count for _, count in self.recursive_use_counts), default=0),
        }

    def corrupt_leaf(self, original_atom: Atom, corrupted_atom: Atom) -> "Proof":
        """
        Creates a new Proof tree where the leaf matching original_atom is replaced
        by a corrupted leaf (corrupted_atom), and the path is marked invalid.
        """
        # Base case: this is the leaf we want to corrupt
        if self.is_base_fact():
            if self.goal == original_atom:
                return Proof(
                    goal=corrupted_atom,
                    rule=None,
                    sub_proofs=tuple(),
                    recursive_use_counts=self.recursive_use_counts,
                    substitutions=self.substitutions,
                    is_valid=False,
                    is_corrupted_leaf=True,
                )
            return self

        # Recursive step
        if self.rule:
            new_sub_proofs = []
            changed = False
            for sp in self.sub_proofs:
                # Recursively try to corrupt the leaf in sub-proofs
                new_sp = sp.corrupt_leaf(original_atom, corrupted_atom)
                new_sub_proofs.append(new_sp)

                # If the sub-proof changed (i.e., it contained the target leaf),
                # then this node also changes
                if new_sp is not sp:
                    changed = True

            if changed:
                return Proof(
                    goal=self.goal,  # Goal remains same (but is now invalidly derived)
                    rule=self.rule,
                    sub_proofs=tuple(new_sub_proofs),
                    recursive_use_counts=self.recursive_use_counts,
                    substitutions=self.substitutions,
                    is_valid=False,  # Invalid because a child is invalid
                )

        # If not found or not changed, return self
        return self

    def save_visualization(
        self,
        filepath: str,
        format: str = "pdf",
        title: Optional[str] = None,
        root_label: Optional[str] = None,
    ) -> None:
        """
        Save proof tree visualization to file.

        Args:
            filepath: Output file path (without extension)
            format: Output format ("pdf", "png", "svg")
            title: Optional title for the graph
            root_label: Optional custom label for the root node
        """

        dot = self._create_graphviz(root_label=root_label)

        if title:
            dot.attr(label=title, labelloc="t", fontsize="16", fontname="FiraCode-Bold")

        try:
            dot.render(filepath, format=format, cleanup=True)
            # print(f"✓ Saved proof visualization to: {filepath}.{format}")
            # Save .dot file for verification
            # dot.save(filepath + ".dot")
        except Exception as e:
            print(f"✗ Failed to render graph: {e}")
            # Save .dot file as fallback
            dot.save(filepath + ".dot")
            print(f"  Saved .dot file to: {filepath}.dot")

    def _create_graphviz(self, root_label: Optional[str] = None) -> Any:
        """
        Create a Graphviz graph object for this proof tree.

        Returns:
            graphviz.Digraph: The graph object
        """
        import graphviz

        dot = graphviz.Digraph(comment="Proof Tree")

        # Layout settings
        dot.attr(rankdir="BT")  # Bottom to top (premises support conclusions)
        dot.attr(splines="ortho")
        dot.attr(nodesep="0.6", ranksep="0.8")
        dot.attr("node", shape="plain", fontname="FiraCode")

        # Track node IDs
        node_counter = [0]  # Use list for mutable counter in closure
        node_ids: Dict[Proof, str] = {}

        def add_proof_node(proof: Proof, parent_id: Optional[str] = None) -> str:
            """Recursively add proof nodes to graph."""
            # Reuse node if already created (DAG structure)
            if proof in node_ids:
                return node_ids[proof]

            # Create unique node ID
            node_id = f"node_{node_counter[0]}"
            node_counter[0] += 1
            node_ids[proof] = node_id

            # Determine node styling
            if proof.is_corrupted_leaf:
                header_color = "#FFEBEE"  # Light red
                border_color = "#C62828"  # Dark red
                type_label = "CORRUPTED FACT"
            elif not proof.is_valid:
                # Custom label for root node if provided
                if proof == self and root_label == "DERIVED NEGATIVE FACT":
                    header_color = "#FFEBEE"  # Light red
                    border_color = "#C62828"  # Dark red
                    type_label = root_label
                elif proof == self and root_label:
                    header_color = "#FFEBEE"  # Light red
                    border_color = "#EF9A9A"  # Lighter red
                    type_label = root_label
                elif proof.is_base_fact():
                    header_color = "#FFEBEE"  # Light red
                    border_color = "#EF9A9A"  # Lighter red
                    type_label = "BASE FACT"
                else:
                    header_color = "#FFEBEE"  # Light red
                    border_color = "#EF9A9A"  # Lighter red
                    type_label = f"Rule: {proof.rule.name} (INVALID)"
            elif proof.is_base_fact():
                header_color = "#E8F5E9"  # Light green
                border_color = "#2E7D32"  # Dark green
                type_label = "BASE FACT"
            else:
                header_color = "#E3F2FD"  # Light blue
                border_color = "#1565C0"  # Dark blue
                # Format rule as: premise1, premise2 -> conclusion
                if proof.rule:
                    premises_str = ", ".join([str(p) for p in proof.rule.premises])
                    conclusion_str = str(proof.rule.conclusion)
                    type_label = f"{premises_str} ⇒ {conclusion_str}"
                    # Escape HTML characters for Graphviz label
                    type_label = type_label.replace("<", "&lt;").replace(">", "&gt;")
                else:
                    type_label = "Derived Fact"

            # Format goal atom
            goal_html = self._format_atom_html(proof.goal)

            # Add "NOT" prefix for negative facts
            if proof.is_corrupted_leaf:
                goal_html = f"<B>NOT</B> {goal_html}"
            elif proof == self and root_label == "DERIVED NEGATIVE FACT":
                goal_html = f"<B>NOT</B> {goal_html}"

            if not proof.is_valid and not proof.is_corrupted_leaf:
                # If this is a derived negative fact (propagated), don't cross out
                if proof == self and root_label == "DERIVED NEGATIVE FACT":
                    pass  # Keep goal_html as is
                # If this is the root and we have a custom label, don't append [INVALID]
                elif proof == self and root_label:
                    goal_html = f"<S>{goal_html}</S>"
                else:
                    goal_html = f"<S>{goal_html}</S> <B><FONT COLOR='#C62828'>[INVALID]</FONT></B>"

            # Build HTML label
            label = (
                f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" COLOR="{border_color}">'
                f'<TR><TD BGCOLOR="{border_color}"><FONT COLOR="white"><B>{type_label}</B></FONT></TD></TR>'
                f'<TR><TD BGCOLOR="{header_color}">{goal_html}</TD></TR>'
            )

            # Add substitutions section
            if proof.substitutions:
                label += '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="#555555"><I>Substitutions:</I><BR/>'
                sub_rows = [f"{k.name} &rarr; <B>{self._format_term(v)}</B>" for k, v in proof.substitutions.items()]

                # Split into columns if many substitutions
                if len(sub_rows) > 3:
                    mid = len(sub_rows) // 2 + 1
                    col1 = "<BR/>".join(sub_rows[:mid])
                    col2 = "<BR/>".join(sub_rows[mid:])
                    label += f'<TABLE BORDER="0" CELLSPACING="5"><TR><TD>{col1}</TD><TD>{col2}</TD></TR></TABLE>'
                else:
                    label += "<BR/>".join(sub_rows)
                label += "</FONT></TD></TR>"

            # Add recursion info
            if proof.recursive_use_counts:
                rec_info = ", ".join([f"{name}:{count}" for name, count in proof.recursive_use_counts])
                label += f'<TR><TD BGCOLOR="#FFF3E0"><FONT POINT-SIZE="8">Recursion: {rec_info}</FONT></TD></TR>'

            label += "</TABLE>>"

            # Add node to graph
            dot.node(node_id, label=label)

            # Recursively add sub-proofs
            if proof.rule:
                for i, (premise_pattern, sub_proof) in enumerate(zip(proof.rule.premises, proof.sub_proofs)):
                    sub_id = add_proof_node(sub_proof, node_id)

                    # Edge label
                    edge_label = f"premise {i + 1}:\n{premise_pattern}"
                    edge_label = edge_label.replace("<", "&lt;").replace(">", "&gt;")

                    # Edge from premise to conclusion (BT layout)
                    dot.edge(
                        sub_id,
                        node_id,
                        xlabel=edge_label,
                        fontsize="9",
                        fontcolor="#666666",
                        style="dashed",
                    )

            return node_id

        # Build the graph starting from root
        add_proof_node(self)

        return dot

    def _format_atom_html(self, atom: Atom) -> str:
        """Format an atom with HTML tags for bold Subject/Object."""
        s = self._format_term(atom.subject)
        p = self._format_term(atom.predicate)
        o = self._format_term(atom.object)

        # Beautify RDF Type
        if p == "rdf:type":
            p = '<FONT COLOR="#666666">rdf:type</FONT>'

        return f"<B>{s}</B> {p} <B>{o}</B>"

    def format_tree(self, indent: int = 0) -> str:
        """Format the proof tree as a string."""
        lines = []
        prefix = "  " * indent

        # Format current node
        if self.is_base_fact():
            lines.append(f"{prefix}[BASE] {self.goal}")
        else:
            lines.append(f"{prefix}[RULE: {self.rule.name}] {self.goal}")

        # Format sub-proofs
        for i, sub_proof in enumerate(self.sub_proofs):
            lines.append(sub_proof.format_tree(indent + 1))

        return "\n".join(lines)

    def print(self, indent: int = 0) -> None:
        """Print the proof tree to console."""
        print(self.format_tree(indent))

    def save_text(self, filepath: str) -> None:
        """
        Save proof tree as formatted text file.

        Args:
            filepath: Output file path
        """
        with open(filepath, "w") as f:
            f.write("PROOF TREE\n")
            f.write("=" * 80 + "\n\n")

            # Write statistics
            stats = self.get_statistics()
            f.write("Statistics:\n")
            f.write(f"  Total atoms: {stats['total_atoms']}\n")
            f.write(f"  Base facts: {stats['base_facts']}\n")
            f.write(f"  Inferred facts: {stats['inferred_facts']}\n")
            f.write(f"  Max depth: {stats['depth']}\n")
            f.write(f"  Individuals: {stats['individuals']}\n")
            f.write(f"  Rules used: {', '.join(stats['rules_used'])}\n")
            f.write(f"  Max recursion: {stats['max_recursion']}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            # Write proof tree
            f.write("Proof Tree:\n")
            f.write(self.format_tree())

        print(f"Saved proof tree to: {filepath}")

    def _format_term(self, term: Term) -> str:
        """Helper to format a term."""
        if hasattr(term, "name"):
            return str(term.name)
        if term == RDF.type:
            return "rdf:type"
        return str(term)

    def _format_subs(self) -> str:
        """Helper to format substitutions."""
        items = [f"{v.name}->{self._format_term(t)}" for v, t in self.substitutions.items()]
        return "{" + ", ".join(items) + "}"

    @staticmethod
    def create_base_proof(atom: Atom) -> "Proof":
        """Creates a proof for a base fact (leaf)."""
        if not atom.is_ground():
            raise ValueError("Base fact proof must be for a ground atom.")
        return Proof(goal=atom, rule=None, sub_proofs=tuple(), substitutions={})

    @staticmethod
    def create_derived_proof(
        goal: Atom,
        rule: ExecutableRule,
        sub_proofs: List["Proof"],
        substitutions: Dict[Var, Term],
    ) -> "Proof":
        """Creates a proof for a derived fact (node), tracking substitutions."""
        if not goal.is_ground():
            raise ValueError("Derived proof goal must be ground.")

        # Combine recursive counts from sub-proofs
        new_counts: Dict[str, int] = {}
        for sp in sub_proofs:
            for name, count in sp.recursive_use_counts:
                new_counts[name] = max(new_counts.get(name, 0), count)

        if rule.is_recursive():
            new_counts[rule.name] = new_counts.get(rule.name, 0) + 1

        return Proof(
            goal=goal,
            rule=rule,
            sub_proofs=tuple(sub_proofs),
            recursive_use_counts=frozenset(new_counts.items()),
            substitutions=substitutions.copy(),  # Store the substitutions
        )
