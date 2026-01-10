"""
DESCRIPTION:

    Ontology Parser (OntologyParser)

    Parses an OWL/RDFS ontology file (e.g., .ttl) and translates
    OWL 2 RL axioms into executable rules and constraints for
    the backward chainer.

AUTHOR

    Vincent Van Schependom
"""

from collections import defaultdict
from typing import Dict, List, Set, Union

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

# Custom imports
from synthology.data_structures import (
    Atom,
    Attribute,
    Class,
    Constraint,
    ExecutableRule,
    Relation,
    Var,
)


class OntologyParser:
    """
    Parses an ontology file and extracts schema, rules, and constraints.

    The parser converts OWL 2 RL axioms into a format suitable for
    backward chaining, enabling automatic fact generation from rules.
    """

    def __init__(self, ontology_file: str):
        """
        Initializes the parser, loads the graph, and parses the ontology.

        Args:
            ontology_file (str): Path to the .ttl (Turtle) ontology file.

        Raises:
            Exception: If the ontology file cannot be parsed.
        """
        # Load the RDF graph
        self.graph = Graph()
        try:
            self.graph.parse(ontology_file, format="turtle")
        except Exception as e:
            print(f"Error parsing ontology file: {e}")
            raise

        # Storage for schema elements (populated during discovery phase)
        self.classes: Dict[str, Class] = {}  # name -> Class object
        self.relations: Dict[str, Relation] = {}  # name -> Relation object
        self.attributes: Dict[str, Attribute] = {}  # name -> Attribute object

        # Index counters for unique IDs
        self._class_idx = 0
        self._rel_idx = 0
        self._attr_idx = 0

        # Storage for parsed rules and constraints
        self.rules: List[ExecutableRule] = []  # Inference rules
        self.constraints: List[Constraint] = []  # Integrity constraints
        # Storage for inverse property pairs (for loop detection in chainer)
        self.inverse_properties: Dict[str, Set[str]] = defaultdict(set)

        # Storage for domain and range constraints (for negative sampling)
        self.domains: Dict[str, Set[str]] = defaultdict(set)  # relation_name -> {class_names}
        self.ranges: Dict[str, Set[str]] = defaultdict(set)  # relation_name -> {class_names}

        # Pre-defined variables for rule creation
        # These are reused across rules for consistency
        self.X = Var("X")
        self.Y = Var("Y")
        self.Z = Var("Z")

        # ==================== MAIN PARSING PIPELINE ==================== #
        print("Discovering schema (Classes, Properties)...")
        self._discover_schema()

        print("Parsing ontology axioms into rules and constraints...")
        self._setup_handlers()
        self._parse_rules_and_constraints()

        print("Ontology parsing complete.")

    def _get_clean_name(self, uri: URIRef) -> str:
        """
        Removes the URI prefix to get a clean, human-readable name.

        Examples:
            "http://example.org/family#hasParent" -> "hasParent"
            "http://www.w3.org/2002/07/owl#Thing" -> "Thing"

        Args:
            uri (URIRef): The URI to extract a name from.

        Returns:
            str: The clean name without namespace prefix.
        """
        if isinstance(uri, BNode) or isinstance(uri, Literal) or not isinstance(uri, URIRef):
            return str(uri)

        name_str = str(uri)

        # Try to use rdflib's built-in qname (prefixed name) functionality
        try:
            qname = self.graph.namespace_manager.qname(uri)
            if ":" in qname and not qname.startswith("http"):
                # Extract the local part after the colon
                return qname.split(":", 1)[-1]
        except:
            pass  # Fallback to manual split

        # Manual fallback: split on # or /
        if "#" in name_str:
            return name_str.split("#")[-1]

        return name_str.split("/")[-1]

    # ==================== SCHEMA GET-OR-CREATE METHODS ==================== #

    def _get_class(self, uri: URIRef) -> Class:
        """
        Gets or creates a Class object from a URI.

        Ensures each class is only created once, with consistent indexing.

        Args:
            uri (URIRef): The URI of the class.

        Returns:
            Class: The corresponding Class object.
        """
        name = self._get_clean_name(uri)
        if name not in self.classes:
            self.classes[name] = Class(index=self._class_idx, name=name)
            self._class_idx += 1
        return self.classes[name]

    def _get_relation(self, uri: URIRef) -> Relation:
        """
        Gets or creates a Relation (ObjectProperty) object from a URI.

        Relations connect individuals to other individuals.

        Args:
            uri (URIRef): The URI of the relation.

        Returns:
            Relation: The corresponding Relation object.
        """
        name = self._get_clean_name(uri)
        if name not in self.relations:
            self.relations[name] = Relation(index=self._rel_idx, name=name)
            self._rel_idx += 1
        return self.relations[name]

    def _get_attribute(self, uri: URIRef) -> Attribute:
        """
        Gets or creates an Attribute (DatatypeProperty) object from a URI.

        Attributes connect individuals to literal values (strings, numbers, etc.).

        Args:
            uri (URIRef): The URI of the attribute.

        Returns:
            Attribute: The corresponding Attribute object.
        """
        name = self._get_clean_name(uri)
        if name not in self.attributes:
            self.attributes[name] = Attribute(index=self._attr_idx, name=name)
            self._attr_idx += 1
        return self.attributes[name]

    def _get_term(self, uri: URIRef) -> Union[Relation, Attribute]:
        """
        Gets or creates a property (Relation or Attribute).

        Checks the graph for the property's type. Defaults to Relation if unknown.

        Args:
            uri (URIRef): The URI of the property.

        Returns:
            Union[Relation, Attribute]: The corresponding property object.
        """
        name = self._get_clean_name(uri)

        # Check if already created
        if name in self.relations:
            return self.relations[name]
        if name in self.attributes:
            return self.attributes[name]

        # Check type in the graph
        if (uri, RDF.type, OWL.DatatypeProperty) in self.graph:
            return self._get_attribute(uri)

        # Default to Relation (ObjectProperty)
        return self._get_relation(uri)

    def _discover_schema(self) -> None:
        """
        Pre-populates the schema dicts by finding explicit declarations
        of classes and properties in the graph.

        This phase discovers all schema elements before parsing rules,
        ensuring consistent object references across the parsing process.
        """
        # Find all Classes (OWL and RDFS)
        for s in self.graph.subjects(predicate=RDF.type, object=OWL.Class):
            if isinstance(s, URIRef):
                self._get_class(s)
        for s in self.graph.subjects(predicate=RDF.type, object=RDFS.Class):
            if isinstance(s, URIRef):
                self._get_class(s)

        # Find all Object Properties (Relations)
        for s in self.graph.subjects(predicate=RDF.type, object=OWL.ObjectProperty):
            if isinstance(s, URIRef):
                self._get_relation(s)

        # Find all Datatype Properties (Attributes)
        for s in self.graph.subjects(predicate=RDF.type, object=OWL.DatatypeProperty):
            if isinstance(s, URIRef):
                self._get_attribute(s)

    def _parse_rdf_list(self, node: BNode) -> List[URIRef]:
        """
        Helper to parse an RDF list (used for property chains).

        RDF lists are represented as linked structures using rdf:first and rdf:rest.

        Example:
            ( :hasParent :hasParent ) is encoded as:
            _:b1 rdf:first :hasParent ; rdf:rest _:b2 .
            _:b2 rdf:first :hasParent ; rdf:rest rdf:nil .

        Args:
            node (BNode): The head of the RDF list.

        Returns:
            List[URIRef]: The ordered list of URIs.
        """
        chain: List[URIRef] = []
        curr = node

        # Traverse the linked list structure
        while curr and curr != RDF.nil:
            item = self.graph.value(subject=curr, predicate=RDF.first)
            if item and isinstance(item, URIRef):
                chain.append(item)
            curr = self.graph.value(subject=curr, predicate=RDF.rest)

        return chain

    def _setup_handlers(self) -> None:
        """
        Initializes the predicate-to-handler mapping.

        Each OWL/RDFS predicate is mapped to a specific handler method
        that knows how to convert that axiom into rules or constraints.
        """
        self.handlers = {
            # RDFS Rules
            RDFS.subClassOf: self._handle_subClassOf,
            RDFS.subPropertyOf: self._handle_subPropertyOf,
            RDFS.domain: self._handle_domain,
            RDFS.range: self._handle_range,
            # OWL 2 RL Property Rules
            OWL.inverseOf: self._handle_inverseOf,
            OWL.propertyChainAxiom: self._handle_propertyChainAxiom,
            # OWL 2 RL Type-based rules and constraints
            RDF.type: self._handle_rdf_type,
            OWL.disjointWith: self._handle_disjointWith,
        }

    def _parse_rules_and_constraints(self) -> None:
        """
        Main parsing loop.

        Iterates through all triples in the graph and calls the appropriate
        handler based on the predicate. This is where axioms are converted
        into ExecutableRules and Constraints.
        """
        for s, p, o in self.graph:
            # We only handle triples with URI predicates
            if not isinstance(p, URIRef):
                continue

            # Find and call the handler for this predicate
            handler = self.handlers.get(p)
            if handler:
                try:
                    handler(s, p, o)  # type: ignore
                except Exception as e:
                    print(f"Warning: Error handling triple ({s}, {p}, {o}): {e}")
            # Note: We silently skip unhandled predicates (they're common in ontologies)

    # ==================== INDIVIDUAL AXIOM HANDLERS ==================== #

    def _handle_subClassOf(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        """
        Handles rdfs:subClassOf axioms.

        Rule: (C1 rdfs:subClassOf C2) → (?X rdf:type C1) → (?X rdf:type C2)

        Intuition: If X is a member of subclass C1, then X is also a member
                   of superclass C2.

        Args:
            s (URIRef): The subclass.
            p (URIRef): rdfs:subClassOf predicate.
            o (URIRef): The superclass.
        """
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            c1 = self._get_class(s)
            c2 = self._get_class(o)
            rule = ExecutableRule(
                name=f"rdfs_subClassOf_{c1.name}_{c2.name}",
                conclusion=Atom(self.X, RDF.type, c2),
                premises=[Atom(self.X, RDF.type, c1)],
            )
            self.rules.append(rule)

    def _handle_subPropertyOf(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        """
        Handles rdfs:subPropertyOf axioms.

        Rule: (P1 rdfs:subPropertyOf P2) → (?X P1 ?Y) → (?X P2 ?Y)

        Intuition: If P1 is a sub-property of P2, then whenever P1 holds
                   between X and Y, P2 also holds between them.

        Args:
            s (URIRef): The sub-property.
            p (URIRef): rdfs:subPropertyOf predicate.
            o (URIRef): The super-property.
        """
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            p1 = self._get_term(s)
            p2 = self._get_term(o)
            rule = ExecutableRule(
                name=f"rdfs_subPropertyOf_{p1.name}_{p2.name}",
                conclusion=Atom(self.X, p2, self.Y),
                premises=[Atom(self.X, p1, self.Y)],
            )
            self.rules.append(rule)

    def _handle_domain(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        """
        Handles rdfs:domain axioms.

        Rule: (P rdfs:domain C) → (?X P ?Y) → (?X rdf:type C)

        Intuition: If property P has domain C, then the subject of any
                   P triple must be a member of class C.

        Args:
            s (URIRef): The property.
            p (URIRef): rdfs:domain predicate.
            o (URIRef): The domain class.
        """
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            prop = self._get_term(s)
            cls = self._get_class(o)

            # Store domain constraint for negative sampling
            self.domains[prop.name].add(cls.name)

            rule = ExecutableRule(
                name=f"rdfs_domain_{prop.name}_{cls.name}",
                conclusion=Atom(self.X, RDF.type, cls),
                premises=[Atom(self.X, prop, self.Y)],
            )
            self.rules.append(rule)

    def _handle_range(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        """
        Handles rdfs:range axioms.

        Rule: (P rdfs:range C) → (?X P ?Y) → (?Y rdf:type C)

        Intuition: If property P has range C, then the object of any
                   P triple must be a member of class C.

        NOTE: For DatatypeProperty (Attribute), range typically specifies
              a datatype (e.g., xsd:integer), not a class. We still create
              the rule for consistency, though it may not be used in practice.

        Args:
            s (URIRef): The property.
            p (URIRef): rdfs:range predicate.
            o (URIRef): The range class or datatype.
        """
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            prop = self._get_term(s)

            # Check if range is a datatype (for attributes)
            if (o, RDF.type, RDFS.Datatype) in self.graph or str(o).startswith(str(XSD)):
                # This is a datatype range - we don't create a rule for this
                # as it's a type constraint on literal values, not class membership
                return

            cls = self._get_class(o)

            # Store range constraint for negative sampling
            self.ranges[prop.name].add(cls.name)

            rule = ExecutableRule(
                name=f"rdfs_range_{prop.name}_{cls.name}",
                conclusion=Atom(self.Y, RDF.type, cls),
                premises=[Atom(self.X, prop, self.Y)],
            )
            self.rules.append(rule)

    def _handle_inverseOf(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        """
        Handles owl:inverseOf axioms.

        Rules:
        1. (P1 owl:inverseOf P2) → (?X P1 ?Y) → (?Y P2 ?X)
        2. (P1 owl:inverseOf P2) → (?Y P2 ?X) → (?X P1 ?Y)

        Intuition: If P1 and P2 are inverses, then P1(X,Y) implies P2(Y,X)
                   and vice versa. Example: hasParent and hasChild.

        Args:
            s (URIRef): First property.
            p (URIRef): owl:inverseOf predicate.
            o (URIRef): Second property (inverse of first).
        """
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            p1 = self._get_relation(s)
            p2 = self._get_relation(o)

            # Store inverse relationship
            self.inverse_properties[p1.name].add(p2.name)
            self.inverse_properties[p2.name].add(p1.name)

            # Forward direction: P1 → P2
            rule1 = ExecutableRule(
                name=f"owl_inverseOf_{p1.name}_{p2.name}",
                conclusion=Atom(self.Y, p2, self.X),
                premises=[Atom(self.X, p1, self.Y)],
            )

            # Backward direction: P2 → P1
            rule2 = ExecutableRule(
                name=f"owl_inverseOf_{p2.name}_{p1.name}",
                conclusion=Atom(self.X, p1, self.Y),
                premises=[Atom(self.Y, p2, self.X)],
            )

            self.rules.append(rule1)
            self.rules.append(rule2)

    def _handle_propertyChainAxiom(self, s: URIRef, p: URIRef, o: BNode) -> None:
        """
        Handles owl:propertyChainAxiom axioms.

        Rule for chain of length 2:
        (P owl:propertyChainAxiom (P1 P2)) → (?X P1 ?Y) ∧ (?Y P2 ?Z) → (?X P ?Z)

        Intuition: If P is composed of P1 followed by P2, then whenever
                   P1(X,Y) and P2(Y,Z) both hold, P(X,Z) also holds.
                   Example: grandparent = parent ∘ parent

        Args:
            s (URIRef): The resulting property.
            p (URIRef): owl:propertyChainAxiom predicate.
            o (BNode): The RDF list of properties in the chain.
        """
        if isinstance(s, URIRef) and isinstance(o, BNode):
            p_chain = self._get_relation(s)
            chain_list = self._parse_rdf_list(o)

            if len(chain_list) == 2:
                # Standard case: chain of 2 properties
                p1 = self._get_relation(chain_list[0])
                p2 = self._get_relation(chain_list[1])
                rule = ExecutableRule(
                    name=f"owl_chain_{p1.name}_{p2.name}_{p_chain.name}",
                    conclusion=Atom(self.X, p_chain, self.Z),
                    premises=[Atom(self.X, p1, self.Y), Atom(self.Y, p2, self.Z)],
                )
                self.rules.append(rule)

            elif len(chain_list) == 1:
                # Degenerate case: chain of 1 property (equivalent to subPropertyOf)
                p1 = self._get_relation(chain_list[0])
                rule = ExecutableRule(
                    name=f"rdfs_subPropertyOf_{p1.name}_{p_chain.name}",
                    conclusion=Atom(self.X, p_chain, self.Y),
                    premises=[Atom(self.X, p1, self.Y)],
                )
                self.rules.append(rule)

            # Note: Chains of length > 2 would require dynamic rule generation
            # and are not currently supported

    def _handle_rdf_type(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        """
        Handles rdf:type axioms that define special property types.

        Supported types:
        - owl:SymmetricProperty:    P(X,Y) → P(Y,X)
        - owl:TransitiveProperty:   P(X,Y) ∧ P(Y,Z) → P(X,Z)
        - owl:ReflexiveProperty:    P(X,X) for all X in domain
        - owl:IrreflexiveProperty:  NOT P(X,X) [constraint]
        - owl:FunctionalProperty:   P(X,Y1) ∧ P(X,Y2) → Y1 = Y2 [constraint]

        Args:
            s (URIRef): The property or class.
            p (URIRef): rdf:type predicate.
            o (URIRef): The type (e.g., owl:SymmetricProperty).
        """
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            return

        # ==================== SYMMETRIC PROPERTY ==================== #
        if o == OWL.SymmetricProperty:
            # Rule: (P rdf:type owl:SymmetricProperty) → (?X P ?Y) → (?Y P ?X)
            # Intuition: If P is symmetric, then P(X,Y) implies P(Y,X).
            #            Example: "sibling" is symmetric.
            prop = self._get_relation(s)
            rule = ExecutableRule(
                name=f"owl_symmetric_{prop.name}",
                conclusion=Atom(self.Y, prop, self.X),
                premises=[Atom(self.X, prop, self.Y)],
            )
            self.rules.append(rule)

        # ==================== TRANSITIVE PROPERTY ==================== #
        elif o == OWL.TransitiveProperty:
            # Rule: (P rdf:type owl:TransitiveProperty) →
            #       (?X P ?Y) ∧ (?Y P ?Z) → (?X P ?Z)
            # Intuition: If P is transitive, then P(X,Y) and P(Y,Z) imply P(X,Z).
            #            Example: "ancestor" is transitive.
            prop = self._get_relation(s)
            rule = ExecutableRule(
                name=f"owl_transitive_{prop.name}",
                conclusion=Atom(self.X, prop, self.Z),
                premises=[Atom(self.X, prop, self.Y), Atom(self.Y, prop, self.Z)],
            )
            self.rules.append(rule)

        # ==================== REFLEXIVE PROPERTY ==================== #
        elif o == OWL.ReflexiveProperty:
            # Rule: (P rdf:type owl:ReflexiveProperty) → (?X rdf:type Domain(P)) → (?X P ?X)
            # Intuition: If P is reflexive, then every individual in P's domain
            #            is related to itself via P.
            #            Example: "knows" might be reflexive (everyone knows themselves).
            #
            # NOTE: This creates a zero-premise rule that generates reflexive triples
            #       for any individual. In practice, you'd want to limit this to
            #       individuals in the property's domain, but for synthetic generation
            #       we allow the chainer to generate X as needed.
            prop = self._get_relation(s)
            rule = ExecutableRule(
                name=f"owl_reflexive_{prop.name}",
                conclusion=Atom(self.X, prop, self.X),
                premises=[],  # Zero premises - axiom that X relates to itself
            )
            self.rules.append(rule)

        # ==================== IRREFLEXIVE PROPERTY (CONSTRAINT) ==================== #
        elif o == OWL.IrreflexiveProperty:
            # Constraint: (P rdf:type owl:IrreflexiveProperty)
            # This forbids P(X,X) - i.e., no entity can have this property with itself.
            # Example: "parent" is irreflexive (no one is their own parent).
            prop = self._get_term(s)
            constraint = Constraint(
                name=f"owl_irreflexive_{prop.name}",
                constraint_type=OWL.IrreflexiveProperty,
                terms=[prop, self.X],  # Represents that (?X P ?X) is forbidden
            )
            self.constraints.append(constraint)

        # ==================== FUNCTIONAL PROPERTY (CONSTRAINT) ==================== #
        elif o == OWL.FunctionalProperty:
            # Constraint: (P rdf:type owl:FunctionalProperty)
            # This means P can have at most one value for each subject.
            # Example: "birthDate" is functional (everyone has exactly one birth date).
            #
            # In backward chaining, this means if we generate P(X, Y1) and then
            # try to generate P(X, Y2), we must ensure Y1 = Y2.
            prop = self._get_term(s)
            constraint = Constraint(
                name=f"owl_functional_{prop.name}",
                constraint_type=OWL.FunctionalProperty,
                terms=[prop, self.X, self.Y],  # Represents uniqueness constraint
            )
            self.constraints.append(constraint)

    def _handle_disjointWith(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        """
        Handles owl:disjointWith axioms.

        Constraint: (C1 owl:disjointWith C2)
        This forbids any individual from being a member of both C1 and C2.

        Intuition: Disjoint classes have no overlap.
                   Example: "Person" and "Building" are typically disjoint.

        Args:
            s (URIRef): First class.
            p (URIRef): owl:disjointWith predicate.
            o (URIRef): Second class (disjoint with first).
        """
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            c1 = self._get_class(s)
            c2 = self._get_class(o)
            constraint = Constraint(
                name=f"owl_disjoint_{c1.name}_{c2.name}",
                constraint_type=OWL.disjointWith,
                terms=[
                    c1,
                    c2,
                    self.X,
                ],  # Represents that (?X type C1) ∧ (?X type C2) is forbidden
            )
            self.constraints.append(constraint)

    def print_summary(self) -> None:
        """
        Prints a summary of the parsed schema, rules, and constraints.

        Useful for debugging and verifying the parsing results.
        """
        print("\n--- Ontology Parser Summary ---")
        print(f"  Classes:    {len(self.classes)}")
        for name in self.classes:
            print(f"    - {name}")

        print(f"\n  Relations:  {len(self.relations)}")
        for name in self.relations:
            print(f"    - {name}")

        print(f"\n  Attributes: {len(self.attributes)}")
        for name in self.attributes:
            print(f"    - {name}")

        print("\n--- Parsed Rules ---")
        if not self.rules:
            print("  No rules were parsed.")
        for rule in self.rules:
            print(f"  {rule}")

        print("\n--- Parsed Constraints ---")
        if not self.constraints:
            print("  No constraints were parsed.")
        for constraint in self.constraints:
            print(f"  {constraint}")

        print("---------------------------------")
