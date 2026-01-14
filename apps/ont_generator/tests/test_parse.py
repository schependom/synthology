import pytest
from unittest.mock import MagicMock, patch
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from ont_generator.parse import OntologyParser


class TestOntologyParser:
    @pytest.fixture
    def mock_graph(self):
        return Graph()

    @pytest.fixture
    def parser(self, mock_graph):
        with patch("ont_generator.parse.Graph", return_value=mock_graph):
            # We mock Graph.parse to avoid needing a real file
            with patch.object(mock_graph, "parse"):
                parser = OntologyParser("dummy.ttl")
        return parser

    def test_discover_schema(self, parser):
        """Test that schema elements are correctly discovered from the graph."""
        # Add some schema definitions to the graph manually
        Person = URIRef("http://example.org/Person")
        hasParent = URIRef("http://example.org/hasParent")
        age = URIRef("http://example.org/age")

        parser.graph.add((Person, RDF.type, OWL.Class))
        parser.graph.add((hasParent, RDF.type, OWL.ObjectProperty))
        parser.graph.add((age, RDF.type, OWL.DatatypeProperty))

        # Run discovery
        parser._discover_schema()

        assert "Person" in parser.classes
        assert "hasParent" in parser.relations
        assert "age" in parser.attributes

    def test_handle_subclass_of(self, parser):
        """Test conversion of rdfs:subClassOf to a rule."""
        Student = URIRef("http://example.org/Student")
        Person = URIRef("http://example.org/Person")

        # Manually trigger handler
        parser._handle_subClassOf(Student, RDFS.subClassOf, Person)

        assert len(parser.rules) == 1
        rule = parser.rules[0]

        # Rule: Student(X) -> Person(X)
        assert rule.name == "rdfs_Student_subClassOf_Person"
        assert len(rule.premises) == 1
        assert rule.premises[0].predicate == RDF.type
        assert rule.premises[0].object.name == "Student"
        assert rule.conclusion.predicate == RDF.type
        assert rule.conclusion.object.name == "Person"

    def test_handle_inverse_of(self, parser):
        """Test conversion of owl:inverseOf to rules."""
        hasParent = URIRef("http://example.org/hasParent")
        hasChild = URIRef("http://example.org/hasChild")

        parser._handle_inverseOf(hasParent, OWL.inverseOf, hasChild)

        assert len(parser.rules) == 2

        # Rule 1: hasParent(X, Y) -> hasChild(Y, X)
        rule1 = parser.rules[0]
        assert rule1.premises[0].predicate.name == "hasParent"
        assert rule1.conclusion.predicate.name == "hasChild"

        # Rule 2: hasChild(X, Y) -> hasParent(Y, X)
        rule2 = parser.rules[1]
        assert rule2.premises[0].predicate.name == "hasChild"
        assert rule2.conclusion.predicate.name == "hasParent"
