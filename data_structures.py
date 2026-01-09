"""
DESCRIPTION:

    Data structures for representing knowledge graphs.
    This replaces the proprietary reldata format with standard Python classes.

AUTHOR:

    Vincent Van Schependom
"""

from dataclasses import dataclass
from typing import List
from enum import Enum


@dataclass
class Class:
    """Represents a class in the knowledge graph."""

    index: int
    name: str


@dataclass
class Relation:
    """Represents a relation type in the knowledge graph."""

    index: int
    name: str


@dataclass
class Individual:
    """Represents an individual entity in the knowledge graph."""

    index: int
    name: str
    classes: List["Membership"]


@dataclass
class Membership:
    """Represents class membership of an individual."""

    cls: Class
    is_member: bool  # True if member, False if explicitly not a member
    is_inferred: bool  # True if this is a known fact (vs. inferred)


@dataclass
class Triple:
    """Represents a relational triple (subject, predicate, object)."""

    subject: Individual
    predicate: Relation
    object: Individual
    positive: bool  # True for positive predicate, False for negated predicate
    is_inferred: bool  # True if this is a known fact (vs. inferred)


@dataclass
class KnowledgeGraph:
    """
    Complete knowledge graph containing
    classes,        (class index, class name)
    relations,      (relation index, relation name)
    individuals,    (individual index, individual name)
    triples.        (subject, predicate, object, positive, is_fact)

    NOTE:
        The `.index` attribute of each Class, Relation, and Individual
        object (e.g., `individuals[i].index`) MUST be equal to its
        index in its respective list (e.g., `i`). This allows for O(1) lookups.
    """

    classes: List[Class]
    relations: List[Relation]
    individuals: List[Individual]
    triples: List[Triple]


@dataclass
class DataType(Enum):
    INF = 1
    SPEC = 2
    ALL = 3
