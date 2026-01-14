import pytest
from unittest.mock import MagicMock
from omegaconf import OmegaConf

from synthology.data_structures import (
    Atom,
    Class,
    Individual,
    KnowledgeGraph,
    Proof,
    Relation,
    Triple,
    Var,
    ExecutableRule,
)
from ont_generator.negative_sampler import NegativeSampler


class TestNegativeSampler:
    @pytest.fixture
    def sampler(self):
        schema_classes = {"Person": Class(0, "Person")}
        schema_relations = {
            "hasParent": Relation(0, "hasParent"),
            "hasGrandparent": Relation(1, "hasGrandparent"),
        }
        cfg = OmegaConf.create({})
        return NegativeSampler(schema_classes, schema_relations, cfg, verbose=True)

    def test_proof_based_corruption_premise_verification(self, sampler):
        """
        Verify that we only generate negatives when the 'other' premises of a rule
        still hold true. This ensures 'hard negatives' are logically consistent
        in their failure.

        Scenario:
        Rule: hasParent(X, Y) ^ hasParent(Y, Z) -> hasGrandparent(X, Z)

        KG:
        - hasParent(me, mom)
        - hasParent(mom, gma)
        - hasGrandparent(me, gma) [Derived]

        Individuals: me, mom, gma, stranger
        """

        # 1. Setup Data Structures
        person_cls = sampler.schema_classes["Person"]
        has_parent = sampler.schema_relations["hasParent"]
        has_grandparent = sampler.schema_relations["hasGrandparent"]

        me = Individual(0, "me")
        mom = Individual(1, "mom")
        gma = Individual(2, "gma")
        stranger = Individual(3, "stranger")

        individuals = [me, mom, gma, stranger]

        # 2. Define Rule
        # hasParent(X, Y) ^ hasParent(Y, Z) -> hasGrandparent(X, Z)
        var_x = Var("X")
        var_y = Var("Y")
        var_z = Var("Z")

        premise1 = Atom(var_x, has_parent, var_y)
        premise2 = Atom(var_y, has_parent, var_z)
        updated_conclusion = Atom(var_x, has_grandparent, var_z)

        rule = ExecutableRule(name="grandparent_rule", conclusion=updated_conclusion, premises=[premise1, premise2])

        # 3. Create Proof for positive fact: hasGrandparent(me, gma)
        # Substitutions: X->me, Y->mom, Z->gma
        substitutions = {var_x: me, var_y: mom, var_z: gma}

        # Proof tree construction
        # Leaf 1: hasParent(me, mom)
        leaf1 = Proof(goal=Atom(me, has_parent, mom), rule=None)
        # Leaf 2: hasParent(mom, gma)
        leaf2 = Proof(goal=Atom(mom, has_parent, gma), rule=None)

        root_proof = Proof(
            goal=Atom(me, has_grandparent, gma), rule=rule, sub_proofs=(leaf1, leaf2), substitutions=substitutions
        )

        # 4. Construct KG
        triples = [
            Triple(me, has_parent, mom, positive=True),
            Triple(mom, has_parent, gma, positive=True),
            Triple(me, has_grandparent, gma, positive=True, proofs=[root_proof]),
        ]

        kg = KnowledgeGraph(
            attributes=[],
            classes=[person_cls],
            relations=[has_parent, has_grandparent],
            individuals=individuals,
            triples=triples,
            memberships=[],
            attribute_triples=[],
        )

        # Index existing facts (critical for premise check)
        sampler._index_existing_facts(kg)

        # ------------------------------------------------------------------ #
        # TEST CASE A: Valid Hard Negative
        # Corrupt premise 2: hasParent(mom, gma) -> hasParent(mom, stranger)
        #
        # Check: Does hasParent(me, mom) still hold? YES.
        # Result: Should accept hasGrandparent(me, stranger) as valid negative.
        # ------------------------------------------------------------------ #

        corrupted_base_atom_A = Atom(mom, has_parent, stranger)

        # Simulate substitution update: Z -> stranger
        subst_A = substitutions.copy()
        subst_A[var_z] = stranger

        # Run check
        is_valid_A = sampler._check_premises_satisfied(rule, subst_A, corrupted_base_atom_A)
        assert is_valid_A is True, "Should be valid because the other premise (me, mom) exists."

        # ------------------------------------------------------------------ #
        # TEST CASE B: Invalid Hard Negative
        # Corrupt premise 1: hasParent(me, mom) -> hasParent(me, stranger)
        #
        # Check: Does hasParent(stranger, gma) exist? NO.
        # Result: Should REJECT hasGrandparent(me, gma) (or whatever follows)
        # because the chain is broken at link 2 as well.
        # ------------------------------------------------------------------ #

        corrupted_base_atom_B = Atom(me, has_parent, stranger)

        # Simulate substitution update: Y -> stranger
        subst_B = substitutions.copy()
        subst_B[var_y] = stranger

        # Run check
        # premise1 becomes hasParent(me, stranger) -> MATCHES CORRUPTION (skipped)
        # premise2 becomes hasParent(stranger, gma) -> DOES NOT EXIST
        is_valid_B = sampler._check_premises_satisfied(rule, subst_B, corrupted_base_atom_B)
        assert is_valid_B is False, "Should be invalid because the second premise (stranger, gma) is missing."
