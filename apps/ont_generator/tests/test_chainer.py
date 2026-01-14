import pytest
from unittest.mock import MagicMock
from omegaconf import OmegaConf
from rdflib.namespace import OWL, RDF

from synthology.data_structures import (
    Atom,
    Class,
    Constraint,
    ExecutableRule,
    Individual,
    Relation,
    Var,
)
from ont_generator.chainer import BackwardChainer


class TestBackwardChainer:
    @pytest.fixture
    def chainer(self):
        # minimal config
        cfg = OmegaConf.create(
            {
                "generator": {
                    "max_recursion": 2,
                    "global_max_depth": 5,
                    "max_proofs_per_atom": 1,
                    "use_signature_sampling": False,
                    "individual_pool_size": 10,
                    "individual_reuse_prob": 0.5,
                },
                "export_proofs": False,
            }
        )
        return BackwardChainer(all_rules=[], cfg=cfg)

    def test_unify_success(self, chainer):
        """Test successful unification of a ground goal with a pattern."""
        # Goal: hasParent(me, mom)
        # Pattern: hasParent(X, Y)

        rel = Relation(0, "hasParent")
        me = Individual(0, "me")
        mom = Individual(1, "mom")

        var_x = Var("X")
        var_y = Var("Y")

        goal = Atom(me, rel, mom)
        pattern = Atom(var_x, rel, var_y)

        subst = chainer._unify(goal, pattern)

        assert subst is not None
        assert subst[var_x] == me
        assert subst[var_y] == mom

    def test_unify_fail_mismatch(self, chainer):
        """Test unification failure due to mismatched structure/constants."""
        rel1 = Relation(0, "hasParent")
        rel2 = Relation(1, "hasChild")  # Different relation
        me = Individual(0, "me")
        mom = Individual(1, "mom")
        var_x = Var("X")
        var_y = Var("Y")

        # Relation mismatch
        goal = Atom(me, rel1, mom)
        pattern = Atom(var_x, rel2, var_y)
        assert chainer._unify(goal, pattern) is None

        # Object mismatch (if pattern had a constant)
        pattern_const = Atom(var_x, rel1, me)  # expects object 'me'
        assert chainer._unify(goal, pattern_const) is None

    def test_check_constraints_disjoint(self, chainer):
        """Test that disjoint class constraint is enforced."""
        person = Class(0, "Person")
        building = Class(1, "Building")
        ind = Individual(0, "Ind_0")

        # Setup chainer with disjoint classes
        chainer.disjoint_classes = {person: {building}, building: {person}}

        # Atoms implying Ind_0 is both Person and Building
        atoms = {Atom(ind, RDF.type, person), Atom(ind, RDF.type, building)}

        assert chainer._check_constraints(atoms) is False

    def test_check_constraints_irreflexive(self, chainer):
        """Test that irreflexive property constraint is enforced."""
        parent = Relation(0, "hasParent")
        ind = Individual(0, "Ind_0")

        chainer.irreflexive_properties = {parent}

        # Reflexive triple: Ind_0 hasParent Ind_0
        atoms = {Atom(ind, parent, ind)}

        assert chainer._check_constraints(atoms) is False

    def test_check_constraints_functional(self, chainer):
        """Test that functional property constraint is enforced."""
        birth_place = Relation(0, "birthPlace")
        ind = Individual(0, "Ind_0")
        loc1 = Individual(1, "London")
        loc2 = Individual(2, "Paris")

        chainer.functional_properties = {birth_place}

        # Functional violation: Two different birth places
        atoms = {Atom(ind, birth_place, loc1), Atom(ind, birth_place, loc2)}

        assert chainer._check_constraints(atoms) is False

        # Valid case: One birth place
        valid_atoms = {Atom(ind, birth_place, loc1)}
        assert chainer._check_constraints(valid_atoms) is True
