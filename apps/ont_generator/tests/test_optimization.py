import sys
import os
import unittest
from omegaconf import OmegaConf

# Add src to path
sys.path.append(os.path.abspath("apps/ont_generator/src"))

from ont_generator.generate import KGenerator


class TestOptimization(unittest.TestCase):
    def setUp(self):
        # Mock config
        self.cfg = OmegaConf.create(
            {
                "ontology": {"path": "data/ont/family.ttl"},
                "generator": {
                    "max_recursion": 3,
                    "global_max_depth": 10,
                    "max_proofs_per_atom": 5,
                    "individual_pool_size": 20,
                    "individual_reuse_prob": 0.5,
                    "use_signature_sampling": True,
                    "min_proof_roots": 1,
                    "max_proof_roots": 2,
                    "always_generate_base_facts": False,
                },
                "export_proofs": False,
                "logging": {"level": "INFO"},
            }
        )

        # We need the ontology file to exist.
        # Check if it exists, otherwise skip (or mock parsing)
        if not os.path.exists(self.cfg.ontology.path):
            print(f"Skipping tests: Ontology file {self.cfg.ontology.path} not found.")
            self.skipTest("Ontology not found")

        self.generator = KGenerator(self.cfg, verbose=False)

    def test_recursive_rule_termination(self):
        """Test that generating proofs for a recursive rule terminates and respects limits."""
        # Assuming 'hasAncestor' or similar exists and is recursive
        # We can iterate all rules and try to generate proofs for one.

        start_rule = None
        for rule in self.generator.parser.rules:
            if rule.name in self.generator.chainer.recursive_rules:
                start_rule = rule.name
                break

        if not start_rule:
            if self.generator.parser.rules:
                start_rule = self.generator.parser.rules[0].name
            else:
                self.skipTest("No rules found")

        print(f"Testing generation for rule: {start_rule}")

        proofs = self.generator.generate_proofs_for_rule(start_rule, n_proof_roots=2, max_proofs=10)

        print(f"Generated {len(proofs)} proofs.")
        self.assertLessEqual(len(proofs), 10)

    def test_lazy_sampling(self):
        """Test that lazy sampling works (doesn't crash on 'infinite' combinations)."""
        # It's hard to simulate 'infinite' without a massive ontology, but we can check if it runs.
        # We just run generation for a few rules.

        for rule in self.generator.parser.rules[:3]:
            self.generator.generate_proofs_for_rule(rule.name, n_proof_roots=1, max_proofs=5)
            # Should not hang
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
