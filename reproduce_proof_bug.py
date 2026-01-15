
import pytest
from synthology.data_structures import Atom, Proof, ExecutableRule, Var, Individual, Relation, Term

def test_propagated_proof_consistency():
    # Setup
    # A(x) -> B(x)
    # B(x) -> C(x)
    # Base: A(0)
    # Derivations: B(0), C(0)
    
    # Vocabulary
    ind0 = Individual(0, "0")
    ind1 = Individual(1, "1")
    rel_A = Relation(0, "A")
    rel_B = Relation(1, "B")
    rel_C = Relation(2, "C")
    
    var_X = Var("X")
    
    # Rule 1: A(X) -> B(X)
    rule1 = ExecutableRule(
        name="Rule1",
        conclusion=Atom(var_X, rel_B, var_X), # B(X, X) - wait let's simpler: B(ind, ind)
        premises=[Atom(var_X, rel_A, var_X)] 
    )
    # Note: Using (S, P, O). Let's say predicates are relations. 
    # A(0,0) -> B(0,0)
    
    # Rule 2: B(X) -> C(X)
    rule2 = ExecutableRule(
        name="Rule2",
        conclusion=Atom(var_X, rel_C, var_X),
        premises=[Atom(var_X, rel_B, var_X)]
    )
    
    # Base Proof: A(0,0)
    atom_A0 = Atom(ind0, rel_A, ind0)
    proof_A0 = Proof.create_base_proof(atom_A0)
    
    # Derived Proof 1: B(0,0)
    atom_B0 = Atom(ind0, rel_B, ind0)
    proof_B0 = Proof.create_derived_proof(
        goal=atom_B0,
        rule=rule1,
        sub_proofs=[proof_A0],
        substitutions={var_X: ind0}
    )
    
    # Derived Proof 2: C(0,0) (ROOT)
    atom_C0 = Atom(ind0, rel_C, ind0)
    proof_C0 = Proof.create_derived_proof(
        goal=atom_C0,
        rule=rule2,
        sub_proofs=[proof_B0],
        substitutions={var_X: ind0}
    )
    
    # Corruption: Change A(0,0) -> A(1,1)
    # We expect propagation: B(0,0) -> B(1,1) and C(0,0) -> C(1,1)
    
    from ont_generator.negative_sampler import NegativeSampler
    # Hacky way to access _create_propagated_proof without full init
    sampler = NegativeSampler({}, {}, None)
    
    corrupted_base = Atom(ind1, rel_A, ind1)
    
    # Term mapping: 0 -> 1
    term_mapping = {ind0: ind1}
    
    # We need to simulate how _create_propagated_proof is called with new_subst
    # But wait, original `_create_propagated_proof` implementation asks for `new_subst` 
    # which is the ROOT substitution.
    # In my new design, I want to pass a mapping.
    # But let's verify the OLD behavior (current bug).
    
    # Current signature:
    # _create_propagated_proof(self, original_proof, original_base_fact, corrupted_base_fact, new_goal, new_subst, is_root)
    
    # New Root Goal: C(1,1)
    atom_C1 = Atom(ind1, rel_C, ind1)
    new_subst_root = {var_X: ind1}
    
    propagated_proof = sampler._create_propagated_proof(
        original_proof=proof_C0,
        original_base_fact=atom_A0,
        corrupted_base_fact=corrupted_base,
        term_mapping=term_mapping
    )
    
    print("\nPropagated Proof Structure:")
    print(f"Root Goal: {propagated_proof.goal}")
    print(f"Sub-Proof 1 Goal: {propagated_proof.sub_proofs[0].goal}")
    print(f"Sub-Sub-Proof 1 Goal: {propagated_proof.sub_proofs[0].sub_proofs[0].goal}")
    
    # Assertions
    # Root should match new goal
    assert propagated_proof.goal == atom_C1
    
    # Leaf should match corrupted base
    leaf = propagated_proof.sub_proofs[0].sub_proofs[0]
    assert leaf.goal == corrupted_base
    assert leaf.is_corrupted_leaf
    
    # MIDDLE NODE (B)
    # This is where the bug is.
    # Expected: B(1,1)
    # Actual (Bug): B(0,0)
    middle_node = propagated_proof.sub_proofs[0]
    atom_B1 = Atom(ind1, rel_B, ind1)
    
    if middle_node.goal == atom_B0:
        print("BUG REPRODUCED: Middle node goal is still B(0,0)")
    elif middle_node.goal == atom_B1:
        print("Logic is correct: Middle node goal is B(1,1)")
    else:
        print(f"Unexpected middle node goal: {middle_node.goal}")
        
    assert middle_node.goal == atom_B1, f"Middle node should be B(1,1) but was {middle_node.goal}"

if __name__ == "__main__":
    try:
        test_propagated_proof_consistency()
        print("Test Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
