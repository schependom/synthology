from typing import List

from synthology.data_structures import KnowledgeGraph


def preprocess_knowledge_graph(
    kg: KnowledgeGraph,
) -> dict[str, List]:
    """
    Preprocesses a knowledge graph for training

    Args:
        kg:         Knowledge graph to preprocess

    Returns:
        Dictionary containing:
        - List of   message passing     triples                                 (base facts = specified)
        - List of   message passing     membership vectors \in {-1,0,1}^|C|     (base facts = specified)
        - List of   TEST                triples                                 (inferred)
        - List of   TEST                membership vectors                      (inferred)
        - List of   ALL=target          triples                                 (specified AND inferred)
        - List of   ALL=target          membership vectors                      (specified AND inferred)
    """

    # Base fact triples and all triples
    base_triples = [t for t in kg.triples if t.is_base_fact]
    inferred_triples = [t for t in kg.triples if not t.is_base_fact]
    all_triples = kg.triples

    # Base fact memberships and all memberships
    base_memberships = []
    inferred_memberships = []
    all_memberships = []

    # Non-factual memberships are memberships that are not known facts,
    # i.e., they are not explicitly stated in the knowledge graph.

    for individual in kg.individuals:
        # Initialize vectors with zeros
        base_membership_vec = [0] * len(kg.classes)
        inferred_membership_vec = [0] * len(kg.classes)
        all_membership_vec = [0] * len(kg.classes)

        # Populate based on class memberships
        for membership in individual.classes:
            class_idx = membership.cls.index

            # Membership value: 1 if member, -1 if not member, 0 if unknown
            #
            # -> based on indicator function
            #           1_KB : individuals(KB) -> {-1,0,1}^|C|
            #           1_KB(i) = ( 1 if i is member of C
            #                      -1 if i is not member of C
            #                       0 if otherwise )
            #
            # -> see page 7 in the RRN paper
            #
            membership_value = 1 if membership.is_member else -1

            # Set in all membership vector
            all_membership_vec[class_idx] = membership_value

            # Only set in base membership vector
            if membership.is_base_fact:
                base_membership_vec[class_idx] = membership_value
            else:
                inferred_membership_vec[class_idx] = membership_value

        # Note that 1_KB(i) = 0 for all classes C where the membership is unknown
        # (i.e., not explicitly stated in individual.classes)

        base_memberships.append(base_membership_vec)
        inferred_memberships.append(inferred_membership_vec)
        all_memberships.append(all_membership_vec)

    # Group triples by predicate for efficient batching
    base_grouped = group_triples_to_tensors(base_triples, len(kg.relations))
    inferred_grouped = group_triples_to_tensors(inferred_triples, len(kg.relations))
    all_grouped = group_triples_to_tensors(all_triples, len(kg.relations))

    return {
        "base_triples": base_triples,
        "base_grouped": base_grouped,
        "base_memberships": base_memberships,
        "inferred_triples": inferred_triples,
        "inferred_grouped": inferred_grouped,
        "inferred_memberships": inferred_memberships,
        "all_triples": all_triples,
        "all_grouped": all_grouped,
        "all_memberships": all_memberships,
        "individuals": kg.individuals,
    }

def group_triples_to_tensors(triples: List, relation_count: int) -> dict:
    """
    Groups triples by their predicate index and converts to PyTorch tensors for efficient batching.

    CONTEXT:
        In the RRN, relation updates are performed in batches where each batch corresponds to
        a specific relation type r (or its negation \neg r).
        
        Instead of iterating over triples one-by-one, we group all triples <s, r, o>
        that share the same relation r into a single bucket. This allows us to gather
        the embeddings for all s and all o simultaneously and apply the specific
        update function for r in one vectorized operation.

    MAPPING TO LAYERS:
        The RRN model defines a sequence of update layers:
            Layer 0:                Class Update
            Layer 1..R:             Positive Relation Updates (for r_1 ... r_R)
            Layer R+1..2R:          Negative Relation Updates (for \neg r_1 ... \neg r_R)
        
        Therefore, a triple <s, r_k, o> maps to:
            - Layer k+1             if it is a positive fact
            - Layer k+R+1           if it is a negative fact (negated relation)

    OUTPUT STRUCTURE:
        The output is a dictionary mapping the Layer Index to the indices of subjects and objects:
        {
            layer_idx: {
                "s": LongTensor([s_1, s_2, ..., s_{Br}]),   # Indices of subjects
                "o": LongTensor([o_1, o_2, ..., o_{Br}])    # Indices of objects
            },
            ...
        }
        where Br is the batch size (number of triples) for that specific relation.
    """
    import torch
    from collections import defaultdict

    # Grouping structure
    grouped = defaultdict(lambda: {"s_idx": [], "o_idx": []})
    
    for triple in triples:
        p_idx = triple.predicate.index
        
        # Determine layer index (logic from rrn_batched.py)
        if triple.positive:
            # Positive relation update layer
            # Layer index = k + 1 (since layer 0 is class update)
            layer_idx = p_idx + 1
        else:
            # Negative relation update layer
            # Layer index = k + R + 1
            layer_idx = p_idx + relation_count + 1
            
        grouped[layer_idx]["s_idx"].append(triple.subject.index)
        grouped[layer_idx]["o_idx"].append(triple.object.index)

    # Convert to tensors
    tensor_grouped = {}
    for layer_idx, data in grouped.items():
        tensor_grouped[layer_idx] = {
            "s": torch.tensor(data["s_idx"], dtype=torch.long),
            "o": torch.tensor(data["o_idx"], dtype=torch.long)
        }
    
    return tensor_grouped
