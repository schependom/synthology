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

    return {
        "base_triples": base_triples,
        "base_memberships": base_memberships,
        "inferred_triples": inferred_triples,
        "inferred_memberships": inferred_memberships,
        "all_triples": all_triples,
        "all_memberships": all_memberships,
    }
