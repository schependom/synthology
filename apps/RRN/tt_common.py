"""
DESCRIPTION

    Common code for RRN training and testing.

AUTHOR

    Vincent Van Schependom

NOTATION

    *               matrix multiplication
    \cdot           dot product
    [;]             vector concatenation
    (a x b x c)     tensor of shape (a, b, c)

    d               embedding size
    N               number of RRN iterations
    C               |classes(KB)|, i.e. number of classes
    R               |relations(KB)|, i.e. number of relations
    M               number of individuals in the KB
    Br              batch size in RRN relation updates (number of triples for a specific predicate, either positive or negative)
    Bt              batch size in training (here: Bt = M = number of individuals in the KB, since 1 batch = 1 KB)
"""

# PyTorch
import torch
import torch.nn as nn

# Standard libraries
from typing import Tuple

# Own modules
from data_structures import KnowledgeGraph
from rrn_model_batched import RRN, ClassesMLP, RelationMLP


def initialize_model(
    embedding_size: int,
    iterations: int,
    reference_kg: KnowledgeGraph,
    device: torch.device,
) -> Tuple[RRN, nn.ModuleList]:
    #
    # ------------------------------ INITIALIZE RRN ------------------------------ #

    rrn = RRN(
        embedding_size=embedding_size,
        iterations=iterations,
        classes=reference_kg.classes,
        relations=reference_kg.relations,
    ).to(device)

    # ------------------------------ INITIALIZE MLPs ----------------------------- #

    # Initialize MLPs
    mlps = nn.ModuleList()

    # MLP^{C_i} = P(<s, member_of, o> | KB)
    mlps.append(ClassesMLP(embedding_size, len(reference_kg.classes)).to(device))

    # One MLP per relation type (positive or negative predicate)
    for _ in reference_kg.relations:
        # MLP^{R_i} = P(<s, R_i, o> | KB)
        mlps.append(RelationMLP(embedding_size).to(device))

    return rrn, mlps
