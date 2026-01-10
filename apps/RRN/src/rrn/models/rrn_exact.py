# Exact RRN implementation with pytorch_lightning
"""
DESCRIPTION

    Recursive Reasoning Network (RRN) implementation
    with sequential relation updates, following Algorithm 1 from the paper exactly.

    This sequential implementation updates entity embeddings one triple at a time,
    which is extremely inefficient but precisely matches the paper's algorithmic description.

AUTHOR

    Vincent Van Schependom

NOTATION

    * matrix multiplication
    \cdot           dot product
    [;]             vector concatenation
    (a x b x c)     tensor of shape (a, b, c)

    d               embedding size
    N               number of RRN iterations
    K               |classes(KB)|, i.e. number of classes
    R               |relations(KB)|, i.e. number of relations
    M               number of individuals in the KB
    Bt              batch size in training (here: number of individuals in the KB, since 1 batch = 1 KB)
    Br              batch size in RRN relation updates (number of triples for a specific predicate, either positive or negative)
"""

# PyTorch
# Typing
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Own data structures
from synthology.data_structures import Class, Relation, Triple

from .rrn_common import ClassUpdate, RelationUpdate


class RRN(nn.Module):
    """
    Recursive Reasoning Network.

    Iteratively updates embeddings by propagating information through
    both class memberships and relational triples.

    This implementation follows Algorithm 1 from the paper exactly,
    with sequential updates where each triple update immediately affects
    the embedding matrix for subsequent updates within the same iteration.

    Args:
        embedding_size:     Dimensionality of entity embeddings     -> d in RRN paper
        iterations:         Number of update iterations             -> N in RRN paper
        classes:            List of Class objects (index, name)     -> K = |classes(KB)|
        relations:          List of Relation objects (index, name)  -> R = |relations(KB)|
    """

    def __init__(
        self,
        embedding_size: int,
        iterations: int,
        classes: List[Class],  # distinct classes in the KB
        relations: List[Relation],  # distinct relations in the KB
    ):
        super(RRN, self).__init__()

        # Set embedding size, nb relations, and iterations
        self._embedding_size = embedding_size  # d
        self._relation_count = len(relations)  # R = |relations(KB)|
        self._iterations = iterations  # N

        # Create update layers
        self.layers = nn.ModuleList()

        # Layer 0: Class update
        self.layers.append(ClassUpdate(embedding_size, len(classes)))

        # Layers [1 -> relation_count]: POSITIVE relation updates
        # We create one RelationUpdate layer per distinct predicate p.
        for _ in relations:
            self.layers.append(RelationUpdate(embedding_size))

        # Layers [relation_count+1 -> relation_count*2]: NEGATED relation updates
        # We create one RelationUpdate layer per distinct negated predicate ~p.
        for _ in relations:
            self.layers.append(RelationUpdate(embedding_size))

    def forward(
        self,
        triples: List[Triple],
        # memberships -> for each individual (first list): -1,0,1 for each class (second list)
        memberships: List[List[int]],
        embedding_m: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs iterative updating following Algorithm 1 from the paper.
        Updates are applied sequentially, one triple at a time.
        """

        # Get current device from the model parameters (works with Lightning)
        device = next(self.parameters()).device

        # M individuals: i_1, i_2, ..., i_M
        num_individuals = len(memberships)

        # --------------------------------- EMBEDDING -------------------------------- #

        # Randomly initialize E = [e_{i_1} , e_{i_2} , . . . , e_{i_M} ]^T if not provided
        if embedding_m is None:
            # Embedding size: d
            d = self._embedding_size

            # Uniform between [-1, 1] with shape (num_individuals, d)
            uniform_embeddings = torch.rand(num_individuals, d, device=device) * 2 - 1

            # Normalize initial embeddings
            embedding_m = F.normalize(
                uniform_embeddings,
                p=2,  # L2 norm
                dim=1,  # Normalize along embedding dimension
            )

        # Convert memberships list-of-lists (i.e. List[List[int]]) to a tensor
        memberships_tensor = torch.as_tensor(memberships, dtype=torch.float32).to(device)

        # Do N iterations of message passing
        for _t in range(self._iterations):
            # embedding_m:
            #   - embedding matrix
            #   - shape: (num_individuals, embedding_size)
            #   - either generated above or from previous iteration t-1

            # ---------------------------------------------------------------------------- #
            #                                 CLASS UPDATE                                 #
            # ---------------------------------------------------------------------------- #

            # Read the embeddings from state (t-1),
            #   pass it to the class update layer,
            #   along with the memberships tensor,
            #   and get the updated embeddings after class-based updates.
            embedding_m = self.layers[0](embedding_m, memberships_tensor)

            # ---------------------------------------------------------------------------- #
            #                                RELATION UPDATE                               #
            # ---------------------------------------------------------------------------- #

            # Clone the embedding matrix to prevent inplace modification errors.
            # This separates the read tensor (embedding_m) from the write tensor (embedding_m_next).
            embedding_m_next = embedding_m.clone()

            # ---------------------- SEQUENTIAL TRIPLE PROCESSING ----------------------- #

            # Iterate over each triple sequentially and update embeddings one at a time.
            # Unlike the batched version, updates are applied immediately to embedding_m_next.
            for triple in triples:
                s_idx = triple.subject.index
                o_idx = triple.object.index
                p_idx = triple.predicate.index

                # Determine which layer to use based on positive/negative relation
                if triple.positive:
                    # Positive relation update layer
                    layer_idx = p_idx + 1
                else:
                    # Negative relation update layer
                    layer_idx = p_idx + self._relation_count + 1

                # Read current embeddings for this triple
                e_s = embedding_m[s_idx, :]
                e_o = embedding_m[o_idx, :]

                # Update both embeddings (already normalized inside the layer)
                new_e_s, new_e_o = self.layers[layer_idx](e_s, e_o)

                # Write updated embeddings to the next state tensor
                # This is an inplace operation on embedding_m_next,
                # which is safe because new_e_s and new_e_o depend on embedding_m,
                # not embedding_m_next.
                embedding_m_next[s_idx, :] = new_e_s
                embedding_m_next[o_idx, :] = new_e_o

            # The state for the next iteration is the matrix with all updates applied
            embedding_m = embedding_m_next

        # After all iterations, return the final embedding matrix
        return embedding_m
