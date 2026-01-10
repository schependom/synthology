"""
DESCRIPTION

    Recursive Reasoning Network (RRN) implementation
    with batched relation updates, which allows for maximal GPU utilization.

    The original RRN implementation had a huge performance bottleneck due to
    sequentially updating entity embeddings one triple at a time, leading to
    many small GPU operations, which were stalled by CPU-GPU communication overhead.

    This batched implementation overcomes that by grouping triples by their
    relation type and performing updates for all triples of the same relation
    in a single batch operation.

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
from collections import defaultdict

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
        Performs iterative updating using a batched sum-of-updates approach.
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
            updated_embedding_m = self.layers[0](embedding_m, memberships_tensor)

            # ---------------------------------------------------------------------------- #
            #                                RELATION UPDATE                               #
            # ---------------------------------------------------------------------------- #

            # Create a buffer to accumulate all update terms
            #   -> intialized to zeros
            #   -> same shape as embedding_m
            update_accumulator = torch.zeros_like(updated_embedding_m, device=device)

            # ------------------ CREATE BATCHES PER (NEGATED) PREDICATE ------------------ #

            # Group all triples by their update layer.
            # We have three update layer types: (classes, positive relations, negative relations)
            # But classes are already handled above.
            grouped_triples = defaultdict(lambda: {"s_idx": [], "o_idx": []})
            # This is a dict of dicts:
            #   key: layer index
            #   value: dict with    keys:   "s_idx" and "o_idx"
            #                       values: lists of subject and object indices respectively
            # Because of the lambda,
            # keys that do not exist yet will be initialized with empty lists.

            # Iterate over all triples and add them to the group dict
            for triple in triples:
                p_idx = triple.predicate.index

                if triple.positive:
                    # Positive relation update layer
                    layer_idx = p_idx + 1
                else:
                    # Negative relation update layer
                    layer_idx = p_idx + self._relation_count + 1

                # Append subject and object indices to the appropriate lists
                grouped_triples[layer_idx]["s_idx"].append(triple.subject.index)
                grouped_triples[layer_idx]["o_idx"].append(triple.object.index)

            # ---------------------- LOOP OVER (NEGATED) PREDICATES ---------------------- #

            # Iterate over each update layer and perform batched updates.
            #
            #   ->  This means that for every distinct relation type (and its negation),
            #       we gather all embedding deltas and apply them in a single operation.
            #
            #   ->  Denote the batch size for the layer as Br
            #       (number of triples with that relation type).
            #
            #   ->  1 BATCH = 1 PREDICATE TYPE (either positive or negated)
            #
            for layer_idx, data in grouped_triples.items():
                # Lists of subject and object indices for this layer (i.e. relation type)
                s_idx_batch = data["s_idx"]
                o_idx_batch = data["o_idx"]

                # Create long tensors
                s_idx_tensor = torch.tensor(s_idx_batch, dtype=torch.long, device=device).unsqueeze(1)
                o_idx_tensor = torch.tensor(o_idx_batch, dtype=torch.long, device=device).unsqueeze(1)
                # unsqueeze to make it (Br x 1) = ('number of triples' x 1)

                # Get the embeddings for the subjects and objects
                #  that appear in relations of this specific (negated) predicate.
                e_s_batch = updated_embedding_m[s_idx_batch, :]
                e_o_batch = updated_embedding_m[o_idx_batch, :]

                # Compute gated update terms for all triples in this batch
                update_term_s_batch, update_term_o_batch = self.layers[layer_idx](e_s_batch, e_o_batch)

                # Add updates to the accumulator matrix
                # using scatter_add to add each update term to the correct entity index.
                update_accumulator.scatter_add_(0, s_idx_tensor.expand_as(update_term_s_batch), update_term_s_batch)
                update_accumulator.scatter_add_(0, o_idx_tensor.expand_as(update_term_o_batch), update_term_o_batch)

                # scatter_add_:
                #   - dim=0: along the rows (entity dimension)
                #   - index: indices where to add the updates
                #   - src: the update terms to add

                # expand_as:
                #   - because index needs to have the same shape as src,
                #     we expand the (Br x 1) index tensor to (Br x d)

            # Apply the final aggregated update
            #   (this is the replacement of the for-loop in the original RRN)
            # and normalize along the embedding dimension
            new_embedding_m = F.normalize(updated_embedding_m + update_accumulator, p=2, dim=1)

            # Update embedding matrix for next iteration
            embedding_m = new_embedding_m

        # After all iterations, return the final embedding matrix
        return embedding_m
