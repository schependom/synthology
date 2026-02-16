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
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Own data structures
from synthology.data_structures import Class, Relation, Triple


class RelationUpdateSubject(nn.Module):
    """
    Generates subject embedding update terms based on a batch of <s, p, o> triples.

    One 'batch' corresponds to all triples with the same predicate p, either positive or negated.
    Denote Br as the batch size in RRN relation updates -> Br = number of triples of a (positive/negative) predicate.

    g(s,o)      = sigmoid(V_s * e_s + V_o * e_o)                    -> GATE
    \hat{e}_s^1 = ReLU(W_s * e_s + W_o * e_o + e_s (e_o^T * w))     -> DIRECTION/CANDIDATE
    \hat{e}_s^2 = e_o + \hat{e}_s^1 * g(s,o)                        -> UPDATED EMBEDDING
    e_s         = \hat{e}_s^2 / ||\hat{e}_s^2||_2                   -> NORMALIZATION
    """

    def __init__(self, embedding_size: int):
        super(RelationUpdateSubject, self).__init__()

        # Gate
        self.V_s = nn.Linear(embedding_size, embedding_size, bias=False)
        self.V_o = nn.Linear(embedding_size, embedding_size, bias=False)

        # Direction (candidate)
        self.W_s = nn.Linear(embedding_size, embedding_size, bias=False)
        self.W_o = nn.Linear(embedding_size, embedding_size, bias=False)

        # Interaction
        self.w = nn.Parameter(torch.randn(embedding_size))

    def forward(
        self,
        subject_embeddings: torch.Tensor,  # (Br x d)
        object_embeddings: torch.Tensor,  # (Br x d)
    ) -> torch.Tensor:
        """
        Args:
            subject_embeddings:  Batch of Br subject embedding vectors of size (d)
            object_embeddings:   Batch of Br object embedding vectors of size (d)

        Returns:
            The update term: gate * direction
        """

        # g(s,o) = sigmoid(V_s * e_s + V_o * e_o)
        gate = torch.sigmoid(self.V_s(subject_embeddings) + self.V_o(object_embeddings))

        # Interaction term can be rewritten to be computed more efficiently:
        #
        # Original:
        #       e_s * e_o^T * w,
        # where
        #       e_s \in R^{Br x d},
        #       e_o \in R^{Br x d},
        #       w   \in R^{d}
        #
        # Rewriting the term:
        #       e_s * e_o^T * w  ==  e_s * (e_o^T * w)
        #                        ==  e_s * (e_o \cdot w)
        dot_product_ow = torch.sum(object_embeddings * self.w.unsqueeze(0), dim=1, keepdim=True)
        interaction_term = subject_embeddings * dot_product_ow

        # Update direction
        direction = F.relu(self.W_s(subject_embeddings) + self.W_o(object_embeddings) + interaction_term)

        # Return only the gated update term
        return gate * direction


class RelationUpdateObject(nn.Module):
    """
    Generates object embedding update terms based on a batch of <s, p, o> triples.

    One 'batch' corresponds to all triples with the same predicate p, either positive or negated.
    Denote Br as the batch size in RRN relation updates -> Br = number of triples of a (positive/negative) predicate.

    g(s,o)      = sigmoid(V_s * e_s + V_o * e_o)                    -> GATE
    \hat{e}_o^1 = ReLU(W_s * e_s + W_o * e_o + e_o (e_s^T * w))     -> DIRECTION/CANDIDATE
    \hat{e}_o^2 = e_o + \hat{e}_o^1 * g(s,o)                        -> UPDATED EMBEDDING
    e_o         = \hat{e}_o^2 / ||\hat{e}_o^2||_2                   -> NORMALIZATION
    """

    def __init__(self, embedding_size: int):
        super(RelationUpdateObject, self).__init__()

        # Gate
        self.V_s = nn.Linear(embedding_size, embedding_size, bias=False)
        self.V_o = nn.Linear(embedding_size, embedding_size, bias=False)

        # Direction (candidate)
        self.W_s = nn.Linear(embedding_size, embedding_size, bias=False)
        self.W_o = nn.Linear(embedding_size, embedding_size, bias=False)

        # Interaction
        self.w = nn.Parameter(torch.randn(embedding_size))

    def forward(
        self,
        subject_embeddings: torch.Tensor,  # (Br x d)
        object_embeddings: torch.Tensor,  # (Br x d)
    ) -> torch.Tensor:
        """
        Args:
            subject_embeddings:  Batch of Br subject embedding vectors of size (d)
            object_embeddings:   Batch of Br object embedding vectors of size (d)

        Returns:
            The update term: gate * direction
        """

        # g(s,o) = sigmoid(V_s * e_s + V_o * e_o)
        gate = torch.sigmoid(self.V_s(subject_embeddings) + self.V_o(object_embeddings))

        # Interaction term can be rewritten to be computed more efficiently:
        #
        # Original:
        #       e_s * e_o^T * w,
        # where
        #       e_s \in R^{Br x d},
        #       e_o \in R^{Br x d},
        #       w   \in R^{d}
        #
        # Rewriting the term:
        #       e_s * e_o^T * w  ==  e_s * (e_o^T * w)
        #                        ==  e_s * (e_o \cdot w)
        dot_product_ow = torch.sum(object_embeddings * self.w.unsqueeze(0), dim=1, keepdim=True)
        interaction_term = subject_embeddings * dot_product_ow

        # Update direction
        direction = F.relu(self.W_s(subject_embeddings) + self.W_o(object_embeddings) + interaction_term)

        # Return only the gated update term
        return gate * direction


class RelationUpdate(nn.Module):
    """
    Updates both subject and object gated embedding terms for a batch of <s, p, o> triples.

    A 'batch' corresponds to all triples with the same predicate p,
    either positive or negated.
    """

    def __init__(self, embedding_size: int):
        super(RelationUpdate, self).__init__()
        self.update_subject = RelationUpdateSubject(embedding_size)
        self.update_object = RelationUpdateObject(embedding_size)

    def forward(
        self, subject_embedding: torch.Tensor, object_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            subject_embedding: Embedding vector for the subject entity
            object_embedding: Embedding vector for the object entity

        Returns:
            Tuple of (update_subject, update_object) gated update terms,
            that can be added to the current embeddings to obtain updated embeddings.
        """
        # Update^{subject}
        subject_update_term = self.update_subject(subject_embedding, object_embedding)
        # Update^{object}
        object_update_term = self.update_object(subject_embedding, object_embedding)

        return subject_update_term, object_update_term


class ClassUpdate(nn.Module):
    """
    Updates embeddings of all individuals in the KB based on their class memberships.

    g(i)        = sigmoid(V * [e_i; 1_KB(i)])     -> GATE
    \hat{e}_i^1 = ReLU(W * [e_i; 1_KB(i)])        -> DIRECTION/CANDIDATE
    \hat{e}_i^2 = e_i + \hat{e}_i^1 * g(i)        -> UPDATED EMBEDDING
    e_i         = \hat{e}_i^2 / ||\hat{e}_i^2||_2 -> NORMALIZATION
    """

    def __init__(self, embedding_size: int, classes_count: int):
        super(ClassUpdate, self).__init__()

        # Gate
        input_size = embedding_size + classes_count
        self.V = nn.Linear(input_size, embedding_size, bias=False)

        # Direction (candidate)
        self.W = nn.Linear(input_size, embedding_size, bias=False)

    def forward(
        self,
        individual_embeddings: torch.Tensor,  # e_i \in R^d,    size (N x d)
        individual_memberships: torch.Tensor,  # 1_KB(i),       size (N x K)
    ) -> torch.Tensor:
        """
        Args:
            individual_embeddings:  Current embeddings of the individuals,
                                    size (N x d)
            individual_memberships: Matrix of K class membership values (-1, 0, or 1) for each of the N individuals,
                                    size (N x K)

        Returns:
            Updated and normalized individual embeddings
        """

        # Concatenate full matrices (column-wise) for all individuals
        # (N x (d + K))
        concatenation = torch.cat((individual_embeddings, individual_memberships), dim=1)

        # Gate:
        #   g(i) = sigmoid(V * [e_i; 1_KB(i)])
        gate = torch.sigmoid(self.V(concatenation))  # Output shape (N x d)

        # Update direction (candidate):
        #   \hat{e}_i^1 = ReLU(W * [e_i; 1_KB(i)])
        direction = F.relu(self.W(concatenation))  # Output shape (N x d)

        # Apply gated update and normalize
        #   \hat{e}_i^2 = e_i + \hat{e}_i^1 * g(i)
        #   e_i = \hat{e}_i^2 / ||\hat{e}_i^2||_2
        updated_individuals = F.normalize(
            individual_embeddings + gate * direction,  # element-wise multiplication *
            p=2,  # L2 norm
            dim=1,  # Normalize along embedding dimension
        )
        # PS: dim=0 is the batch (1 KB) dimension

        return updated_individuals


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
        grouped_triples: Dict[int, Dict[str, torch.Tensor]],
        # memberships -> for each individual (first list): -1,0,1 for each class (second list)
        memberships: List[List[int]],
        embedding_m: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs iterative updating using a batched sum-of-updates approach.
        
        Args:
            grouped_triples: Dictionary of pre-grouped triples indices.
                             Key: layer_idx (relation index)
                             Value: {"s": LongTensor, "o": LongTensor}
            memberships: List of class memberships for each individual.
            embedding_m: Optional initial embeddings.
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

            # ---------------------- LOOP OVER (NEGATED) PREDICATES ---------------------- #

            # Iterate over each update layer and perform batched updates.
            # Using pre-grouped indices.
            #
            #   ->  1 BATCH = 1 PREDICATE TYPE (either positive or negated)
            #
            #   Let 'layer_idx' correspond to a specific relation update module UR_k
            #   Let 'data' contain the subject indices S and object indices O for all triples
            #   mapped to this relation type.
            #
            #   S = [s_1, s_2, ..., s_Br]^T
            #   O = [o_1, o_2, ..., o_Br]^T
            #   where Br is the number of triples for this relation.
            #
            for layer_idx, data in grouped_triples.items():
                
                # Get indices tensors and move to device
                # s_idx_tensor and o_idx_tensor are used for scattering updates back to the accumulator.
                # Shape: (Br x 1)
                s_idx_tensor = data["s"].to(device).unsqueeze(1) 
                o_idx_tensor = data["o"].to(device).unsqueeze(1) 

                # Get the indices for gathering embeddings.
                # data["s"] is a 1D LongTensor of size (Br)
                s_idx = data["s"].to(device)
                o_idx = data["o"].to(device)
                
                # GATHER EMBEDDINGS
                #
                #   E_S = E[S]  = [e_{s_1}, ..., e_{s_Br}]^T    (Size: Br x d)
                #   E_O = E[O]  = [e_{o_1}, ..., e_{o_Br}]^T    (Size: Br x d)
                #
                e_s_batch = updated_embedding_m[s_idx, :]
                e_o_batch = updated_embedding_m[o_idx, :]

                # COMPUTE UPDATES
                #
                #   Compute update terms for all triples in this batch simultaneously.
                #
                #   U_S, U_O = UR_k(E_S, E_O)
                #
                #   U_S = [u_{s_1}, ..., u_{s_Br}]^T
                #   U_O = [u_{o_1}, ..., u_{o_Br}]^T
                #
                update_term_s_batch, update_term_o_batch = self.layers[layer_idx](e_s_batch, e_o_batch)

                # ACCUMULATE UPDATES
                #
                #   Add the computed updates to the accumulator matrix.
                #   Since an entity can appear multiple times in S or O (involved in multiple triples),
                #   we use scatter_add_ to sum up all contributions.
                #
                #   Accumulator[s_i] += u_{s_i}
                #   Accumulator[o_i] += u_{o_i}
                #
                update_accumulator.scatter_add_(0, s_idx_tensor.expand_as(update_term_s_batch), update_term_s_batch)
                update_accumulator.scatter_add_(0, o_idx_tensor.expand_as(update_term_o_batch), update_term_o_batch)

            # Apply the final aggregated update
            #   (this is the replacement of the for-loop in the original RRN)
            # and normalize along the embedding dimension
            new_embedding_m = F.normalize(updated_embedding_m + update_accumulator, p=2, dim=1)

            # Update embedding matrix for next iteration
            embedding_m = new_embedding_m

        # After all iterations, return the final embedding matrix
        return embedding_m


class ClassesMLP(nn.Module):
    """
    MLP for predicting class memberships based on individual embeddings.

    Args:
        embedding_size:     Dimensionality of input embeddings
        classes_count:      Number of classes to predict
        num_hidden_layers:  Number of hidden layers

    Paper:
        MLP^{C_i} = P(<s, member_of, C_i> | KB)
    """

    def __init__(
        self,
        embedding_size: int,
        classes_count: int,
        hidden_size: Optional[int] = None,
        num_hidden_layers: int = 1,
    ):
        super(ClassesMLP, self).__init__()

        self.num_hidden_layers = num_hidden_layers

        # TODO tune hidden layer size
        hidden_size = (embedding_size + classes_count) // 2 if hidden_size is None else hidden_size

        # --------------------------- CREATE HIDDEN LAYERS --------------------------- #

        # TODO tune number of hidden layers
        self.hidden_layers = nn.ModuleList()
        current_size = embedding_size
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(current_size, hidden_size))
            current_size = hidden_size

        # ----------------------------- OUTPUT LAYER ----------------------------- #

        # Go from last current_size to classes_count
        self.output_layer = nn.Linear(current_size, classes_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  Entity embedding vectors matrix (in batch)
                Shape: (Br x d)

        Returns:
            Class membership probabilities (sigmoid activated)
            Shape: (Br x K)
        """

        # Pass through hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Final output layer with sigmoid activation for probabilities
        x = self.output_layer(x)

        # Return logits (no sigmoid; use BCEWithLogitsLoss)
        return x


class RelationMLP(nn.Module):
    """
    MLP for predicting relation existence between entity pairs.

    Args:
        embedding_size:     Dimensionality of input embeddings
        hidden_size:        Size of hidden layers
        num_hidden_layers:  Number of hidden layers

    Paper:
        MLP^{R_i} = P(<s, R_i, o> | KB)
    """

    def __init__(
        self,
        embedding_size: int,
        hidden_size: Optional[int] = None,
        num_hidden_layers: int = 1,
    ):
        super(RelationMLP, self).__init__()

        # TODO tune hidden layer size
        hidden_size = (2 * embedding_size + 1) // 2 if hidden_size is None else hidden_size

        # --------------------------- CREATE HIDDEN LAYERS --------------------------- #

        # TODO tune number of hidden layers
        self.hidden_layers = nn.ModuleList()
        current_size = embedding_size * 2
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(current_size, hidden_size))
            current_size = hidden_size  # for next layer

        # ----------------------------- OUTPUT LAYER ----------------------------- #

        self.output_layer = nn.Linear(current_size, 1)

    def forward(self, s_embedding: torch.Tensor, o_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_embedding: Subject entity embedding(s) (in batch)
            o_embedding: Object entity embedding(s) (in batch)

        Returns:
            Probability that the relation holds (sigmoid activated)
        """

        # Concatenate subject and object embeddings along last dim
        #   -> s_embedding: (Br x d)
        #   -> o_embedding: (Br x d)
        #   -> x: (Br x (d+d)) = (Br x 2d)
        x = torch.cat((s_embedding, o_embedding), dim=-1)

        # Pass through hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Final output layer with sigmoid activation for probability
        x = self.output_layer(x)

        # Return logits (no sigmoid; use BCEWithLogitsLoss)
        # for calculating P(<s, R_i, o> | KB) with sigmoid
        return x
