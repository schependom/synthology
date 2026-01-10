"""
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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Own data structures
from synthology.data_structures import Class, Relation, Triple


class RelationUpdateSubject(nn.Module):
    """
    Generates subject embedding updates based on <s, p, o> triples.

    g(s,o)      = sigmoid(V_s * e_s + V_o * e_o)                    -> GATE
    \hat{e}_s^1 = ReLU(W_s * e_s + W_o * e_o + e_s (e_o^T * w))     -> DIRECTION/CANDIDATE
    \hat{e}_s^2 = e_s + \hat{e}_s^1 * g(s,o)                        -> UPDATED EMBEDDING
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
        subject_embedding: torch.Tensor,  # (d x 1)
        object_embedding: torch.Tensor,  # (d x 1)
    ) -> torch.Tensor:
        """
        Args:
            subject_embedding:  Subject embedding vector of size (d x 1)
            object_embedding:   Object embedding vector of size (d x 1)

        Returns:
            Updated and normalized subject embedding
        """

        # g(s,o) = sigmoid(V_s * e_s + V_o * e_o)
        gate = torch.sigmoid(self.V_s(subject_embedding) + self.V_o(object_embedding))

        # Interaction term can be rewritten to be computed more efficiently:
        #
        # Original:
        #       e_s * e_o^T * w,
        # where
        #       e_s \in R^{d}
        #       e_o \in R^{d}
        #       w   \in R^{d}
        #
        # Rewriting the term:
        #       e_s * e_o^T * w  ==  e_s * (e_o^T * w)
        #                        ==  e_s * (e_o \cdot w)
        if subject_embedding.dim() == 1:
            dot_product_ow = torch.sum(object_embedding * self.w)
            interaction_term = subject_embedding * dot_product_ow
        else:
            dot_product_ow = torch.sum(object_embedding * self.w.unsqueeze(0), dim=1, keepdim=True)
            interaction_term = subject_embedding * dot_product_ow

        # Update direction
        direction = F.relu(self.W_s(subject_embedding) + self.W_o(object_embedding) + interaction_term)

        # Apply gated update and normalize
        updated = F.normalize(
            subject_embedding + gate * direction,
            p=2,  # L2 norm
            dim=-1,  # Normalize along embedding dimension
        )

        return updated


class RelationUpdateObject(nn.Module):
    """
    Generates object embedding updates based on <s, p, o> triples.

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
        subject_embedding: torch.Tensor,  # (d x 1)
        object_embedding: torch.Tensor,  # (d x 1)
    ) -> torch.Tensor:
        """
        Args:
            subject_embedding:  Subject embedding vector of size (d x 1)
            object_embedding:   Object embedding vector of size (d x 1)

        Returns:
            Updated and normalized object embedding
        """

        # g(s,o) = sigmoid(V_s * e_s + V_o * e_o)
        gate = torch.sigmoid(self.V_s(subject_embedding) + self.V_o(object_embedding))

        # Interaction term can be rewritten to be computed more efficiently:
        #
        # Original:
        #       e_o * e_s^T * w,
        # where
        #       e_s \in R^{d}
        #       e_o \in R^{d}
        #       w   \in R^{d}
        #
        # Rewriting the term:
        #       e_o * e_s^T * w  ==  e_o * (e_s^T * w)
        #                        ==  e_o * (e_s \cdot w)
        if object_embedding.dim() == 1:
            dot_product_sw = torch.sum(subject_embedding * self.w)
            interaction_term = object_embedding * dot_product_sw
        else:
            dot_product_sw = torch.sum(subject_embedding * self.w.unsqueeze(0), dim=1, keepdim=True)
            interaction_term = object_embedding * dot_product_sw

        # Update direction
        direction = F.relu(self.W_s(subject_embedding) + self.W_o(object_embedding) + interaction_term)

        # Apply gated update and normalize
        updated = F.normalize(
            object_embedding + gate * direction,
            p=2,  # L2 norm
            dim=-1,  # Normalize along embedding dimension
        )

        return updated


class RelationUpdate(nn.Module):
    """
    Updates both subject and object embeddings for <s, p, o> triples.

    Combines RelationUpdateSubject and RelationUpdateObject to update
    both entities in a relationship simultaneously.
    """

    def __init__(self, embedding_size: int):
        super(RelationUpdate, self).__init__()
        self.update_subject = RelationUpdateSubject(embedding_size)
        self.update_object = RelationUpdateObject(embedding_size)

    def forward(
        self,
        subject_embedding: torch.Tensor,
        object_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            subject_embedding: Embedding vector for the subject entity
            object_embedding: Embedding vector for the object entity

        Returns:
            Tuple of (updated_subject, updated_object) normalized embeddings
        """
        # Update^{subject}
        updated_subject = self.update_subject(subject_embedding, object_embedding)
        # Update^{object}
        updated_object = self.update_object(subject_embedding, object_embedding)

        return updated_subject, updated_object


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
        individual_embeddings: torch.Tensor,  # e_i \in R^d,    size (M x d)
        individual_memberships: torch.Tensor,  # 1_KB(i),       size (M x K)
    ) -> torch.Tensor:
        """
        Args:
            individual_embeddings:  Current embeddings of the individuals,
                                    size (M x d)
            individual_memberships: Matrix of K class membership values (-1, 0, or 1) for each of the M individuals,
                                    size (M x K)

        Returns:
            Updated and normalized individual embeddings
        """

        # Concatenate full matrices (column-wise) for all individuals
        # (M x (d + K))
        concatenation = torch.cat((individual_embeddings, individual_memberships), dim=1)

        # Gate:
        #   g(i) = sigmoid(V * [e_i; 1_KB(i)])
        gate = torch.sigmoid(self.V(concatenation))  # Output shape (M x d)

        # Update direction (candidate):
        #   \hat{e}_i^1 = ReLU(W * [e_i; 1_KB(i)])
        direction = F.relu(self.W(concatenation))  # Output shape (M x d)

        # Apply gated update and normalize
        #   \hat{e}_i^2 = e_i + \hat{e}_i^1 * g(i)
        #   e_i = \hat{e}_i^2 / ||\hat{e}_i^2||_2
        updated_individuals = F.normalize(
            individual_embeddings + gate * direction,  # element-wise multiplication *
            p=2,  # L2 norm
            dim=1,  # Normalize along embedding dimension
        )

        return updated_individuals


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


class ClassesMLP(nn.Module):
    """
    MLP for predicting class memberships based on individual embeddings.

    Args:
        embedding_size:     Dimensionality of input embeddings
        classes_count:      Number of classes to predict
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
                Shape: (B x d)

        Returns:
            Class membership probabilities (sigmoid activated)
            Shape: (B x K)
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
        embedding_size: Dimensionality of input embeddings
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
        #   -> s_embedding: (B x d)
        #   -> o_embedding: (B x d)
        #   -> x: (B x (d+d)) = (B x 2d)
        x = torch.cat((s_embedding, o_embedding), dim=-1)

        # Pass through hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Final output layer with sigmoid activation for probability
        x = self.output_layer(x)

        # Return logits (no sigmoid; use BCEWithLogitsLoss)
        return x
