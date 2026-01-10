# Common between batched and exact
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
