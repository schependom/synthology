"""
DESCRIPTION

    Train script of the revised Recursive Reasoning Network (RRN) implementation
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

# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #

# Pytorch
import os
import sys
from collections import defaultdict
from typing import List, Tuple

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from device import get_device
from omegaconf import DictConfig

# Global settings
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tt_common import initialize_model

from apps.RRN.data import (
    custom_collate_fn,
    load_knowledge_graphs,
    preprocess_knowledge_graph,
)

# Own modules
from data_structures import Triple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------- #
#                                   FUNCTIONS                                  #
# ---------------------------------------------------------------------------- #


def save_checkpoint(model: nn.Module, filename: str) -> None:
    """
    Saves model state dict to file.

    Args:
        model:      PyTorch model to save
        filename:   Path to save checkpoint
    """
    try:
        torch.save(model.state_dict(), filename)
    except Exception as e:
        print(f"Error saving checkpoint to {filename}: {e}", file=sys.stderr)


def compute_loss(
    mlps: nn.ModuleList,
    embeddings: torch.Tensor,
    triples: List[Triple],
    membership_targets: List[List[int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes loss for both class memberships and relations of a batch.

    Handles unknown class memberships (0) by masking them out of the loss calculation.
    Applies dynamic positive-class weighting to the relation loss to combat data imbalance.

    Args:

        mlps:               List of MLP classifiers
                                - classes
                                - positive predicates
                                - negative predicates

        embeddings:         Entity embeddings from RRN (Bt x d)

        triples:            List of ALL Triples in the batch (both inferred and base facts)

        membership_targets: List of class membership labels for individuals in the batch
                            This is a list of lists, where each inner list corresponds to
                            the class memberships of one individual, based on [1_KB(i)]_l
                            from the RRN paper:
                                -1 :    not a member
                                1:      member
                                0:      otherwise
                            Note that we use ALL memberships (both base facts and inferred)

        device:             Device to run computations on

    Returns

        Tuple of (total_loss, class_loss, relation_loss)

    ! Note:
        Because 1 TRAINING batch = 1 KB, the batch size is equal to the number of individuals in the KB.
            - Don't confuse the training batch with batching in the RRN model itself, which is done per predicate.
            - Bt = M (number of individuals in KB) for training batches.
            - Br = number of triples for a specific predicate (positive or negative) for RRN relation update batches.
    """

    # Create a non-reduced version of the BCEWithLogitsLoss
    #   ->  'Non-reduced' means it returns loss per sample instead of averaging
    #   ->  This allows us to apply masks and weights manually
    non_reduced_criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)

    # ---------------------------------------------------------------------------- #
    #                               CLASS PREDICTIONS                              #
    # ---------------------------------------------------------------------------- #

    # TODO Check if the embeddings are already on the correct device
    # TODO Check if we actually need to mask the unknowns here, because if 1_KB(i) = 0, then the RRN model should predict 0 as well.

    # ---------------------- FORWARD PASS THROUGH CLASS MLP ---------------------- #

    # Batch prediction logits for all individuals
    cls_logits = mlps[0](embeddings)  # (Bt, C)

    # ------------------------------ CLASS TARGETS ------------------------------ #

    # Targets tensor
    targets_tensor_cls = torch.as_tensor(membership_targets, dtype=torch.float32, device=device)

    # ----------------------------------- MASK ----------------------------------- #

    # Mask is
    #   1.0 where label is NOT unknown (±1)
    #   0.0 where label is unknown (0)
    mask = (targets_tensor_cls != 0).float()

    # Convert labels from {-1,0,1} to {0.0, 0.5, 1.0} for BCE
    cls_targets = (targets_tensor_cls + 1.0) / 2.0

    # Compute un-reduced BCEWithLogits loss PER SAMPLE (not averaged)
    #   -> logits are raw outputs
    #   -> they are converted to probabilities inside the BCEWithLogitsLoss
    #   -> cls_targets has values in {0.0, 0.5, 1.0}
    unreduced_class_loss = non_reduced_criterion(cls_logits, cls_targets)

    # Apply mask to zero out losses for unspecified class memberships in targets
    masked_class_loss = unreduced_class_loss * mask

    # ----------------------------------- LOSS ----------------------------------- #

    # Average loss over known labels only
    num_known_labels = mask.sum().to(device=device)
    if num_known_labels > 0:
        total_class_loss = masked_class_loss.sum() / num_known_labels
    else:
        total_class_loss = torch.zeros((), device=device)

    # ---------------------------------------------------------------------------- #
    #                             RELATION PREDICTIONS                             #
    # ---------------------------------------------------------------------------- #

    # Make a {predicate_index: [triples]} dictionary
    #
    # Note that triples contain both positive and negative samples, so
    # these are grouped together here.
    grouped = defaultdict(list)
    for tr in triples:
        grouped[tr.predicate.index].append(tr)

    # Initialize total relation loss and total relation count
    total_relation_loss = torch.zeros((), device=device)
    total_rel_count = 0

    # For each predicate (negative AND positive), compute loss separately
    for pred_idx, triples_list in grouped.items():
        # Get subject and object indices for this predicate (negative + positive! <> batching in RRN!)
        s_indices = [t.subject.index for t in triples_list]
        o_indices = [t.object.index for t in triples_list]

        # Targets are 1.0 for positive, 0.0 for negative
        #   ->  Because we calculate P(<s, R_i, o> | KB) with a sigmoid, based on the logits.
        #   ->  The chance of a negative triple is P(<s, ~R_i, o> | KB) = 1 - P(<s, R_i, o> | KB)
        rel_targets = [1.0 if t.positive else 0.0 for t in triples_list]

        s_emb = embeddings[s_indices, :]  # [Bt, d]
        o_emb = embeddings[o_indices, :]  # [Bt, d]

        # --------------------- FORWARD PASS THROUGH RELATION MLP -------------------- #

        # Calculate logits that we feed into sigmoid to calculate
        #       P(<s, R_i, o> | KB),
        #       P(<s, ~R_i, o> | KB) = 1 - P(<s, R_i, o> | KB)
        #
        # mlps[0] is for classes, so relation MLPs start from index 1
        rel_logits = mlps[pred_idx + 1](s_emb, o_emb)  # [Bt, 1]

        # ---------------------------------- TARGETS --------------------------------- #

        # Convert to tensor and unsqueeze from [Bt]
        # to [Bt, 1], to match rel_logits shape.
        rel_targets = torch.as_tensor(
            rel_targets,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

        # --------------------------------- WEIGHING --------------------------------- #

        # Because relations are often highly imbalanced (many more negative
        # samples than positive ones), we apply dynamic positive-class weighting
        # based on the ratio of negative to positive samples in this group.

        # Calculate number of positive and negative samples in this group
        num_pos = (rel_targets == 1.0).sum()
        num_neg = (rel_targets == 0.0).sum()

        # Default weights are 1.0 (for both positive and negative predicates)
        all_weights = torch.ones_like(rel_targets, device=device)

        # If we have both positive and negative samples, calculate pos_weight
        if num_pos > 0 and num_neg > 0:
            # The actual weight to apply to positive samples
            ratio = num_neg.float() / num_pos.float()
            pos_weight_val = ratio

            # Apply this weight ONLY to positive (target == 1.0) samples
            all_weights[rel_targets == 1.0] = pos_weight_val

        # -------------------------- ACTUAL LOSS COMPUTATION ------------------------- #

        # Compute un-reduced BCEWithLogits loss
        # per sample (not averaged)
        unreduced_loss = non_reduced_criterion(rel_logits, rel_targets)

        # Apply the weights with element-wise multiplication
        weighted_loss = unreduced_loss * all_weights

        # Compute the weighted losses for this group
        triple_loss = weighted_loss.sum()

        # Get number of triples for this predicate (both positive and negative)
        num_triples = len(triples_list)

        # Update total relation loss and count
        total_relation_loss += triple_loss
        total_rel_count += num_triples

    # After processing all predicates, compute the final average relation loss
    if total_rel_count > 0:
        # Re-average the weighted losses to get the true mean over all relation triples
        total_relation_loss = total_relation_loss / total_rel_count
    else:
        total_relation_loss = torch.zeros((), device=device)

    # ---------------------------------------------------------------------------- #
    #                                  TOTAL LOSS                                  #
    # ---------------------------------------------------------------------------- #

    # Total loss is sum of (average) class and (average) relation losses
    total_loss = total_class_loss + total_relation_loss

    return total_loss, total_class_loss, total_relation_loss


def train_epoch(
    rrn: RRN,
    mlps: nn.ModuleList,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Trains the model for one epoch, so for all batches (all single KB's) in the dataset.

    Args:
        rrn:            Relational Reasoning Network
        mlps:           List of MLP classifiers
        dataloader:     DataLoader for training data
        optimizer:      Optimizer
        device:         Device to run training on
        use_amp:        Whether to use Automatic Mixed Precision (AMP) -> speeds up training on CUDA
        verbose:        Whether to print batch-level progress

    Returns:
        Tuple of (mean_total_loss, mean_class_loss, mean_relation_loss)
    """

    # Set models to training mode
    rrn.train()
    mlps.train()

    scaler = GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    # Enabled defines whether to use scaling (only on CUDA)

    epoch_total_losses = []
    epoch_class_losses = []
    epoch_relation_losses = []

    # Iterate over batches
    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch (always size 1)
        batch_data_dict = batch

        base_triples = batch_data_dict["base_triples"]
        base_memberships = batch_data_dict["base_memberships"]
        # inferred_triples = batch_data_dict["inferred_triples"]
        # inferred_memberships = batch_data_dict["inferred_memberships"]
        all_triples = batch_data_dict["all_triples"]
        all_memberships = batch_data_dict["all_memberships"]

        # Zero gradients
        optimizer.zero_grad()

        if verbose:
            print(
                f"Batch {batch_idx + 1}/{len(dataloader)} of size {len(base_triples)} base fact triples on device {device}"
            )

        # Autocast enables mixed precision, which speeds up training on CUDA devices.
        # Only enabled when Automatic Mixed Precision is requested and device.type == "cuda".
        # If running on a non-CUDA device, it runs in float32.
        with autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            # Forward pass through RRN to get embeddings
            embeddings = rrn(base_triples, base_memberships).to(device)

            # Perform forward pass through MLPs and compute losses
            # ! CRUCIAL: We compute loss on ALL triples and ALL memberships (both base facts and inferred)
            total_loss, class_loss, relation_loss = compute_loss(
                mlps=mlps,
                embeddings=embeddings,
                triples=all_triples,
                membership_targets=all_memberships,
                device=device,
            )

        # Backward pass and optimization step with gradient scaling.
        # Scaler is only enabled on CUDA devices when use_amp is True.
        if scaler.is_enabled():
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Record losses
        epoch_total_losses.append(total_loss.item())
        epoch_class_losses.append(class_loss.item())
        epoch_relation_losses.append(relation_loss.item())

    # Compute mean losses for the epoch
    mean_total_loss = sum(epoch_total_losses) / len(epoch_total_losses)
    mean_class_loss = sum(epoch_class_losses) / len(epoch_class_losses)
    mean_relation_loss = sum(epoch_relation_losses) / len(epoch_relation_losses)

    return mean_total_loss, mean_class_loss, mean_relation_loss


def check_convergence(current_loss: float, previous_loss: float, threshold: float = 0.01) -> bool:
    """
    Checks if training has converged based on relative change in loss.

    Args:
        current_loss:   Current epoch's loss
        previous_loss:  Previous epoch's loss
        threshold:      Relative change threshold for convergence

    Returns:
        True if loss has converged
    """

    # Avoid division by zero
    if previous_loss == 0:
        return False

    # Compute relative change
    relative_change = abs(current_loss - previous_loss) / previous_loss

    # Check if below threshold
    return relative_change < threshold


@hydra.main(version_base=None, config_path="../../configs/rrn", config_name="config")
def train(
    cfg: DictConfig,
) -> None:
    """
    Main training function.

    Args:
        nb_kgs              : Number of knowledge graphs to use for training
        data_dir            : Directory containing knowledge graph data
        embedding_size      : Dimensionality of entity embeddings
        iterations          : Number of RRN update iterations
        learning_rate       : Learning rate for optimizer
        weight_decay        : L2 regularization strength
        patience            : Number of epochs to wait for improvement before stopping
        max_epochs          : Maximum number of training epochs
        checkpoint_path     : Directory for checkpoints
        loader_workers      : Number of parallel workers for loading KGs (default: CPU count - 1)
        verbose             : Whether to print batch-level progress
        checkpoint_subdir   : Subdirectory for this training run's checkpoints
    """

    # ---------------------------------------------------------------------------- #
    #                                     SETUP                                    #
    # ---------------------------------------------------------------------------- #

    device = get_device()
    print(f"Training on device: {device}")

    # ------------------------ Load graphs and preprocess ------------------------ #

    print("Loading knowledge graphs...")

    # Decide worker count; default to CPU count - 1 (>=1)
    # workers = loader_workers if loader_workers is not None else max(1, (os.cpu_count() or 1) - 1)

    kg_list = load_knowledge_graphs(cfg.data_dir)

    # Use nb_kgs KnowledgeGraphs for training
    train_kgs = kg_list[0:nb_kgs]  # exclusive end index

    # Preprocess graphs:
    #   KG -> dict with:
    #       - base_triples
    #       - base_memberships
    #       - inferred_triples
    #       - inferred_memberships
    #       - all_triples
    #       - all_memberships
    print("Preprocessing knowledge graphs...")
    processed_data = [preprocess_knowledge_graph(kg) for kg in train_kgs]

    # ---------------------- Initialize RNN embedding model ---------------------- #

    # Initialize model using first knowledge graph
    #   -> the ontology \Sigma stays the same across all graphs
    #   -> only the data D_i changes
    sample_kg = train_kgs[0]

    print(f"Initializing RRN with {len(sample_kg.classes)} classes and {len(sample_kg.relations)} relations")

    rrn, mlps = initialize_model(
        embedding_size=embedding_size,
        iterations=iterations,
        reference_kg=sample_kg,
        device=device,
    )

    # ------------------------------ Setup optimizer ----------------------------- #

    # Initialize optimizer (Adam as in paper)
    optimizer = optim.Adam(
        params=list(rrn.parameters()) + list(mlps.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # --------------------------- Setup training loop --------------------------- #

    # Prepare data loader with batch size 1 (1 KG per batch)
    dataloader = DataLoader(
        processed_data,
        batch_size=1,
        shuffle=True,
        collate_fn=custom_collate_fn,  # always returns batch size of 1
    )

    print("Starting training...\n")
    print("=" * 70)

    # Setup training loop variables
    epoch = 1
    previous_total_loss = float("inf")
    convergence_counter = 0
    best_loss = float("inf")

    # Checkpoint paths
    parent_checkpoint_path = checkpoint_path
    child_checkpoint_path = os.path.join(parent_checkpoint_path, checkpoint_subdir)

    # Print current absolute checkpoint path
    print(f"Checkpoint path: {os.path.abspath(child_checkpoint_path)}")

    # Create child checkpoint directory
    os.makedirs(child_checkpoint_path, exist_ok=True)

    # ---------------------------------------------------------------------------- #
    #                                 TRAINING LOOP                                #
    # ---------------------------------------------------------------------------- #

    while epoch <= max_epochs:
        print(f"\nEpoch {epoch}/{max_epochs}")
        print("-" * 70)

        # Train for one epoch
        mean_total_loss, mean_class_loss, mean_relation_loss = train_epoch(
            rrn=rrn,
            mlps=mlps,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            use_amp=(device.type == "cuda"),
            verbose=verbose,
        )

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Total Loss:    {mean_total_loss:.6f}")
        print(f"  Class Loss:    {mean_class_loss:.6f}")
        print(f"  Relation Loss: {mean_relation_loss:.6f}")

        # ----------------------------- SAVE CHECKPOINTS ----------------------------- #

        # Save embedding model checkpoint
        save_checkpoint(rrn, f"{child_checkpoint_path}/rrn_epoch-{epoch}.pth")

        # Save MLP checkpoints
        for mlp_idx, mlp in enumerate(mlps):
            save_checkpoint(mlp, f"{child_checkpoint_path}/mlp{mlp_idx}_epoch-{epoch}.pth")

        # -------------------------------- CONVERGED? -------------------------------- #

        # Track best model
        if mean_total_loss < best_loss:
            best_loss = mean_total_loss
            print("  *** New best loss! ***")

        # Check for convergence
        if check_convergence(mean_total_loss, previous_total_loss):
            convergence_counter += 1
            print(f"  Convergence: {convergence_counter}/{patience}")
        else:
            convergence_counter = 0

        # Early stopping
        if convergence_counter >= patience:
            print(f"\n{'=' * 70}")
            print(f"Training converged after {epoch} epochs!")
            print(f"Best loss: {best_loss:.6f}")
            print("=" * 70)
            break

        previous_total_loss = mean_total_loss
        epoch += 1

    if epoch > max_epochs:
        print(f"\n{'=' * 70}")
        print(f"Reached maximum epochs ({max_epochs})")
        print(f"Best loss: {best_loss:.6f}")
        print("=" * 70)

    print("\nTraining complete!")


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    # First argument is checkpoint subdirectory, e.g. the date
    if len(sys.argv) < 2:
        print("Usage: python train.py <checkpoint_subdir>")
        sys.exit(1)

    checkpoint_subdir = sys.argv[1]

    # Get the base data directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"Base dir: {BASE_DIR}")

    data_dir = os.path.join(BASE_DIR, "data/family-tree/out-reldata/train-200")

    print(f"Data directory: {data_dir}")

    # Example usage
    train(
        nb_kgs=200,
        data_dir=data_dir,
        embedding_size=EMBEDDING_SIZE,
        iterations=ITERATIONS,
        learning_rate=0.001,
        weight_decay=1e-6,
        patience=10,
        verbose=VERBOSE,
        checkpoint_subdir=checkpoint_subdir,
    )
