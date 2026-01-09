"""
DESCRIPTION

    Test script of the revised Recursive Reasoning Network (RRN) implementation
    with batched relation updates, which allows for maximal GPU utilization.

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
import os
import sys
from collections import defaultdict
from pathlib import Path

# Standard libraries
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from device import get_device
from globals import EMBEDDING_SIZE, ITERATIONS, TEST_BASE_FACTS, VERBOSE
from rrn_model_batched import RRN, ClassesMLP
from tt_common import initialize_model

from apps.RRN.data import load_knowledge_graphs, preprocess_knowledge_graph

# Own modules
from data_structures import KnowledgeGraph, Triple


def load_model_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """
    Loads model weights from checkpoint.

    Args:
        model           : PyTorch model to load weights into
        checkpoint_path : Path to checkpoint file
        device          : Device to load model on
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))


def get_latest_checkpoint_epoch(checkpoint_dir: Path) -> int:
    """
    Finds the latest checkpoint epoch in the given directory.
    This is handy if all subdirectories in the checkpoint/ directory are created based on the job date,
    like I did on the DTU HPC cluster.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Latest epoch number found in checkpoint filenames
    """

    # Scan directory for checkpoint files
    max_epoch = -1
    for file in checkpoint_dir.iterdir():
        if file.suffix == ".pth":
            parts = file.stem.split("_epoch-")
            if len(parts) == 2 and parts[1].isdigit():
                epoch = int(parts[1])
                if epoch > max_epoch:
                    max_epoch = epoch

    # No checkpoints found
    if max_epoch == -1:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")

    return max_epoch


def load_trained_model(
    checkpoint_dir: str,
    checkpoint_epoch: int,
    reference_kg: KnowledgeGraph,
    embedding_size: int,
    iterations: int,
    device: torch.device,
) -> Tuple[RRN, nn.ModuleList]:
    """
    Loads a trained RRN model and its associated MLPs from checkpoints.

    This function reconstructs the model architecture based on a reference
    knowledge graph (which provides the ontology structure) and then loads
    the trained weights.

    Args:
        checkpoint_dir  : Directory containing checkpoint files
        checkpoint_epoch: Epoch number of checkpoint to load
        reference_kg    : Knowledge graph with the same ontology structure
        embedding_size  : Dimensionality of entity embeddings
        iterations      : Number of RRN message-passing iterations
        device          : Device to load models on

    Returns:
        Tuple of (RRN model, ModuleList of MLPs)
    """

    # Check if checkpoint number is provided
    if checkpoint_epoch is None:
        exit("Checkpoint epoch must be specified to load the model.")

    # ------------------------------ INITIALIZE RRN ------------------------------ #

    rrn, mlps = initialize_model(
        embedding_size=embedding_size,
        iterations=iterations,
        reference_kg=reference_kg,
        device=device,
    )

    # ---------------------------- RESTORE RNN WEIGHTS --------------------------- #

    # if checkpoint_epoch is None, use the latest checkpoint
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_epoch is None:
        checkpoint_epoch = get_latest_checkpoint_epoch(checkpoint_path)

    print(f"Loading checkpoint from epoch {checkpoint_epoch}...")

    rrn_checkpoint = checkpoint_path / f"rrn_epoch-{checkpoint_epoch}.pth"
    if not rrn_checkpoint.exists():
        raise FileNotFoundError(f"RRN checkpoint not found: {rrn_checkpoint}")
    load_model_checkpoint(rrn, str(rrn_checkpoint), device)

    # Load MLP weights
    for mlp_idx, mlp in enumerate(mlps):
        mlp_checkpoint = checkpoint_path / f"mlp{mlp_idx}_epoch-{checkpoint_epoch}.pth"
        if not mlp_checkpoint.exists():
            raise FileNotFoundError(f"MLP checkpoint not found: {mlp_checkpoint}")
        load_model_checkpoint(mlp, str(mlp_checkpoint), device)

    return rrn, mlps


def evaluate_classes(
    mlp: ClassesMLP,
    embeddings: torch.Tensor,
    membership_labels: List[List[int]],
    device: torch.device,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates class membership predictions.

    This version processes all individuals at once and correctly
    masks unknown labels (0.5) from accuracy calculations.

    Args:
        mlp                 : Class prediction MLP
        embeddings          : Individual embeddings from RRN   (num_individuals, embedding_dim)
        membership_labels   : Ground truth membership labels   (list of lists of -1, 0, 1)
        device              : Device to run evaluation on

    Returns:
        Tuple of (accuracies, all_known_scores, positive_scores, negative_scores)
    """

    # Convert {-1, 0, 1} labels to {0.0, 0.5, 1.0}
    # where 0.5 = unknown, 1.0 = positive, 0.0 = negative
    #
    # member_of     <-> label = -1.0 <-> target = 1.0
    # ~member_of    <-> label = 0.0  <-> target = 0.0
    cls_targets = (torch.as_tensor(membership_labels, dtype=torch.float32, device=device) + 1) / 2

    # Forward pass to get logits (num_individuals, num_classes)
    cls_logits = mlp(embeddings)

    # Convert logits to probabilities
    cls_probs = torch.sigmoid(cls_logits)

    # Generate actual predictions (rounded probabilities)
    cls_pred = cls_probs.round()
    # sigmoid(MLP_logit) = P(<s, member_of, C_i> | KB)
    # P(<s, ~member_of, C_i> | KB) = 1 - P(<s, member_of, C_i> | KB)
    #
    # So, all in all, this means that:
    #   round(sigmoid(MLP_logit)) = 1.0 <-> predicted member_of
    #   round(sigmoid(MLP_logit)) = 0.0 <-> predicted ~member_of

    # Calculate scores
    #   -> True/False where prediction matches label or not
    scores = (cls_pred == cls_targets).float()
    # Still (num_individuals, num_classes) shape

    # Create masks for known positive, known negative, and all known labels
    pos_mask = cls_targets == 1.0
    neg_mask = cls_targets == 0.0

    # Mask to filter out the "unknown" (0.5) labels
    known_mask = cls_targets != 0.5

    # Get the scores for each category (True/False tensors)
    positive_scores = scores[pos_mask]
    negative_scores = scores[neg_mask]
    all_known_scores = scores[known_mask]

    # Compute mean accuracies, handling empty tensors
    class_accuracies = {
        "all": all_known_scores.mean().item() if all_known_scores.numel() > 0 else float("nan"),
        "positive": positive_scores.mean().item() if positive_scores.numel() > 0 else float("nan"),
        "negative": negative_scores.mean().item() if negative_scores.numel() > 0 else float("nan"),
    }

    return class_accuracies


def evaluate_triples(
    mlps: nn.ModuleList,
    embeddings: torch.Tensor,
    triples: List[Triple],
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluates relation predictions using batched processing for speed.

    Instead of iterating through triples one by one, we group them by
    predicate index. This allows us to pass all subject/object pairs
    for a specific relation into the MLP in a single tensor operation.

    Args:
        mlps      : List of MLPs (index 0 is classes, indices 1..R are relations)
        embeddings: Entity embeddings from RRN (num_individuals, embedding_dim)
        triples   : List of relation triples to evaluate
        device    : Device to run evaluation on

    Returns:
        Dictionary of accuracies with 'all', 'positive', 'negative' keys.
    """
    # 1. Group triples by predicate index
    #    We need to do this because different predicates use different MLP heads.
    grouped_triples = defaultdict(list)
    for t in triples:
        grouped_triples[t.predicate.index].append(t)

    # buffers to collect boolean hits (1.0 if correct, 0.0 if wrong)
    all_hits = []
    pos_hits = []
    neg_hits = []

    with torch.no_grad():
        for pred_idx, group in grouped_triples.items():
            # 2. Prepare batch tensors
            # Extract indices and targets for this group
            s_indices = [t.subject.index for t in group]
            o_indices = [t.object.index for t in group]

            # Target: 1.0 for positive, 0.0 for negative
            # unsqueeze(1) makes shape (Batch, 1) to match MLP output
            targets = torch.tensor(
                [1.0 if t.positive else 0.0 for t in group],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1)

            # 3. Fetch Embeddings in batch
            # Shape: (Batch_Size, Embedding_Dim)
            batch_s_emb = embeddings[s_indices]
            batch_o_emb = embeddings[o_indices]

            # 4. Forward Pass
            # Remember: mlps[0] is classes, so relation MLPs start at index + 1
            logits = mlps[pred_idx + 1](batch_s_emb, batch_o_emb)
            predictions = torch.sigmoid(logits).round()

            # 5. Compare matches
            # hits is a tensor of 1s (correct) and 0s (wrong)
            hits = (predictions == targets).float().squeeze(1)

            # 6. Separate positive and negative performance for metrics
            # We need to map back to the original list to know which were pos/neg
            # Construct masks based on the targets we created earlier
            is_positive = targets.squeeze(1) == 1.0
            is_negative = targets.squeeze(1) == 0.0

            all_hits.append(hits)
            if is_positive.any():
                pos_hits.append(hits[is_positive])
            if is_negative.any():
                neg_hits.append(hits[is_negative])

    # 7. Aggregate results
    # Concatenate list of tensors into single tensors
    all_hits_t = torch.cat(all_hits) if all_hits else torch.tensor([])
    pos_hits_t = torch.cat(pos_hits) if pos_hits else torch.tensor([])
    neg_hits_t = torch.cat(neg_hits) if neg_hits else torch.tensor([])

    # Calculate means (accuracies)
    triple_accuracies = {
        "all": all_hits_t.mean().item() if all_hits_t.numel() > 0 else float("nan"),
        "positive": pos_hits_t.mean().item() if pos_hits_t.numel() > 0 else float("nan"),
        "negative": neg_hits_t.mean().item() if neg_hits_t.numel() > 0 else float("nan"),
    }

    return triple_accuracies


def test_on_knowledge_graph(
    rrn: RRN,
    mlps: nn.ModuleList,
    test_kg: KnowledgeGraph,
    device: torch.device,
    verbose: bool = True,
    test_base_facts: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Tests the model, trained on an ontology Σ, on a single knowledge graph D_i.

    Args:
        rrn                 : Trained RRN model
        mlps                : Trained MLP classifiers
        test_kg             : Knowledge graph to test on (must share ontology with training)
        device              : Device to run evaluation on
        verbose             : Whether to print detailed progress
        test_base_facts     : Whether to test on base (training) facts as well, next to inferred facts

    Returns:
        Tuple of (class_accuracies, triple_accuracies),
        both dictionaries with 'all', 'positive', 'negative' accuracies
    """

    # Set to evaluation mode
    rrn.eval()
    mlps.eval()

    # Preprocess the test knowledge graph
    preprocessed_data = preprocess_knowledge_graph(test_kg)
    base_triples = preprocessed_data["base_triples"]
    base_memberships = preprocessed_data["base_memberships"]
    inferred_triples = preprocessed_data["inferred_triples"]
    inferred_memberships = preprocessed_data["inferred_memberships"]
    all_triples = preprocessed_data["all_triples"]
    all_memberships = preprocessed_data["all_memberships"]

    print("There are...")
    print(f"  {len(base_triples)} BASE triples")
    print(f"  {len(inferred_triples)} INFERRED triples")
    print(f"  {len(all_triples)} TOTAL triples")
    print(f"  {len(base_memberships)} BASE memberships")
    print(f"  {len(inferred_memberships)} INFERRED memberships")
    print(f"  {len(all_memberships)} TOTAL memberships")

    print(f"Running Evaluation on {'BASE + INFERRED facts' if test_base_facts else 'INFERRED facts ONLY'}")

    # If testing on base facts as well, set target to ALL facts,
    # else only on inferred facts.
    if test_base_facts:
        print("Setting TARGET = ALL facts (base + inferred)")
        target_triples = all_triples
        target_memberships = all_memberships
    else:
        print("Setting TARGET = INFERRED facts ONLY")
        target_triples = inferred_triples
        target_memberships = inferred_memberships

    if verbose:
        print("Test KG Statistics:")
        print(f"  Individuals: {len(test_kg.individuals)}")
        print(f"  Triples:    {len(target_triples)}")
        print(f"        Of which positive triples: {sum(1 for t in target_triples if t.positive)}")
        print(f"        Of which negative triples: {sum(1 for t in target_triples if not t.positive)}")
        print(f"  Memberships: {len(target_memberships)}")
        print(
            f"      Of which positive memberships: {sum(sum(1 for v in labels if v == 1) for labels in target_memberships)}"
        )
        print(
            f"      Of which negative memberships: {sum(sum(1 for v in labels if v == -1) for labels in target_memberships)}"
        )

    # ------------------------------- RUN INFERENCE ------------------------------ #

    # Disable gradient calculations for evaluation
    with torch.no_grad():
        # Generate embeddings using RRN with BASE FACTS!
        print("Initializing RRN with BASE facts...")
        embeddings = rrn(base_triples, base_memberships).to(device)

        # Evaluate class membership predictions on target memberships
        class_accuracies = evaluate_classes(mlps[0], embeddings, target_memberships, device)

        # Evaluate relation predictions on target triples
        triple_accuracies = evaluate_triples(mlps, embeddings, target_triples, device)
    return class_accuracies, triple_accuracies


def test_model(
    checkpoint_dir: str,
    checkpoint_epoch: int,
    test_data_dir: str,
    test_indices: Optional[List[int]] = None,
    embedding_size: int = 100,
    iterations: int = 7,
    verbose: bool = True,
    test_base_facts: bool = True,
) -> dict:
    """
    Main testing function that loads a trained model and evaluates it on test data.

    This function:
    1. Loads the trained RRN and MLPs from checkpoints
    2. Loads test knowledge graphs from the specified directory
    3. Evaluates the model on the test graphs
    4. Returns detailed results

    Args:
        checkpoint_dir  : Subdirectory containing model checkpoints
        checkpoint_epoch: Epoch number of checkpoint to load
        test_data_dir   : Directory containing test knowledge graphs
        test_indices    : Optional list of specific KG indices to test on (if None, tests on all)
        embedding_size  : Dimensionality of entity embeddings
        iterations      : Number of RRN message-passing iterations
        verbose         : Whether to print detailed progress
        test_base_facts : Whether to test on base (training) facts as well, next to inferred facts

    Returns:
        Dictionary containing test results and statistics
    """

    # Select device
    device = get_device()

    if verbose:
        print(f"Testing on device: {device}\n")
        print("=" * 70)

    # ------------------------------ LOAD TEST DATA ------------------------------ #

    if verbose:
        print("Loading test knowledge graphs...")

    test_kgs = load_knowledge_graphs(test_data_dir)

    # Select specific KGs if indices provided,
    # otherwise use all of them.
    if test_indices is not None:
        test_kgs = [test_kgs[i] for i in test_indices]
        if verbose:
            print(f"Selected {len(test_kgs)} knowledge graphs for testing")

    if len(test_kgs) == 0:
        raise ValueError("No test knowledge graphs found!")

    # -------------------------------- LOAD MODEL -------------------------------- #
    # -> use first KG to define structure
    # -> create models (RRN + MLPs)
    # -> load weights from checkpoints

    # if checkpoint_epoch is None, use the latest checkpoint
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_epoch is None:
        checkpoint_epoch = get_latest_checkpoint_epoch(checkpoint_path)

    # Use first KG to determine ontology structure
    reference_kg = test_kgs[0]

    if verbose:
        print("\nOntology Structure:")
        print(f"  Classes: {len(reference_kg.classes)}")
        print(f"  Relations: {len(reference_kg.relations)}")

    # Load trained model
    if verbose:
        print(f"\nLoading model from epoch {checkpoint_epoch}...")

    rrn, mlps = load_trained_model(
        checkpoint_dir=checkpoint_dir,
        checkpoint_epoch=checkpoint_epoch,
        reference_kg=reference_kg,
        embedding_size=embedding_size,
        iterations=iterations,
        device=device,
    )

    if verbose:
        print("Model loaded successfully!")

    # ------------------------------ START EVALUATION ------------------------------ #

    # Test on all knowledge graphs
    results = {
        # Classes
        "all_class_accuracies": [],
        "negative_class_accuracies": [],
        "positive_class_accuracies": [],
        # Triples
        "all_triple_accuracies": [],
        "positive_triple_accuracies": [],
        "negative_triple_accuracies": [],
        # KB stats
        "num_individuals": [],
        "num_triples": [],
    }

    if test_base_facts:
        what_on = "BASE + INFERRED facts"
    else:
        what_on = "INFERRED facts ONLY"

    if verbose:
        print("\n" + "=" * 70)
        print(f"Running Evaluation on {what_on}")
        print("=" * 70)

    for idx, test_kg in enumerate(test_kgs):
        if verbose:
            print(f"\nTesting on KG {idx + 1}/{len(test_kgs)}...")

        class_accuracies, triple_accuracies = test_on_knowledge_graph(
            rrn=rrn,
            mlps=mlps,
            test_kg=test_kg,
            device=device,
            verbose=verbose,
            test_base_facts=test_base_facts,
        )

        # Overall accuracies
        results["all_class_accuracies"].append(class_accuracies["all"])
        results["positive_class_accuracies"].append(class_accuracies["positive"])
        results["negative_class_accuracies"].append(class_accuracies["negative"])
        results["all_triple_accuracies"].append(triple_accuracies["all"])
        results["positive_triple_accuracies"].append(triple_accuracies["positive"])
        results["negative_triple_accuracies"].append(triple_accuracies["negative"])

        # KB stats
        results["num_individuals"].append(len(test_kg.individuals))
        results["num_triples"].append(len(test_kg.triples))

        if verbose:
            print(f"  Class Accuracy:  {class_accuracies['all']:.4f} ({class_accuracies['all'] * 100:.2f}%)")
            print(f"    Positive:      {class_accuracies['positive']:.4f} ({class_accuracies['positive'] * 100:.2f}%)")
            print(f"    Negative:      {class_accuracies['negative']:.4f} ({class_accuracies['negative'] * 100:.2f}%)")
            print(f"  Triple Accuracy: {triple_accuracies['all']:.4f} ({triple_accuracies['all'] * 100:.2f}%)")
            print(
                f"    Positive:      {triple_accuracies['positive']:.4f} ({triple_accuracies['positive'] * 100:.2f}%)"
            )
            print(
                f"    Negative:      {triple_accuracies['negative']:.4f} ({triple_accuracies['negative'] * 100:.2f}%)"
            )

    # --------------------------- COMPUTE SUMMARY STATS -------------------------- #

    results["mean_class_accuracy"] = sum(results["all_class_accuracies"]) / len(results["all_class_accuracies"])
    results["mean_positive_class_accuracy"] = sum(results["positive_class_accuracies"]) / len(
        results["positive_class_accuracies"]
    )
    results["mean_negative_class_accuracy"] = sum(results["negative_class_accuracies"]) / len(
        results["negative_class_accuracies"]
    )
    results["mean_triple_accuracy"] = sum(results["all_triple_accuracies"]) / len(results["all_triple_accuracies"])
    results["mean_positive_triple_accuracy"] = sum(results["positive_triple_accuracies"]) / len(
        results["positive_triple_accuracies"]
    )
    results["mean_negative_triple_accuracy"] = sum(results["negative_triple_accuracies"]) / len(
        results["negative_triple_accuracies"]
    )

    results["min_class_accuracy"] = min(results["all_class_accuracies"])
    results["min_positive_class_accuracy"] = min(results["positive_class_accuracies"])
    results["min_negative_class_accuracy"] = min(results["negative_class_accuracies"])
    results["min_triple_accuracy"] = min(results["all_triple_accuracies"])
    results["min_positive_triple_accuracy"] = min(results["positive_triple_accuracies"])
    results["min_negative_triple_accuracy"] = min(results["negative_triple_accuracies"])
    results["max_class_accuracy"] = max(results["all_class_accuracies"])
    results["max_positive_class_accuracy"] = max(results["positive_class_accuracies"])
    results["max_negative_class_accuracy"] = max(results["negative_class_accuracies"])
    results["max_triple_accuracy"] = max(results["all_triple_accuracies"])
    results["max_positive_triple_accuracy"] = max(results["positive_triple_accuracies"])
    results["max_negative_triple_accuracy"] = max(results["negative_triple_accuracies"])

    if verbose:
        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)
        print(f"Number of test KGs:        {len(test_kgs)}")

        print(f"\nClass Membership Predictions ({what_on}):")
        print(
            f"  Mean Accuracy:           {results['mean_class_accuracy']:.4f} ({results['mean_class_accuracy'] * 100:.2f}%)"
        )
        print(f"  Accuracy Range:          {results['min_class_accuracy']:.4f} - {results['max_class_accuracy']:.4f}")
        print(
            f"  Mean Positive Accuracy:  {results['mean_positive_class_accuracy']:.4f} ({results['mean_positive_class_accuracy'] * 100:.2f}%)"
        )
        print(
            f"  Positive Accuracy Range: {results['min_positive_class_accuracy']:.4f} - {results['max_positive_class_accuracy']:.4f}"
        )
        print(
            f"  Mean Negative Accuracy:  {results['mean_negative_class_accuracy']:.4f} ({results['mean_negative_class_accuracy'] * 100:.2f}%)"
        )
        print(
            f"  Negative Accuracy Range: {results['min_negative_class_accuracy']:.4f} - {results['max_negative_class_accuracy']:.4f}"
        )

        print(f"\nRelation Predictions ({what_on}):")
        print(
            f"  Mean Accuracy:           {results['mean_triple_accuracy']:.4f} ({results['mean_triple_accuracy'] * 100:.2f}%)"
        )
        print(f"  Accuracy Range:          {results['min_triple_accuracy']:.4f} - {results['max_triple_accuracy']:.4f}")
        print(
            f"  Mean Positive Accuracy:  {results['mean_positive_triple_accuracy']:.4f} ({results['mean_positive_triple_accuracy'] * 100:.2f}%)"
        )
        print(
            f"  Positive Accuracy Range: {results['min_positive_triple_accuracy']:.4f} - {results['max_positive_triple_accuracy']:.4f}"
        )
        print(
            f"  Mean Negative Accuracy:  {results['mean_negative_triple_accuracy']:.4f} ({results['mean_negative_triple_accuracy'] * 100:.2f}%)"
        )
        print(
            f"  Negative Accuracy Range: {results['min_negative_triple_accuracy']:.4f} - {results['max_negative_triple_accuracy']:.4f}"
        )
        print("=" * 70)

    return results


# ---------------------------------------------------------------------------- #
#                               MAIN ENTRY POINT                               #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ----------------------- CHECKPOINT AND DATA DIRECTORY ---------------------- #

    # First argument is checkpoint dir
    if len(sys.argv) < 2:
        print("Usage: python test.py <checkpoint_dir>")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]

    print("\n" + "=" * 70)
    print(f"Checkpoint Directory: {checkpoint_dir}")

    # Get the base data directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data/family-tree/out-reldata/test-20")

    # print to std error
    print(f"Data directory: {data_dir}")

    # --------------------------------- TEST ON N GRAPHS -------------------------------- #

    print("\n\n" + "=" * 70)
    print("Testing on multiple knowledge graphs")
    print("=" * 70 + "\n")

    results = test_model(
        checkpoint_dir=checkpoint_dir,
        checkpoint_epoch=None,  # Use latest checkpoint
        test_data_dir=data_dir,
        test_indices=None,  # Test on all KGs
        embedding_size=EMBEDDING_SIZE,
        iterations=ITERATIONS,
        verbose=VERBOSE,
        test_base_facts=TEST_BASE_FACTS,
    )

    # write results to a file
    with open("test_results.txt", "w") as f:
        f.write("Test Results:\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
