from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

# Import your existing models and structures
# Assuming these are in the python path or relative imports work
from synthology.data_structures import Class, Relation, Triple

from .rrn_batched import RRN as RRNBatched
from .rrn_common import ClassesMLP, RelationMLP
from .rrn_exact import RRN as RRNExact


class RRNSystem(pl.LightningModule):
    """
    PyTorch Lightning Module for the Recursive Reasoning Network.

    Handles:
    - Model initialization (Batched vs Exact)
    - Forward pass (Message passing)
    - Loss computation (Class + Relation with dynamic weighting)
    """

    def __init__(self, cfg: DictConfig, classes: List[Class], relations: List[Relation]):
        """
        Args:
            cfg: Hydra configuration dictionary.
            reference_kg: A sample KnowledgeGraph to initialize shapes (classes/relations).
        """
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # ---------------------------------------------------------------------------- #
        #                               MODEL SELECTION                                #
        # ---------------------------------------------------------------------------- #

        embedding_size = cfg.model.embedding_size
        iterations = cfg.model.iterations

        # Choose implementation based on config
        model_type = cfg.model.get("type", "batched")

        if model_type == "batched":
            self.rrn = RRNBatched(
                embedding_size=embedding_size,
                iterations=iterations,
                classes=classes,
                relations=relations,
            )
        elif model_type == "exact":
            self.rrn = RRNExact(
                embedding_size=embedding_size,
                iterations=iterations,
                classes=classes,
                relations=relations,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'batched' or 'exact'.")

        # ------------------------------ INITIALIZE MLPs ----------------------------- #

        # Initialize MLPs
        self.mlps = nn.ModuleList()

        # MLP^{C_i} = P(<s, member_of, o> | KB)
        self.mlps.append(ClassesMLP(embedding_size, len(classes)))

        # One MLP per relation type (positive or negative predicate)
        for _ in relations:
            # MLP^{R_i} = P(<s, R_i, o> | KB)
            self.mlps.append(RelationMLP(embedding_size))

    def forward(self, triples: List[Triple], memberships: List[List[int]]):
        """
        Forward pass through the RRN to generate embeddings.
        """
        return self.rrn(triples, memberships)

    def configure_optimizers(self):
        """
        Configure optimizer (Adam) and optional schedulers.
        """

        optimizer = instantiate(self.cfg.hyperparams.optimizer, params=self.parameters())
        logger.info(f"Using optimizer: {self.cfg.hyperparams.optimizer._target_}")

        return optimizer

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Override default behavior.
        Since our batch contains custom objects (Triples) and the Model
        handles .to(device) manually in the forward pass, we skip
        Lightning's automatic device transfer here.
        """
        return batch

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Single training step.

        Since 1 Batch = 1 KG, 'batch' is the dictionary returned by custom_collate_fn.
        """

        # Unpack data
        base_triples = batch["base_triples"]
        base_memberships = batch["base_memberships"]

        # ! CRUCIAL: We compute loss on ALL triples and ALL memberships
        # (both base facts and inferred)
        all_triples = batch["all_triples"]
        all_memberships = batch["all_memberships"]

        # 1. RRN Forward pass (Message Passing)
        #    Only uses base facts to generate embeddings
        embeddings = self(base_triples, base_memberships)

        # 2. MLP Predictions & Loss Calculation
        total_loss, class_loss, relation_loss = self._compute_loss(
            embeddings=embeddings, triples=all_triples, membership_targets=all_memberships
        )

        # 3. Logging
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/class_loss", class_loss, on_step=False, on_epoch=True, batch_size=1)
        self.log("train/relation_loss", relation_loss, on_step=False, on_epoch=True, batch_size=1)

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Single validation step for a Knowledge Graph.
        """
        # Unpack Data
        base_triples = batch["base_triples"]
        base_memberships = batch["base_memberships"]

        # In validation, we typically want to measure performance on ALL known facts
        if self.cfg.test_base_facts:
            target_triples = batch["all_triples"]
            target_memberships = batch["all_memberships"]
        else:
            target_triples = batch["inferred_triples"]
            target_memberships = batch["inferred_memberships"]

        # 1. RRN Forward pass (using base facts)
        embeddings = self(base_triples, base_memberships)

        # 2. Compute Loss (validation loss)
        val_loss, val_class_loss, val_rel_loss = self._compute_loss(
            embeddings=embeddings, triples=target_triples, membership_targets=target_memberships
        )

        # 3. Evaluate Metrics
        class_metrics = self._evaluate_classes(embeddings, target_memberships)
        triple_metrics = self._evaluate_triples(embeddings, target_triples)

        # 4. Log everything
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("val/total_loss", val_loss, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/class_loss", val_class_loss, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/relation_loss", val_rel_loss, on_step=False, on_epoch=True, batch_size=1)

        # Compute accuracies with PyTorch
        val_acc = class_metrics.get("acc_all", float("nan"))
        if not torch.isnan(torch.tensor(val_acc)):
            self.log("val/class_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

        for key, value in class_metrics.items():
            self.log(f"val/class_{key}", value, on_step=False, on_epoch=True, batch_size=1)

        for key, value in triple_metrics.items():
            self.log(f"val/triple_{key}", value, on_step=False, on_epoch=True, batch_size=1)

        return val_loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Single test step for a Knowledge Graph.
        Replicates logic from the original test.py.
        """
        # Unpack Data
        base_triples = batch["base_triples"]
        base_memberships = batch["base_memberships"]

        if self.cfg.test_base_facts:
            target_triples = batch["all_triples"]
            target_memberships = batch["all_memberships"]
        else:
            target_triples = batch["inferred_triples"]
            target_memberships = batch["inferred_memberships"]

        # 3. RRN Inference (Always using BASE facts)
        embeddings = self(base_triples, base_memberships)

        # 4. Evaluate Class Predictions
        class_metrics = self._evaluate_classes(embeddings=embeddings, membership_labels=target_memberships)

        # 5. Evaluate Relation Predictions
        triple_metrics = self._evaluate_triples(embeddings=embeddings, triples=target_triples)

        # 6. Logging
        # Lightning automatically accumulates these over the epoch
        for key, value in class_metrics.items():
            self.log(f"test/class_{key}", value, on_step=False, on_epoch=True)

        for key, value in triple_metrics.items():
            self.log(f"test/triple_{key}", value, on_step=False, on_epoch=True)

    def _evaluate_classes(
        self,
        embeddings: torch.Tensor,
        membership_labels: List[List[int]],
    ) -> Dict[str, float]:
        """
        Helper: Evaluates class membership predictions.
        """
        device = self.device

        # Convert {-1, 0, 1} labels to targets {0.0, 0.5, 1.0}
        cls_targets = (torch.as_tensor(membership_labels, dtype=torch.float32, device=device) + 1) / 2

        # Forward pass (mlps[0] is always class MLP)
        cls_logits = self.mlps[0](embeddings)
        cls_probs = torch.sigmoid(cls_logits)
        cls_pred = cls_probs.round()

        # Calculate hits
        scores = (cls_pred == cls_targets).float()

        # Create masks
        pos_mask = cls_targets == 1.0
        neg_mask = cls_targets == 0.0
        known_mask = cls_targets != 0.5

        # Get scores
        positive_scores = scores[pos_mask]
        negative_scores = scores[neg_mask]
        all_known_scores = scores[known_mask]

        # Return dict, avoiding empty tensor errors
        return {
            "acc_all": all_known_scores.mean().item() if all_known_scores.numel() > 0 else float("nan"),
            "acc_pos": positive_scores.mean().item() if positive_scores.numel() > 0 else float("nan"),
            "acc_neg": negative_scores.mean().item() if negative_scores.numel() > 0 else float("nan"),
        }

    def _evaluate_triples(
        self,
        embeddings: torch.Tensor,
        triples: List[Triple],
    ) -> Dict[str, float]:
        """
        Helper: Evaluates relation predictions using batched processing.
        """
        device = self.device

        # Group triples by predicate
        grouped_triples = defaultdict(list)
        for t in triples:
            grouped_triples[t.predicate.index].append(t)

        all_hits = []
        pos_hits = []
        neg_hits = []

        for pred_idx, group in grouped_triples.items():
            s_indices = [t.subject.index for t in group]
            o_indices = [t.object.index for t in group]

            # Targets: 1.0 positive, 0.0 negative
            targets = torch.tensor(
                [1.0 if t.positive else 0.0 for t in group],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1)

            # Get embeddings
            batch_s_emb = embeddings[s_indices]
            batch_o_emb = embeddings[o_indices]

            # Forward Pass (mlps[pred_idx + 1] is relation MLP)
            logits = self.mlps[pred_idx + 1](batch_s_emb, batch_o_emb)
            predictions = torch.sigmoid(logits).round()

            # Compare matches
            hits = (predictions == targets).float().squeeze(1)

            # Separate hits
            is_positive = targets.squeeze(1) == 1.0
            is_negative = targets.squeeze(1) == 0.0

            all_hits.append(hits)
            if is_positive.any():
                pos_hits.append(hits[is_positive])
            if is_negative.any():
                neg_hits.append(hits[is_negative])

        # Aggregate results
        all_hits_t = torch.cat(all_hits) if all_hits else torch.tensor([], device=device)
        pos_hits_t = torch.cat(pos_hits) if pos_hits else torch.tensor([], device=device)
        neg_hits_t = torch.cat(neg_hits) if neg_hits else torch.tensor([], device=device)

        return {
            "acc_all": all_hits_t.mean().item() if all_hits_t.numel() > 0 else float("nan"),
            "acc_pos": pos_hits_t.mean().item() if pos_hits_t.numel() > 0 else float("nan"),
            "acc_neg": neg_hits_t.mean().item() if neg_hits_t.numel() > 0 else float("nan"),
        }

    def _compute_loss(
        self,
        embeddings: torch.Tensor,
        triples: List[Triple],
        membership_targets: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes loss for both class memberships and relations.
        Preserves original logic: masking unknowns and dynamic positive-class weighting.
        """

        device = self.device
        non_reduced_criterion = nn.BCEWithLogitsLoss(reduction="none")

        # ---------------------------------------------------------------------------- #
        #                               CLASS PREDICTIONS                              #
        # ---------------------------------------------------------------------------- #

        # Batch prediction logits for all individuals: (Bt, C)
        cls_logits = self.mlps[0](embeddings)

        # Targets tensor
        targets_tensor_cls = torch.as_tensor(membership_targets, dtype=torch.float32, device=device)

        # Mask is 1.0 where label is NOT unknown (±1), 0.0 where label is unknown (0)
        mask = (targets_tensor_cls != 0).float()

        # Convert labels from {-1,0,1} to {0.0, 0.5, 1.0} for BCE
        cls_targets = (targets_tensor_cls + 1.0) / 2.0

        # Compute un-reduced loss PER SAMPLE
        unreduced_class_loss = non_reduced_criterion(cls_logits, cls_targets)

        # Apply mask to zero out losses for unspecified class memberships
        masked_class_loss = unreduced_class_loss * mask

        # Average loss over known labels only
        num_known_labels = mask.sum()
        if num_known_labels > 0:
            total_class_loss = masked_class_loss.sum() / num_known_labels
        else:
            total_class_loss = torch.zeros((), device=device)

        # ---------------------------------------------------------------------------- #
        #                             RELATION PREDICTIONS                             #
        # ---------------------------------------------------------------------------- #

        # Group triples by predicate index
        grouped = defaultdict(list)
        for tr in triples:
            grouped[tr.predicate.index].append(tr)

        total_relation_loss = torch.zeros((), device=device)
        total_rel_count = 0

        # For each predicate (negative AND positive), compute loss separately
        for pred_idx, triples_list in grouped.items():
            s_indices = [t.subject.index for t in triples_list]
            o_indices = [t.object.index for t in triples_list]

            # Targets are 1.0 for positive, 0.0 for negative
            # P(<s, ~R_i, o> | KB) = 1 - P(<s, R_i, o> | KB)
            rel_targets = [1.0 if t.positive else 0.0 for t in triples_list]

            s_emb = embeddings[s_indices, :]
            o_emb = embeddings[o_indices, :]

            # mlps[0] is for classes, so relation MLPs start from index 1
            rel_logits = self.mlps[pred_idx + 1](s_emb, o_emb)  # [Bt, 1]

            rel_targets = torch.as_tensor(rel_targets, dtype=torch.float32, device=device).unsqueeze(1)

            # --------------------------- DYNAMIC WEIGHTING -------------------------- #

            # Calculate number of positive and negative samples in this group
            num_pos = (rel_targets == 1.0).sum()
            num_neg = (rel_targets == 0.0).sum()

            all_weights = torch.ones_like(rel_targets, device=device)

            if num_pos > 0 and num_neg > 0:
                # The actual weight to apply to positive samples
                ratio = num_neg.float() / num_pos.float()
                all_weights[rel_targets == 1.0] = ratio

            # Compute weighted loss
            unreduced_loss = non_reduced_criterion(rel_logits, rel_targets)
            weighted_loss = unreduced_loss * all_weights
            triple_loss = weighted_loss.sum()

            total_relation_loss += triple_loss
            total_rel_count += len(triples_list)

        if total_rel_count > 0:
            total_relation_loss = total_relation_loss / total_rel_count
        else:
            total_relation_loss = torch.zeros((), device=device)

        # ---------------------------------------------------------------------------- #
        #                                  TOTAL LOSS                                  #
        # ---------------------------------------------------------------------------- #

        total_loss = total_class_loss + total_relation_loss
        return total_loss, total_class_loss, total_relation_loss
