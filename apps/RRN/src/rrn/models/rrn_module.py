import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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
from .rrn_batched import ClassesMLP, RelationMLP
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
        self.mlps.append(
            ClassesMLP(
                embedding_size=embedding_size,
                classes_count=len(classes),
                hidden_size=cfg.model.hidden_size,
                num_hidden_layers=cfg.model.num_hidden_layers,
            )
        )

        # One MLP per relation type (positive or negative predicate)
        for _ in relations:
            # MLP^{R_i} = P(<s, R_i, o> | KB)
            self.mlps.append(
                RelationMLP(
                    embedding_size=embedding_size,
                    hidden_size=cfg.model.hidden_size,
                    num_hidden_layers=cfg.model.num_hidden_layers,
                )
            )

    def forward(self, grouped_triples: Dict[int, Dict[str, torch.Tensor]], memberships: List[List[int]]):
        """
        Forward pass through the RRN to generate embeddings.
        """
        return self.rrn(grouped_triples, memberships)

    def configure_optimizers(self):
        """
        Configure optimizer (Adam) and optional schedulers.
        """

        optimizer = instantiate(self.cfg.hyperparams.optimizer, params=self.parameters())
        logger.info(f"Using optimizer: {self.cfg.hyperparams.optimizer._target_}")

        scheduler_kwargs = {"mode": "min", "factor": 0.5, "patience": 15}
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, **scheduler_kwargs)
        except TypeError:
            # Older PyTorch versions do not support the `verbose` kwarg.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
                "frequency": 1,
            },
        }

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
        base_grouped = batch["base_grouped"]
        base_memberships = batch["base_memberships"]

        # ! CRUCIAL: We compute loss on ALL triples and ALL memberships
        # (both base facts and inferred)
        all_triples = batch["all_triples"]
        all_memberships = batch["all_memberships"]

        # 1. RRN Forward pass (Message Passing)
        #    Only uses base facts to generate embeddings
        embeddings = self(base_grouped, base_memberships)

        # 2. MLP Predictions & Loss Calculation
        total_loss, class_loss, relation_loss = self._compute_loss(
            embeddings=embeddings, triples=all_triples, membership_targets=all_memberships
        )

        # 3. Evaluate Metrics
        individuals = batch.get("individuals", None)
        class_metrics = self._evaluate_classes(embeddings, all_memberships, individuals)
        triple_metrics = self._evaluate_triples(embeddings, all_triples)

        # 4. Logging
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/class_loss", class_loss, on_step=True, on_epoch=True, batch_size=1)
        self.log("train/relation_loss", relation_loss, on_step=True, on_epoch=True, batch_size=1)

        for key, value in class_metrics.items():
            if not math.isnan(value):
                self.log(f"train/class_{key}", value, on_step=False, on_epoch=True, batch_size=1)

        for key, value in triple_metrics.items():
            if not math.isnan(value):
                self.log(f"train/triple_{key}", value, on_step=False, on_epoch=True, batch_size=1)

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Single validation step for a Knowledge Graph.
        """
        # Unpack Data
        base_grouped = batch["base_grouped"]
        base_memberships = batch["base_memberships"]

        # In validation, we typically want to measure performance on ALL known facts
        # Default to True to avoid NaNs if validation set has no inferred facts
        if self.cfg.get("test_base_facts", True):
            target_triples = batch["all_triples"]
            target_memberships = batch["all_memberships"]
        else:
            target_triples = batch["inferred_triples"]
            target_memberships = batch["inferred_memberships"]

        # 1. RRN Forward pass (using base facts)
        embeddings = self(base_grouped, base_memberships)

        # 2. Compute Loss (validation loss)
        val_loss, val_class_loss, val_rel_loss = self._compute_loss(
            embeddings=embeddings, triples=target_triples, membership_targets=target_memberships
        )

        # 3. Evaluate Metrics
        individuals = batch.get("individuals", None)
        class_metrics = self._evaluate_classes(embeddings, target_memberships, individuals)
        triple_metrics = self._evaluate_triples(embeddings, target_triples)

        # 4. Log everything
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("val/total_loss", val_loss, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/class_loss", val_class_loss, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/relation_loss", val_rel_loss, on_step=False, on_epoch=True, batch_size=1)

        for key, value in class_metrics.items():
            if not math.isnan(value):
                self.log(
                    f"val/class_{key}", value, on_step=False, on_epoch=True, prog_bar=(key == "acc_all"), batch_size=1
                )

        for key, value in triple_metrics.items():
            if not math.isnan(value):
                self.log(
                    f"val/triple_{key}", value, on_step=False, on_epoch=True, prog_bar=(key == "acc_all"), batch_size=1
                )

        return val_loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Single test step for a Knowledge Graph.
        Replicates logic from the original test.py.
        """
        # Unpack Data
        base_grouped = batch["base_grouped"]
        base_memberships = batch["base_memberships"]

        if self.cfg.get("test_base_facts", True):
            target_triples = batch["all_triples"]
            target_memberships = batch["all_memberships"]
        else:
            target_triples = batch["inferred_triples"]
            target_memberships = batch["inferred_memberships"]

        # 3. RRN Inference (Always using BASE facts)
        embeddings = self(base_grouped, base_memberships)

        # 4. Compute Loss (test loss)
        test_loss, test_class_loss, test_rel_loss = self._compute_loss(
            embeddings=embeddings, triples=target_triples, membership_targets=target_memberships
        )

        # 5. Evaluate Class Predictions
        individuals = batch.get("individuals", None)
        class_metrics = self._evaluate_classes(
            embeddings=embeddings, membership_labels=target_memberships, individuals=individuals
        )

        # 6. Evaluate Relation Predictions
        triple_metrics = self._evaluate_triples(embeddings=embeddings, triples=target_triples)

        # 7. Logging
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("test/total_loss", test_loss, on_step=False, on_epoch=True, batch_size=1)
        self.log("test/class_loss", test_class_loss, on_step=False, on_epoch=True, batch_size=1)
        self.log("test/relation_loss", test_rel_loss, on_step=False, on_epoch=True, batch_size=1)

        # Lightning automatically accumulates these over the epoch
        for key, value in class_metrics.items():
            if not math.isnan(value):
                self.log(f"test/class_{key}", value, on_step=False, on_epoch=True, batch_size=1)

        for key, value in triple_metrics.items():
            if not math.isnan(value):
                self.log(f"test/triple_{key}", value, on_step=False, on_epoch=True, batch_size=1)

    def _evaluate_classes(
        self,
        embeddings: torch.Tensor,
        membership_labels: List[List[int]],
        individuals: Optional[List[Any]] = None,
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

        # New: Hops Bucketing
        hops_hits = defaultdict(list)
        type_hits = defaultdict(list)

        if individuals:
            # We need to map (ind_idx, cls_idx) to hops
            # Iterate through individuals and their sparse memberships
            for i, ind in enumerate(individuals):
                # Ensure ind has classes attribute
                if not hasattr(ind, "classes"):
                    continue

                for mem in ind.classes:
                    cls_idx = mem.cls.index
                    # Check if this membership was evaluated (i.e. not masked out/unknown)
                    if cls_targets[i, cls_idx] != 0.5:
                        hops = int(mem.metadata.get("hops", 0)) if hasattr(mem, "metadata") else 0
                        bucket = min(hops, 3)
                        hit = scores[i, cls_idx].item()
                        hops_hits[bucket].append(hit)

                        fact_type = "base_fact"
                        if hasattr(mem, "metadata"):
                            fact_type = str(mem.metadata.get("type", "base_fact")).lower()
                        type_hits[fact_type].append(hit)

        # Create masks
        pos_mask = cls_targets == 1.0
        neg_mask = cls_targets == 0.0
        known_mask = cls_targets != 0.5

        # Get scores
        positive_scores = scores[pos_mask]
        negative_scores = scores[neg_mask]
        all_known_scores = scores[known_mask]

        # Binary metrics are computed on known labels only (exclude unknown = 0.5)
        known_probs = cls_probs[known_mask]
        known_targets = cls_targets[known_mask]
        binary_metrics = self._compute_binary_metrics(known_probs, known_targets)

        # Return dict, avoiding empty tensor errors
        hop_acc = {f"acc_hops_{k}": (sum(v) / len(v) if len(v) > 0 else float("nan")) for k, v in hops_hits.items()}
        # Keep a stable set of hop keys in logs so charts don't disappear when
        # a bucket has no support in a given validation pass.
        for k in range(4):
            hop_acc.setdefault(f"acc_hops_{k}", float("nan"))

        return {
            "acc_all": all_known_scores.mean().item() if all_known_scores.numel() > 0 else float("nan"),
            "acc_pos": positive_scores.mean().item() if positive_scores.numel() > 0 else float("nan"),
            "acc_neg": negative_scores.mean().item() if negative_scores.numel() > 0 else float("nan"),
            **binary_metrics,
            **hop_acc,
            **{f"acc_type_{k}": (sum(v) / len(v) if len(v) > 0 else float("nan")) for k, v in type_hits.items()},
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

        # for buckets
        hops_hits = defaultdict(list)
        type_hits = defaultdict(list)

        # Group triples by predicate
        grouped_triples = defaultdict(list)
        for t in triples:
            grouped_triples[t.predicate.index].append(t)

        all_hits = []
        pos_hits = []
        neg_hits = []
        all_probs = []
        all_targets = []

        for pred_idx, group in grouped_triples.items():
            if pred_idx == -1:
                # Skip class membership triples (handled separately)
                continue

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
            probs = torch.sigmoid(logits).squeeze(1)
            targets_1d = targets.squeeze(1)
            predictions = probs.round()

            # Compare matches
            hits = (predictions == targets_1d).float()

            # Separate hits
            is_positive = targets_1d == 1.0
            is_negative = targets_1d == 0.0

            all_hits.append(hits)
            if is_positive.any():
                pos_hits.append(hits[is_positive])
            if is_negative.any():
                neg_hits.append(hits[is_negative])
            all_probs.append(probs)
            all_targets.append(targets_1d)

            # Bucketize by hops
            for i, t in enumerate(group):
                # Extract hops from metadata (default 0 for explicitly stated/base facts)
                hops = int(t.metadata.get("hops", 0)) if hasattr(t, "metadata") else 0
                # Treat hops >= 3 as bucket 3
                bucket = min(hops, 3)
                hops_hits[bucket].append(hits[i].item())

                fact_type = "base_fact"
                if hasattr(t, "metadata"):
                    fact_type = str(t.metadata.get("type", "base_fact")).lower()
                type_hits[fact_type].append(hits[i].item())

        # Aggregate results
        all_hits_t = torch.cat(all_hits) if all_hits else torch.tensor([], device=device)
        pos_hits_t = torch.cat(pos_hits) if pos_hits else torch.tensor([], device=device)
        neg_hits_t = torch.cat(neg_hits) if neg_hits else torch.tensor([], device=device)
        probs_t = torch.cat(all_probs) if all_probs else torch.tensor([], device=device)
        targets_t = torch.cat(all_targets) if all_targets else torch.tensor([], device=device)
        binary_metrics = self._compute_binary_metrics(probs_t, targets_t)

        hop_acc = {f"acc_hops_{k}": (sum(v) / len(v) if len(v) > 0 else float("nan")) for k, v in hops_hits.items()}
        for k in range(4):
            hop_acc.setdefault(f"acc_hops_{k}", float("nan"))

        return {
            "acc_all": torch.cat(all_hits).mean().item() if all_hits else float("nan"),
            "acc_pos": torch.cat(pos_hits).mean().item() if pos_hits else float("nan"),
            "acc_neg": torch.cat(neg_hits).mean().item() if neg_hits else float("nan"),
            **binary_metrics,
            **hop_acc,
            **{f"acc_type_{k}": (sum(v) / len(v) if len(v) > 0 else float("nan")) for k, v in type_hits.items()},
        }

    def _compute_binary_metrics(self, probs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute binary classification metrics for probabilities and binary targets.

        Args:
            probs: Model probabilities in [0, 1].
            targets: Binary labels in {0.0, 1.0}.
        """
        if probs.numel() == 0 or targets.numel() == 0:
            return {
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "fpr": float("nan"),
                "auc_roc": float("nan"),
            }

        probs = probs.float()
        targets = targets.float()

        # Binary metrics are only meaningful if both classes are present.
        pos_count = (targets == 1.0).sum().item()
        neg_count = (targets == 0.0).sum().item()
        if pos_count == 0 or neg_count == 0:
            return {
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "fpr": float("nan"),
                "auc_roc": float("nan"),
            }

        preds = (probs >= 0.5).float()

        tp = ((preds == 1.0) & (targets == 1.0)).sum().float()
        fp = ((preds == 1.0) & (targets == 0.0)).sum().float()
        tn = ((preds == 0.0) & (targets == 0.0)).sum().float()
        fn = ((preds == 0.0) & (targets == 1.0)).sum().float()

        precision = tp / (tp + fp) if (tp + fp).item() > 0 else torch.tensor(float("nan"), device=probs.device)
        recall = tp / (tp + fn) if (tp + fn).item() > 0 else torch.tensor(float("nan"), device=probs.device)
        fpr = fp / (fp + tn) if (fp + tn).item() > 0 else torch.tensor(float("nan"), device=probs.device)

        if not torch.isnan(precision).item() and not torch.isnan(recall).item() and (precision + recall).item() > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = torch.tensor(float("nan"), device=probs.device)

        auc_roc = self._compute_auc_roc(probs, targets)

        return {
            "precision": precision.item() if not torch.isnan(precision) else float("nan"),
            "recall": recall.item() if not torch.isnan(recall) else float("nan"),
            "f1": f1.item() if not torch.isnan(f1) else float("nan"),
            "fpr": fpr.item() if not torch.isnan(fpr) else float("nan"),
            "auc_roc": auc_roc,
        }

    def _compute_auc_roc(self, probs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute ROC-AUC from probabilities without external dependencies.
        Returns NaN when only one class is present.
        """
        pos_count = (targets == 1.0).sum()
        neg_count = (targets == 0.0).sum()

        if pos_count.item() == 0 or neg_count.item() == 0:
            return float("nan")

        order = torch.argsort(probs, descending=True)
        sorted_targets = targets[order]

        tps = torch.cumsum((sorted_targets == 1.0).float(), dim=0)
        fps = torch.cumsum((sorted_targets == 0.0).float(), dim=0)

        tpr = tps / pos_count.float()
        fpr = fps / neg_count.float()

        # Anchor curve at origin for trapezoidal integration.
        tpr = torch.cat([torch.zeros(1, device=tpr.device), tpr])
        fpr = torch.cat([torch.zeros(1, device=fpr.device), fpr])

        return torch.trapz(tpr, fpr).item()

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

        # Create weights mask dynamically to counteract class imbalance
        num_pos = ((cls_targets == 1.0) & (mask == 1.0)).sum()
        num_neg = ((cls_targets == 0.0) & (mask == 1.0)).sum()

        all_weights = torch.ones_like(cls_targets, device=device)

        if num_pos > 0 and num_neg > 0:
            ratio = num_neg.float() / num_pos.float()
            ratio = torch.clamp(ratio, max=10.0)
            all_weights[(cls_targets == 1.0) & (mask == 1.0)] = ratio

        # Compute un-reduced loss PER SAMPLE
        unreduced_class_loss = non_reduced_criterion(cls_logits, cls_targets)

        # Apply weights and mask
        masked_class_loss = unreduced_class_loss * mask * all_weights

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
            if pred_idx == -1:
                # Skip class membership triples (handled separately)
                continue

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
                # Clamp ratio to prevent extreme gradient explosions on imbalanced graphs
                ratio = torch.clamp(ratio, max=10.0)
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

        total_loss = (0.2 * total_class_loss) + total_relation_loss
        return total_loss, total_class_loss, total_relation_loss
