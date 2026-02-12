
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

class LogGraphArtifacts(pl.Callback):
    """
    Callback to log a sample Knowledge Graph from the validation set to WandB.
    Logs base facts, inferred facts, and class memberships as a table.
    """
    def __init__(self, log_every_n_epochs: int = 5, num_samples: int = 1):
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # Check for WandB logger
        if not isinstance(trainer.logger, WandbLogger):
            return

        # Get validation dataloader from datamodule
        # Note: trainer.datamodule might not be available if not used, but RRN uses it.
        if not trainer.datamodule:
            logger.warning("LogGraphArtifacts: No DataModule found in trainer.")
            return

        val_loader = trainer.datamodule.val_dataloader()
        if not val_loader:
             return

        logger.info(f"Logging {self.num_samples} validation sample(s) to WandB...")
        
        # Iterate and log
        columns = ["Epoch", "Sample ID", "Subject", "Predicate", "Object", "Fact Type", "Hops", "Label", "Score", "Prediction", "Correct"]
        data = []

        # We need to manually iterate since dataloader is likely infinite or complex
        # Just grab the first N batches
        for i, batch in enumerate(val_loader):
            if i >= self.num_samples:
                break
            
            # Extract data
            individuals = batch.get("individuals", [])
            base_triples = batch.get("base_triples", [])
            base_memberships = batch.get("base_memberships", [])
            all_triples = batch.get("all_triples", [])
            
            # Move to device
            device = pl_module.device
            # Note: Triples/Individuals are lists of objects, not tensors, so no .to(device) needed for them
            # But we need to ensure the RRN runs on the right device w.r.t embeddings
            
            # 1. Run RRN to get embeddings
            # We use no_grad to avoid tracking gradients
            with torch.no_grad():
                embeddings = pl_module(base_triples, base_memberships)
            
            # We assume batch size 1 (one graph)
            sample_id = f"val_sample_{i}"
            
            # 1. Log Triples
            for t in all_triples:
                fact_type = "Base Fact" if t.is_base_fact else "Inferred"
                hops = t.get_hops()
                # Determine type string from metadata if available (for precise negative types)
                if hasattr(t, "metadata"):
                    meta_type = t.metadata.get("type", None)
                    if meta_type: fact_type = meta_type
                
                # Compute Score
                # Identify predicate index
                pred_idx = t.predicate.index
                
                # Get embeddings for s and o
                s_idx = t.subject.index
                o_idx = t.object.index
                
                s_emb = embeddings[s_idx].unsqueeze(0)
                o_emb = embeddings[o_idx].unsqueeze(0)
                
                # Run relation MLP
                # mlps[0] is Class MLP, so relation MLPs start at 1
                with torch.no_grad():
                    logits = pl_module.mlps[pred_idx + 1](s_emb, o_emb)
                    prob = torch.sigmoid(logits).item()
                
                prediction = 1 if prob >= 0.5 else 0
                label = 1 if t.positive else 0
                is_correct = (prediction == label)

                row = [
                    trainer.current_epoch,
                    sample_id,
                    t.subject.name,
                    t.predicate.name,
                    t.object.name,
                    fact_type,
                    hops,
                    label,
                    f"{prob:.4f}",
                    prediction,
                    is_correct
                ]
                data.append(row)
                
            # 2. Log Memberships
            # Logic: iterate individuals and their linked memberships for sparse access
            
            if individuals:
                for ind in individuals:
                    if hasattr(ind, "classes"):
                        # Pre-compute ind embedding
                        ind_idx = ind.index
                        ind_emb = embeddings[ind_idx].unsqueeze(0)
                        
                        # Run Class MLP once for this individual
                        with torch.no_grad():
                            cls_logits = pl_module.mlps[0](ind_emb)
                            cls_probs = torch.sigmoid(cls_logits).squeeze(0) # [Num_Classes]
                        
                        for m in ind.classes:
                             fact_type = "Base Fact" if m.is_base_fact else "Inferred"
                             hops = m.get_hops()
                             if hasattr(m, "metadata"):
                                meta_type = m.metadata.get("type", None)
                                if meta_type: fact_type = meta_type
                             
                             # Get score from pre-computed probs
                             cls_idx = m.cls.index
                             prob = cls_probs[cls_idx].item()
                             
                             prediction = 1 if prob >= 0.5 else 0
                             label = 1 if m.is_member else 0
                             is_correct = (prediction == label)

                             row = [
                                trainer.current_epoch,
                                sample_id,
                                m.individual.name,
                                "rdf:type",
                                m.cls.name,
                                fact_type,
                                hops,
                                label,
                                f"{prob:.4f}",
                                prediction,
                                is_correct
                            ]
                             data.append(row)

        if data:
            # Log as Table
            table = wandb.Table(columns=columns, data=data)
            trainer.logger.experiment.log({"val/sample_facts": table})
