import torch

def _compute_pr_auc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    pos_count = (targets == 1.0).sum()
    neg_count = (targets == 0.0).sum()

    if pos_count.item() == 0 or neg_count.item() == 0:
        return float("nan")

    order = torch.argsort(probs, descending=True)
    sorted_targets = targets[order]

    tps = torch.cumsum((sorted_targets == 1.0).float(), dim=0)
    fps = torch.cumsum((sorted_targets == 0.0).float(), dim=0)

    precision = tps / (tps + fps + 1e-12)
    recall = tps / pos_count.float()

    # Anchor curve at (recall=0, precision=1) for stable integration.
    precision = torch.cat([torch.ones(1, device=precision.device), precision])
    recall = torch.cat([torch.zeros(1, device=recall.device), recall])

    return torch.trapz(precision, recall).item()

probs = torch.tensor([0.9, 0.8, 0.7, 0.6])
targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
print("PR AUC 1:", _compute_pr_auc(probs, targets))

probs_nan = torch.tensor([float('nan'), float('nan'), 0.7, 0.6])
targets_nan = torch.tensor([1.0, 0.0, 1.0, 0.0])
print("PR AUC NaNs:", _compute_pr_auc(probs_nan, targets_nan))
