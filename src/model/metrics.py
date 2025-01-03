import torch
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass(slots=True)
class BatchMetricsAccumulator:
    """Accumulates metrics across batches for final computation."""

    similarities: list = field(default_factory=list)

    def reset(self):
        self.similarities = []

    def accumulate(self):
        metrics = {
            "mean_cosine_similarity": np.mean(self.similarities),
            "std_cosine_similarity": np.std(self.similarities),
            "min_cosine_similarity": np.min(self.similarities),
            "max_cosine_similarity": np.max(self.similarities),
        }
        return metrics

    def __str__(self):
        return f"BatchMetricsAccumulator({', '.join(f'{attr}={getattr(self, attr)}' for attr in vars(self))})"


def compute_metrics(
    eval_pred, compute_result: bool = False, metrics_accumulator: BatchMetricsAccumulator = None
) -> Dict[str, float]:
    """
    Compute cosine similarity metrics between protein and text embeddings.

    Args:
        eval_pred: EvalPrediction object containing predictions and labels
        compute_result: If True, compute final metrics from accumulated batch statistics
        metrics_accumulator: Accumulator for batch-level statistics

    Returns:
        Dictionary containing computed metrics
    """
    predictions, labels = eval_pred
    proj_protein_embeds, proj_text_embeds = predictions

    protein_embeddings = proj_protein_embeds
    text_embeddings = proj_text_embeds

    if isinstance(protein_embeddings, np.ndarray):
        protein_embeddings = torch.from_numpy(protein_embeddings)
    if isinstance(text_embeddings, np.ndarray):
        text_embeddings = torch.from_numpy(text_embeddings)

    if len(proj_protein_embeds.shape) > 2:
        protein_embeddings = torch.mean(protein_embeddings, dim=1)
    if len(text_embeddings.shape) > 2:
        text_embeddings = torch.mean(text_embeddings, dim=1)

    protein_embeddings = torch.nn.functional.normalize(protein_embeddings, p=2, dim=-1)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)

    similarities = torch.sum(protein_embeddings * text_embeddings, dim=-1)

    similarities_np = similarities.cpu().numpy()
    # similarities_np = np.abs(similarities_np)
    metrics_accumulator.similarities.extend(similarities_np.tolist())

    results = metrics_accumulator.accumulate()
    if compute_result:
        metrics_accumulator.reset()

    return results


def metrics_factory():
    metrics_accumulator = BatchMetricsAccumulator()

    def metrics_wrapper(eval_preds, compute_result):
        return compute_metrics(
            eval_preds,
            compute_result=compute_result,
            metrics_accumulator=metrics_accumulator,
        )

    return metrics_wrapper
