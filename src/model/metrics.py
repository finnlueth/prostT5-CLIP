import torch
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class BatchMetricsAccumulator:
    """Accumulates metrics across batches for final computation."""

    total_similarity: float = 0.0
    total_samples: int = 0
    all_similarities: list = field(default_factory=list)
    
    def __str__(self):
        return f"BatchMetricsAccumulator(total_similarity={self.total_similarity}, total_samples={self.total_samples}, all_similarities={self.all_similarities})"


def compute_metrics(
    eval_pred, compute_result: bool = False, metrics_accumulator: Optional[BatchMetricsAccumulator] = None
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
    print("compute_metrics")
    if metrics_accumulator is None:
        metrics_accumulator = BatchMetricsAccumulator()

    if compute_result:
        # Compute final metrics from accumulated statistics
        if metrics_accumulator.total_samples == 0:
            return {
                "mean_cosine_similarity": 0.0,
                "std_cosine_similarity": 0.0,
                "min_cosine_similarity": 0.0,
                "max_cosine_similarity": 0.0,
            }

        mean_similarity = metrics_accumulator.total_similarity / metrics_accumulator.total_samples
        all_sims = np.array(metrics_accumulator.all_similarities)

        return {
            "mean_cosine_similarity": mean_similarity,
            "std_cosine_similarity": np.std(all_sims),
            "min_cosine_similarity": np.min(all_sims),
            "max_cosine_similarity": np.max(all_sims),
        }

    # Extract embeddings from the predictions
    # The model output contains proj_protein_embeds and proj_text_embeds
    print((type(eval_pred)))
    print("----")
    print(len(eval_pred.predictions))
    print(type(eval_pred.predictions[0].shape))
    
    protein_embeds, text_embeds = eval_pred.predictions

    # Convert to torch tensors if they're numpy arrays
    if isinstance(protein_embeds, np.ndarray):
        protein_embeds = torch.from_numpy(protein_embeds)
    if isinstance(text_embeds, np.ndarray):
        text_embeds = torch.from_numpy(text_embeds)

    # Mean pool across sequence length if needed
    if len(protein_embeds.shape) > 2:
        protein_embeds = torch.mean(protein_embeds, dim=1)
    if len(text_embeds.shape) > 2:
        text_embeds = torch.mean(text_embeds, dim=1)

    # Normalize embeddings
    protein_embeds = torch.nn.functional.normalize(protein_embeds, p=2, dim=-1)
    text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)

    # Calculate cosine similarity
    similarities = torch.sum(protein_embeds * text_embeds, dim=-1)

    # Convert to numpy for accumulation
    similarities_np = similarities.cpu().numpy()

    # Accumulate batch statistics
    batch_mean = float(np.mean(similarities_np))
    metrics_accumulator.total_similarity += batch_mean * len(similarities_np)
    metrics_accumulator.total_samples += len(similarities_np)
    metrics_accumulator.all_similarities.extend(similarities_np.tolist())

    # Return batch-level metrics
    return {
        "batch_mean_cosine_similarity": batch_mean,
        "batch_std_cosine_similarity": float(np.std(similarities_np)),
        "batch_min_cosine_similarity": float(np.min(similarities_np)),
        "batch_max_cosine_similarity": float(np.max(similarities_np)),
    }



def metrics_factory():
    metrics_accumulator = BatchMetricsAccumulator()
    
    # print("metrics_factory")
    # print(metrics_accumulator)

    def metrics_wrapper(eval_pred, compute_result):
        return compute_metrics(
            eval_pred,
            compute_result=compute_result,
            metrics_accumulator=metrics_accumulator,
            )

    return metrics_wrapper
