import wandb


def log_gradients(model, step):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            wandb.log({f"gradients/{name}_norm": grad_norm}, step=step)


def log_similarity_stats(similarity_matrix):
    wandb.log({
        "similarity/mean": similarity_matrix.mean().item(),
        "similarity/std": similarity_matrix.std().item(),
        "similarity/max": similarity_matrix.max().item(),
        "similarity/min": similarity_matrix.min().item(),
    })
