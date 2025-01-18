import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_flat_steps: int, num_training_steps: int, num_cycles: int, min_lr_ratio: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    
    if current_step < (num_warmup_steps + num_flat_steps):
        return 1.0
    
    progress = float(current_step - num_warmup_steps - num_flat_steps) / float(max(1, num_training_steps - num_warmup_steps - num_flat_steps))
    if progress >= 1.0:
        return min_lr_ratio
    
    cosine_decay = max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    num_flat_steps: int = 0,
    num_cycles: int = 1, 
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that:
    Increases linearly from 0 to the initial lr set in the optimizer during warmup period.
    Stays constant at the initial lr for a specified number of steps (flat top period).
    Decreases following the values of the cosine function between the initial lr and min_lr with several hard restarts.
    
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_flat_steps (`int`, *optional*, defaults to 0):
            The number of steps to maintain peak learning rate before decay.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        min_lr_ratio (`float`, *optional*, defaults to 0.0):
            The minimum learning rate ratio compared to the initial lr.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_flat_steps=num_flat_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
