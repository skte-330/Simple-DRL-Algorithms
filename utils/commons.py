import random
import numpy as np
import torch

def linear_schedule(
    cur_step: int,
    start_eps: float,
    end_eps: float,
    warmup_steps: int,
    final_decay_steps: int,
) -> float:
    """
    epsilon = start_eps       if cur_step <= warmup_steps
            = linear decay    if warmup_steps < cur_step < final_decay_steps
            = end_eps         if cur_step >= final_decay_steps
    """
    ratio = min(max((cur_step - warmup_steps) / (final_decay_steps - warmup_steps), 0.0), 1.0)
    return start_eps + (end_eps - start_eps) * ratio


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)