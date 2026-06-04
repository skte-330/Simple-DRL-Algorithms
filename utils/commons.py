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


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.

    delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
    A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    """
    deltas = rewards + gamma * next_values * (1.0 - dones) - values
    advantages = torch.zeros_like(values)

    gae = torch.tensor(0.0, dtype=torch.float32, device=values.device)
    for t in reversed(range(rewards.shape[0])):
        gae = deltas[t] + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns