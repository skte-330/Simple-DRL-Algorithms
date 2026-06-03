import torch
import numpy as np
import random
from collections import deque
from typing import Tuple

class ReplayBuffer:
    """A simple replay buffer for off-policy algorithms."""
    def __init__(self, capacity: int, device: str | torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device(device)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        # making the first dim be the batch dim
        states = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device).view(batch_size, -1) 
        next_states = torch.as_tensor(np.asarray(next_states, dtype=np.float32), device=self.device).view(batch_size, -1)

        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).view(batch_size, 1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(batch_size, 1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(batch_size, 1)

        return states, actions, rewards, next_states, dones

class AtariReplayBuffer:
    """
    Replay buffer for Atari observations.

    Observations are stored as uint8 with shape (frame_stack, 84, 84).
    They are converted to float32 and normalized to [0, 1] during sampling.
    """

    def __init__(self, capacity: int, device: str | torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device(device)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: float) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.uint8),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.uint8),
                float(done),
            )
        )

    def sample(self, batch_size: int):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.as_tensor(np.stack(states, axis=0), dtype=torch.float32, device=self.device).div_(255.0)
        next_states = torch.as_tensor(np.stack(next_states, axis=0), dtype=torch.float32, device=self.device).div_(255.0)

        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).view(-1, 1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)

        return states, actions, rewards, next_states, dones
    

class DDPGReplayBuffer:
    """
    Replay buffer for continuous actions.
    """
    def __init__(self, capacity: int, device: str | torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device(device)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device).view(batch_size, -1)
        next_states = torch.as_tensor(np.asarray(next_states, dtype=np.float32), device=self.device).view(batch_size, -1)
        actions = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=self.device).view(batch_size, -1)

        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(batch_size, 1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(batch_size, 1)

        return states, actions, rewards, next_states, dones