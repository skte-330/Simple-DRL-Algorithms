import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from tqdm import tqdm

from utils.env import make_atari_env
from utils.buffers import AtariReplayBuffer
from utils.commons import linear_schedule, set_seed

@dataclass
class AtariDQNConfig:
    # Basic settings
    env_id: str = "ALE/Pong-v5"
    gamma: float = 0.99
    buffer_capacity: int = 100000
    batch_size: int = 64
    target_update_frequency: int = 500      # in gradient steps
    train_every: int = 4                    # in env steps
    max_episodes: int = 1000
    max_env_steps: int | None = None

    # Atari preprocessing
    clip_reward: bool = True
    repeat_action_probability: float = 0.0

    # Training config
    lr: float = 1e-4
    grad_clip_norm: float | None = 10.0

    # Exploration (epsilon-greedy)
    eps_start: float = 1.0
    eps_end: float = 0.05
    final_decay_steps: int = 200000
    warmup_steps: int = 50000

    # Improvements
    double_dqn: bool = False
    dueling_dqn: bool = False

    # Repro & device
    seed: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AtariQNetwork(nn.Module):
    """
    Nature DQN-style CNN for Atari:
    input:  (B, 4, 84, 84)
    output: (B, num_actions)
    """
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingAtariQNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.common = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(512, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        common = self.common(features)
        value = self.value(common)
        advantage = self.advantage(common)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# -----------------------------
# DQN Agent
# -----------------------------
class AtariDQNAgent:
    def __init__(self, cfg: AtariDQNConfig):
        set_seed(cfg.seed)

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.env = make_atari_env(cfg.env_id, cfg.seed, cfg.repeat_action_probability)

        obs_shape = self.env.observation_space.shape
        if len(obs_shape) != 3:
            raise ValueError(f"Expected stacked Atari obs shape (C, H, W), got {obs_shape}.")

        in_channels = int(obs_shape[0])
        num_actions = int(self.env.action_space.n)

        q_net_cls = DuelingAtariQNetwork if cfg.dueling_dqn else AtariQNetwork
        self.q_net = q_net_cls(in_channels, num_actions).to(self.device)
        self.target_q_net = q_net_cls(in_channels, num_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay_buffer = AtariReplayBuffer(cfg.buffer_capacity, self.device)

        self.global_step = 0
        self.gradient_step = 0
        self.epsilon = cfg.eps_start

    @torch.no_grad()
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.epsilon = linear_schedule(self.global_step, self.cfg.eps_start, self.cfg.eps_end, self.cfg.warmup_steps, self.cfg.final_decay_steps)

            if self.global_step < self.cfg.warmup_steps or np.random.random() < self.epsilon:
                return int(self.env.action_space.sample())

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        state_tensor = state_tensor.unsqueeze(0) / 255.0
        q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self) -> dict[str, float]:
        cfg = self.cfg
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(cfg.batch_size)

        # Get current Q values
        q_values = self.q_net(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            if cfg.double_dqn:
                # select action by main net, evalute aciton by target net
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_q_net(next_states).gather(1, next_actions)
            else:
                next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True).values

            target_q_values = rewards + cfg.gamma * next_q_values * (1.0 - dones)

        # Compute loss and update
        loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), cfg.grad_clip_norm)
        self.optimizer.step()

        # Update target network periodically
        self.gradient_step += 1
        if self.gradient_step % cfg.target_update_frequency == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss": float(loss.item()),
            "q_value": float(q_values.mean().item()),
        }

    def train(self) -> list[dict[str, float]]:
        cfg = self.cfg
        metrics: list[dict[str, float]] = []
        last_loss: float | None = None
        last_q_value: float | None = None

        # Running episodes of cfg.max_episodes
        iterator = tqdm(range(cfg.max_episodes), desc="Atari DQN")
        stop_training = False
        for episode in iterator:
            reset_seed = None if cfg.seed is None else cfg.seed + episode
            state, _ = self.env.reset(seed=reset_seed)

            episode_return = 0.0
            episode_length = 0
            done = False

            # run until one episode ends core algorithm logics
            while not done:
                action = self.select_action(state, explore=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # add data to replay buffer
                clipped_reward = float(np.sign(reward)) if cfg.clip_reward else float(reward)
                self.replay_buffer.append(state, action, clipped_reward, next_state, float(done))

                episode_return += float(reward)
                episode_length += 1
                self.global_step += 1
                state = next_state

                should_update = (
                    len(self.replay_buffer) >= max(cfg.warmup_steps, cfg.batch_size)
                    and self.global_step % cfg.train_every == 0
                )
                if should_update:
                    update_info = self.update()
                    last_loss = update_info["loss"]
                    last_q_value = update_info["q_value"]

                if cfg.max_env_steps is not None and self.global_step >= cfg.max_env_steps:
                    stop_training = True
                    break

            # logging metrics
            recent_returns = [m["episode_return"] for m in metrics[-19:]] + [episode_return]
            mean_return = float(np.mean(recent_returns))

            row = {
                "episode": float(episode + 1),
                "global_step": float(self.global_step),
                "episode_return": float(episode_return),
                "episode_length": float(episode_length),
                "mean_return_20": mean_return,
                "epsilon": float(self.epsilon),
                "loss": 0.0 if last_loss is None else float(last_loss),
                "q_value": 0.0 if last_q_value is None else float(last_q_value),
            }
            metrics.append(row)

            iterator.set_postfix(
                {
                    "return": f"{episode_return:.1f}",
                    "mean20": f"{mean_return:.1f}",
                    "eps": f"{self.epsilon:.3f}",
                    "step": self.global_step,
                }
            )

            if stop_training:
                break

        return metrics

    def save(self, path: str | Path) -> None:
        print(f"Saving model to {path}.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.cfg.to_dict(),
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "gradient_step": self.gradient_step,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> None:
        print(f"Loading model from {path}.")
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_q_net.load_state_dict(checkpoint["target_q_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.gradient_step = int(checkpoint.get("gradient_step", 0))
        self.epsilon = float(checkpoint.get("epsilon", self.cfg.eps_end))
