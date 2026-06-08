import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from tqdm import tqdm

from utils.commons import set_seed
from utils.buffers import DDPGReplayBuffer
from utils.commons import soft_update

@dataclass
class DDPGConfig:
    # Basic settings
    env_id: str = "Pendulum-v1"
    gamma: float = 0.99
    buffer_capacity: int = 100000
    batch_size: int = 64
    train_every: int = 1                 # gradient step frequency (in env steps)
    max_episodes: int = 200

    # Training config
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    grad_clip_norm: float | None = 10.0
    hidden_dim: int = 128
    tau: float = 0.005                   # target network soft-update coefficient

    # Exploration
    noise_scale: float = 0.1             # std of Gaussian action noise
    warmup_steps: int = 1000             # random actions before gradient updates

    # Repro & device
    seed: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActorNetwork(nn.Module):
    """
    Deterministic policy network.
    Outputs continuous actions scaled to the environment action range.
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int = 128,
    ):
        super().__init__()

        action_low = torch.as_tensor(action_low, dtype=torch.float32).view(1, act_dim)
        action_high = torch.as_tensor(action_high, dtype=torch.float32).view(1, act_dim)
        action_scale = (action_high - action_low) / 2.0
        action_mid = (action_high + action_low) / 2.0

        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_mid", action_mid)

        self.l1 = nn.Linear(obs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x)) * self.action_scale + self.action_mid


class CriticNetwork(nn.Module):
    """
    Q network for continuous control.
    Inputs state and action, outputs Q(s, a).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.l1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


def make_env(env_id: str, seed: int | None = None):
    print(f"Creating environment {env_id}.")
    env = gym.make(env_id)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


# -----------------------------
# DDPG Agent
# -----------------------------
class DDPGAgent:
    def __init__(self, cfg: DDPGConfig):
        set_seed(cfg.seed)

        env = make_env(cfg.env_id, cfg.seed)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("DDPG only supports continuous Box action spaces.")
        if not np.all(np.isfinite(env.action_space.low)) or not np.all(np.isfinite(env.action_space.high)):
            raise ValueError("DDPG requires finite action bounds.")

        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.action_shape = env.action_space.shape
        self.action_low = env.action_space.low.astype(np.float32).reshape(-1)
        self.action_high = env.action_space.high.astype(np.float32).reshape(-1)

        # Neural networks for actor and critic approximation
        self.actor = ActorNetwork(obs_dim, act_dim, self.action_low, self.action_high, cfg.hidden_dim).to(self.device)
        self.target_actor = ActorNetwork(obs_dim, act_dim, self.action_low, self.action_high, cfg.hidden_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.requires_grad_(False)

        self.critic = CriticNetwork(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.target_critic = CriticNetwork(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.replay_buffer = DDPGReplayBuffer(cfg.buffer_capacity, self.device)

        self.global_step = 0
        self.gradient_step = 0

    @torch.no_grad()
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        # During warmup, random actions improve initial replay-buffer coverage.
        if explore and self.global_step < self.cfg.warmup_steps:
            return np.asarray(self.env.action_space.sample(), dtype=np.float32)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        action = self.actor(state_tensor).cpu().numpy().reshape(-1)

        if explore:
            noise = np.random.normal(0.0, self.cfg.noise_scale, size=action.shape)
            action = action + noise

        action = np.clip(action, self.action_low, self.action_high)
        return action.astype(np.float32).reshape(self.action_shape)

    def update(self) -> dict[str, float]:
        cfg = self.cfg
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(cfg.batch_size)

        # Update critic
        q_values = self.critic(states, actions)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            target_q_values = rewards + cfg.gamma * next_q_values * (1.0 - dones)

        critic_loss = F.mse_loss(q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        self.critic_optimizer.step()

        # Update actor
        for p in self.critic.parameters():
            p.requires_grad_(False)

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad_(True)

        # Soft update target networks
        soft_update(self.target_actor, self.actor, cfg.tau)
        soft_update(self.target_critic, self.critic, cfg.tau)

        self.gradient_step += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_value": float(q_values.mean().item()),
        }

    def train(self) -> list[dict[str, float]]:
        cfg = self.cfg
        metrics: list[dict[str, float]] = []

        last_actor_loss: float | None = None
        last_critic_loss: float | None = None
        last_q_value: float | None = None

        iterator = tqdm(range(cfg.max_episodes), desc="DDPG")
        for episode in iterator:
            reset_seed = None if cfg.seed is None else cfg.seed + episode
            state, _ = self.env.reset(seed=reset_seed)

            episode_return = 0.0
            episode_length = 0
            done = False

            while not done:
                action = self.select_action(state, explore=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.append(state, action, reward, next_state, float(done))

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
                    last_actor_loss = update_info["actor_loss"]
                    last_critic_loss = update_info["critic_loss"]
                    last_q_value = update_info["q_value"]

            recent_returns = [m["episode_return"] for m in metrics[-19:]] + [episode_return]
            mean_return = float(np.mean(recent_returns))

            row = {
                "episode": float(episode + 1),
                "global_step": float(self.global_step),
                "episode_return": float(episode_return),
                "episode_length": float(episode_length),
                "mean_return_20": mean_return,
                "actor_loss": 0.0 if last_actor_loss is None else float(last_actor_loss),
                "critic_loss": 0.0 if last_critic_loss is None else float(last_critic_loss),
                "q_value": 0.0 if last_q_value is None else float(last_q_value),
            }
            metrics.append(row)

            iterator.set_postfix(
                {
                    "return": f"{episode_return:.1f}",
                    "mean20": f"{mean_return:.1f}",
                    "step": self.global_step,
                }
            )

        return metrics

    def save(self, path: str | Path) -> None:
        print(f"Saving model to {path}.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.cfg.to_dict(),
                "actor": self.actor.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "global_step": self.global_step,
                "gradient_step": self.gradient_step,
            },
            path,
        )

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> None:
        print(f"Loading model from {path}.")
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.gradient_step = int(checkpoint.get("gradient_step", 0))
