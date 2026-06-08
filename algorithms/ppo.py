import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from utils.commons import set_seed, compute_gae
from utils.env import make_env

@dataclass
class PPOConfig:
    # Basic settings
    env_id: str = "CartPole-v1"

    # Reinforcement learning parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO hyperparameters
    clip: float = 0.2
    entropy_coef: float = 0.01
    normalize_advantages: bool = True
    target_kl: float | None = None

    # Network training & optimization
    hidden_dim: int = 128
    actor_lr: float = 2.5e-4
    critic_lr: float = 2.5e-4
    max_grad_norm: float = 1.0

    # Training loop
    batch_size: int = 2048          # rollout length before each PPO update
    total_epochs: int = 100         # number of PPO updates
    update_epochs: int = 4          # optimization epochs per rollout
    num_minibatches: int = 4

    # Repro & device
    seed: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActorNetwork(nn.Module):
    """
    Actor network for discrete action spaces.
    Outputs logits instead of probabilities for numerical stability.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class CriticNetwork(nn.Module):
    """State-value Network V(s)."""
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        set_seed(cfg.seed)

        env = make_env(cfg.env_id, cfg.seed)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("PPOAgent in this file only supports discrete action spaces.")

        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(env.action_space.n)

        self.actor = ActorNetwork(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim, cfg.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, eps=1e-5)

        self.global_step = 0
        self.episode = 0

    def _as_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)

    @torch.no_grad()
    def select_action(self, state_tensor: torch.Tensor, explore: bool = True) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: int
            log_prob: tensor scalar
        """
        logits = self.actor(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)

        if explore:
            action_tensor = dist.sample()
        else:
            action_tensor = torch.argmax(logits, dim=-1)

        log_prob = dist.log_prob(action_tensor)
        return int(action_tensor.item()), log_prob.squeeze(0)

    def update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg

        states = rollout["states"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        advantages = rollout["advantages"]
        returns = rollout["returns"]

        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = states.shape[0]
        minibatch_size = max(1, batch_size // cfg.num_minibatches)

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []

        for _ in range(cfg.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits = self.actor(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                unclipped_obj = ratio * mb_advantages
                clipped_obj = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * mb_advantages
                actor_loss = -torch.min(unclipped_obj, clipped_obj).mean() - cfg.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()

                values = self.critic(mb_states).squeeze(-1)
                critic_loss = F.mse_loss(values, mb_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip).float().mean()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.item()))
                approx_kls.append(float(approx_kl.item()))
                clip_fracs.append(float(clip_frac.item()))

            if cfg.target_kl is not None and approx_kls and approx_kls[-1] > cfg.target_kl:
                break

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else float("nan"),
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else float("nan"),
            "entropy": float(np.mean(entropies)) if entropies else float("nan"),
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else float("nan"),
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else float("nan"),
        }

    @staticmethod
    def compute_explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        y_var = torch.var(y_true)
        diff_var = torch.var(y_true - y_pred)
        return float((1.0 - diff_var / (y_var + 1e-8)).item())

    def train(self) -> list[dict[str, float]]:
        cfg = self.cfg
        metrics: list[dict[str, float]] = []

        obs_shape = self.env.observation_space.shape
        obs_dim = int(np.prod(obs_shape))

        states = torch.zeros((cfg.batch_size, obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.zeros((cfg.batch_size,), dtype=torch.long, device=self.device)
        log_probs = torch.zeros((cfg.batch_size,), dtype=torch.float32, device=self.device)
        rewards = torch.zeros((cfg.batch_size,), dtype=torch.float32, device=self.device)
        dones = torch.zeros((cfg.batch_size,), dtype=torch.float32, device=self.device)
        values = torch.zeros((cfg.batch_size,), dtype=torch.float32, device=self.device)

        reset_seed = cfg.seed
        state, _ = self.env.reset(seed=reset_seed)

        episode_return = 0.0
        episode_length = 0
        completed_returns: list[float] = []
        completed_lengths: list[int] = []

        iterator = tqdm(range(cfg.total_epochs), desc="PPO")
        for epoch in iterator:
            for step in range(cfg.batch_size):
                state_tensor = self._as_state_tensor(state)
                states[step] = state_tensor.squeeze(0)

                with torch.no_grad():
                    action, log_prob = self.select_action(state_tensor, explore=True)
                    value = self.critic(state_tensor).squeeze(-1)
    
                actions[step] = action
                log_probs[step] = log_prob
                values[step] = value

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                rewards[step] = float(reward)
                dones[step] = float(done)

                self.global_step += 1
                episode_return += float(reward)
                episode_length += 1
                state = next_state

                if done:
                    completed_returns.append(float(episode_return))
                    completed_lengths.append(int(episode_length))
                    self.episode += 1

                    episode_return = 0.0
                    episode_length = 0

                    reset_seed = None if cfg.seed is None else cfg.seed + self.episode
                    state, _ = self.env.reset(seed=reset_seed)

            with torch.no_grad():
                next_value = self.critic(self._as_state_tensor(state)).squeeze(-1)
            next_values = torch.cat([values[1:], next_value])

            advantages, returns = compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                next_values=next_values,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            rollout = {
                "states": states,
                "actions": actions,
                "log_probs": log_probs,
                "advantages": advantages.detach(),
                "returns": returns.detach(),
            }

            update_info = self.update(rollout)

            with torch.no_grad():
                value_pred = self.critic(states).squeeze(-1)
                explained_variance = self.compute_explained_variance(value_pred, returns)

            mean_return_20 = float(np.mean(completed_returns[-20:])) if completed_returns else float("nan")
            last_return = float(completed_returns[-1]) if completed_returns else float("nan")
            last_length = float(completed_lengths[-1]) if completed_lengths else float("nan")

            row = {
                "epoch": float(epoch + 1),
                "episode": float(self.episode),
                "global_step": float(self.global_step),
                "last_episode_return": last_return,
                "last_episode_length": last_length,
                "mean_return_20": mean_return_20,
                "explained_variance": float(explained_variance),
                **update_info,
            }
            metrics.append(row)

            iterator.set_postfix(
                {
                    "mean20": f"{mean_return_20:.1f}",
                    "episodes": self.episode,
                    "step": self.global_step,
                    "kl": f"{update_info['approx_kl']:.4f}",
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
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "global_step": self.global_step,
                "episode": self.episode,
            },
            path,
        )

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> None:
        print(f"Loading model from {path}.")
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.episode = int(checkpoint.get("episode", 0))
