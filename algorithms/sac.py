import random
import collections
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple

from utils.commons import set_seed 
from utils.env import make_env
from utils.commons import soft_update

LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0

@dataclass
class SACConfig:
    # Basic settings
    env_id: str = "Pendulum-v1"
    gamma: float = 0.99

    # Network training & optimization
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-3
    batch_size: int = 256
    buffer_capacity: int = 100000
    tau: float = 0.005
    grad_clip_norm: float | None = None

    # Entropy temperature
    init_alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    target_entropy: float | None = None

    # Training loop
    warmup_steps: int = 2000
    train_every: int = 1
    max_episodes: int = 200

    # Repro & device
    seed: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActorNetwork(nn.Module):
    """
    Gaussian policy with tanh squashing.
    Outputs actions in the original Box action range.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int = 256,
    ):
        super().__init__()

        action_scale = torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32)
        action_bias = torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, act_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.fc_mean(x)
        logstd = self.fc_logstd(x)
        logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(logstd)
        return mean, std

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)

        raw_action = dist.rsample()
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * self.action_scale + self.action_bias

        # Tanh-squash correction:
        # log pi(a|s) = log N(u|mean,std) - log |d tanh(u) / du| - log(action_scale)
        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - torch.log(self.action_scale * (1.0 - squashed_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def deterministic(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class CriticNetwork(nn.Module):
    """
    Q(s, a) network.
    SAC uses two independent critics to reduce positive bias.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = state.view(state.shape[0], -1)
        action = action.view(action.shape[0], -1)
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, size: int, device: torch.device | str) -> Tuple[torch.Tensor, ...]:
        samples = random.sample(self.buffer, size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)
        actions = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=device)
        rewards = torch.as_tensor(np.asarray(rewards, dtype=np.float32), device=device).view(-1, 1)
        next_states = torch.as_tensor(np.asarray(next_states, dtype=np.float32), device=device)
        dones = torch.as_tensor(np.asarray(dones, dtype=np.float32), device=device).view(-1, 1)

        return states, actions, rewards, next_states, dones
    

# -----------------------------
# SAC Agent
# -----------------------------
class SACAgent:
    def __init__(self, cfg: SACConfig):
        set_seed(cfg.seed)

        env = make_env(cfg.env_id, cfg.seed)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("SAC only supports continuous Box action spaces.")

        if not np.all(np.isfinite(env.action_space.low)) or not np.all(np.isfinite(env.action_space.high)):
            raise ValueError("SAC requires finite action bounds.")

        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.act_dim = act_dim
        self.action_low = env.action_space.low.astype(np.float32)
        self.action_high = env.action_space.high.astype(np.float32)

        # Networks
        self.actor = ActorNetwork(obs_dim, act_dim, self.action_low, self.action_high, cfg.hidden_dim).to(self.device)

        self.critic_main1 = CriticNetwork(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.critic_main2 = CriticNetwork(obs_dim, act_dim, cfg.hidden_dim).to(self.device)

        self.critic_target1 = CriticNetwork(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.critic_target2 = CriticNetwork(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.critic_target1.load_state_dict(self.critic_main1.state_dict())
        self.critic_target2.load_state_dict(self.critic_main2.state_dict())
        self.critic_target1.requires_grad_(False)
        self.critic_target2.requires_grad_(False)

        # Optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim1 = torch.optim.Adam(self.critic_main1.parameters(), lr=cfg.critic_lr)
        self.critic_optim2 = torch.optim.Adam(self.critic_main2.parameters(), lr=cfg.critic_lr)

        # Automatic entropy tuning
        init_log_alpha = float(np.log(cfg.init_alpha))
        self.log_alpha = torch.tensor([init_log_alpha], dtype=torch.float32, device=self.device)
        self.log_alpha.requires_grad_(cfg.automatic_entropy_tuning)
        self.alpha_optim = (
            torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
            if cfg.automatic_entropy_tuning
            else None
        )
        self.target_entropy = -float(act_dim) if cfg.target_entropy is None else float(cfg.target_entropy)

        self.buffer = ReplayBuffer(cfg.buffer_capacity)

        # Counters
        self.global_step = 0
        self.gradient_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        if explore and self.global_step < self.cfg.warmup_steps:
            return np.asarray(self.env.action_space.sample(), dtype=np.float32)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)

        if explore:
            action, _ = self.actor.sample(state_tensor)
        else:
            action = self.actor.deterministic(state_tensor)

        action_np = action.cpu().numpy().reshape(self.env.action_space.shape).astype(np.float32)
        return np.clip(action_np, self.action_low, self.action_high)

    def update(self) -> dict[str, float]:
        cfg = self.cfg
        states, actions, rewards, next_states, dones = self.buffer.sample(cfg.batch_size, self.device)

        # Critic target
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1 = self.critic_target1(next_states, next_actions)
            next_q2 = self.critic_target2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha.detach() * next_log_probs
            q_target = rewards + cfg.gamma * (1.0 - dones) * next_q

        # Critic 1 update
        q1 = self.critic_main1(states, actions)
        critic_loss1 = F.mse_loss(q1, q_target)

        self.critic_optim1.zero_grad()
        critic_loss1.backward()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic_main1.parameters(), cfg.grad_clip_norm)
        self.critic_optim1.step()

        # Critic 2 update
        q2 = self.critic_main2(states, actions)
        critic_loss2 = F.mse_loss(q2, q_target)

        self.critic_optim2.zero_grad()
        critic_loss2.backward()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic_main2.parameters(), cfg.grad_clip_norm)
        self.critic_optim2.step()

        # Actor update
        for p in self.critic_main1.parameters():
            p.requires_grad_(False)
        for p in self.critic_main2.parameters():
            p.requires_grad_(False)

        new_actions, log_probs = self.actor.sample(states)
        new_q1 = self.critic_main1(states, new_actions)
        new_q2 = self.critic_main2(states, new_actions)
        min_new_q = torch.min(new_q1, new_q2)
        actor_loss = (self.alpha.detach() * log_probs - min_new_q).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
        self.actor_optim.step()

        for p in self.critic_main1.parameters():
            p.requires_grad_(True)
        for p in self.critic_main2.parameters():
            p.requires_grad_(True)

        # Alpha update
        if cfg.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            assert self.alpha_optim is not None
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
        else:
            alpha_loss = torch.tensor(float("nan"), device=self.device)

        # Target networks
        soft_update(self.critic_target1, self.critic_main1, cfg.tau)
        soft_update(self.critic_target2, self.critic_main2, cfg.tau)

        self.gradient_step += 1

        return {
            "critic_loss": float((critic_loss1 + critic_loss2).item()),
            "critic_loss1": float(critic_loss1.item()),
            "critic_loss2": float(critic_loss2.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "target_entropy": float(self.target_entropy),
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
        }

    def train(self) -> list[dict[str, float]]:
        cfg = self.cfg
        metrics: list[dict[str, float]] = []
        last_update_info: dict[str, float] | None = None

        iterator = tqdm(range(cfg.max_episodes), desc="SAC")
        for episode in iterator:
            reset_seed = None if cfg.seed is None else cfg.seed + episode
            state, _ = self.env.reset(seed=reset_seed)
            if reset_seed is not None:
                self.env.action_space.seed(reset_seed)

            episode_return = 0.0
            episode_length = 0
            done = False

            while not done:
                action = self.select_action(state, explore=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Store only true terminal signal for bootstrapping.
                # For time-limit truncation, SAC should usually bootstrap.
                done = terminated or truncated
                terminal = float(terminated)

                self.buffer.append(state, action, float(reward), next_state, terminal)

                episode_return += float(reward)
                episode_length += 1
                self.global_step += 1
                state = next_state

                should_update = (
                    len(self.buffer) >= max(cfg.warmup_steps, cfg.batch_size)
                    and self.global_step % cfg.train_every == 0
                )
                if should_update:
                    last_update_info = self.update()

            recent_returns = [m["episode_return"] for m in metrics[-19:]] + [episode_return]
            mean_return = float(np.mean(recent_returns))

            row = {
                "episode": float(episode + 1),
                "global_step": float(self.global_step),
                "gradient_step": float(self.gradient_step),
                "episode_return": float(episode_return),
                "episode_length": float(episode_length),
                "mean_return_20": mean_return,
                "buffer_size": float(len(self.buffer)),
            }

            if last_update_info is None:
                row.update(
                    {
                        "critic_loss": float("nan"),
                        "critic_loss1": float("nan"),
                        "critic_loss2": float("nan"),
                        "actor_loss": float("nan"),
                        "alpha_loss": float("nan"),
                        "alpha": float(self.alpha.item()),
                        "target_entropy": float(self.target_entropy),
                        "q1_mean": float("nan"),
                        "q2_mean": float("nan"),
                    }
                )
            else:
                row.update(last_update_info)

            metrics.append(row)

            iterator.set_postfix(
                {
                    "return": f"{episode_return:.1f}",
                    "mean20": f"{mean_return:.1f}",
                    "alpha": f"{row['alpha']:.3f}",
                    "step": self.global_step,
                }
            )

        return metrics

    def save(self, path: str | Path) -> None:
        print(f"Saving model to {path}.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.cfg.to_dict(),
            "actor": self.actor.state_dict(),
            "critic_main1": self.critic_main1.state_dict(),
            "critic_main2": self.critic_main2.state_dict(),
            "critic_target1": self.critic_target1.state_dict(),
            "critic_target2": self.critic_target2.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim1": self.critic_optim1.state_dict(),
            "critic_optim2": self.critic_optim2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "global_step": self.global_step,
            "gradient_step": self.gradient_step,
            "target_entropy": self.target_entropy,
        }

        if self.alpha_optim is not None:
            checkpoint["alpha_optim"] = self.alpha_optim.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> None:
        print(f"Loading model from {path}.")
        checkpoint = torch.load(path, map_location=map_location or self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_main1.load_state_dict(checkpoint["critic_main1"])
        self.critic_main2.load_state_dict(checkpoint["critic_main2"])
        self.critic_target1.load_state_dict(checkpoint["critic_target1"])
        self.critic_target2.load_state_dict(checkpoint["critic_target2"])

        self.actor_optim.load_state_dict(checkpoint["actor_optim"])
        self.critic_optim1.load_state_dict(checkpoint["critic_optim1"])
        self.critic_optim2.load_state_dict(checkpoint["critic_optim2"])

        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device).view_as(self.log_alpha))
        if self.alpha_optim is not None and "alpha_optim" in checkpoint:
            self.alpha_optim.load_state_dict(checkpoint["alpha_optim"])

        self.global_step = int(checkpoint.get("global_step", 0))
        self.gradient_step = int(checkpoint.get("gradient_step", 0))
        self.target_entropy = float(checkpoint.get("target_entropy", self.target_entropy))
