import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from utils.commons import set_seed
from utils.env import make_env

# -----------------------------
# Hyperparameters
# -----------------------------
@dataclass
class TRPOConfig:
    # Basic settings
    env_id: str = "CartPole-v1"
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # TRPO hyperparameters
    max_kl: float = 0.01
    cg_iters: int = 10
    cg_damping: float = 0.1
    backtrack_iters: int = 10
    backtrack_coeff: float = 0.8

    # Critic training
    critic_epochs: int = 20
    critic_batch_size: int = 64
    critic_lr: float = 1e-3
    critic_grad_clip_norm: float | None = 10.0

    # Network
    hidden_dim: int = 128

    # Training loop
    batch_size: int = 2048              # rollout steps per policy update
    total_epochs: int = 50              # number of TRPO policy updates
    normalize_advantages: bool = True

    # Repro & device
    seed: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------
# Networks
# -----------------------------
class ActorNetwork(nn.Module):
    """
    Categorical policy network for discrete action spaces.

    The network returns logits instead of probabilities. Using
    torch.distributions.Categorical(logits=...) is numerically more stable than
    manually applying softmax and log.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class CriticNetwork(nn.Module):
    """State-value function V(s)."""
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


# -----------------------------
# General Advantage Estimation
# -----------------------------
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


# -----------------------------
# TRPO Agent
# -----------------------------
class TRPOAgent:
    def __init__(self, cfg: TRPOConfig):
        set_seed(cfg.seed)

        env = make_env(cfg.env_id, cfg.seed)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                "This TRPO implementation uses a categorical policy and only supports "
                "discrete action spaces. For continuous control, use a Gaussian policy."
            )

        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(env.action_space.n)

        self.actor = ActorNetwork(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim, cfg.hidden_dim).to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.global_step = 0
        self.episode_count = 0

    def _as_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)

    def get_dist(self, states: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.actor(states)
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def select_action(self, state_tensor: torch.Tensor, explore: bool = True) -> int:
        dist = self.get_dist(state_tensor)
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(dist.logits, dim=-1)

        return int(action.item())

    @staticmethod
    def get_flat_params(model: nn.Module) -> torch.Tensor:
        return torch.cat([param.detach().view(-1) for param in model.parameters()])

    @staticmethod
    def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
        prev_ind = 0
        with torch.no_grad():
            for param in model.parameters():
                flat_size = int(np.prod(param.shape))
                param.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.shape))
                prev_ind += flat_size

    @staticmethod
    def _flat_grad(grads: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat([grad.contiguous().view(-1) for grad in grads])

    def compute_surrogate_obj(self, states: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor, old_log_probs: torch.Tensor) -> torch.Tensor:
        """surrogate_obj = (new_prob / old_prob) * advantages"""
        new_log_probs = self.get_dist(states).log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        return torch.mean(ratio * advantages)

    def compute_kl(self, states: torch.Tensor, old_dist: torch.distributions.Categorical) -> torch.Tensor:
        """return kl(pi_old || pi_new)"""
        new_dist = self.get_dist(states)
        return torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

    def compute_fvp(self, v: torch.Tensor, states: torch.Tensor, old_dist: torch.distributions.Categorical) -> torch.Tensor:
        """Compute Fisher-vector product: H * v."""
        kl = self.compute_kl(states, old_dist)

        kl_grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        flat_kl_grads = self._flat_grad(kl_grads)

        kl_v_product = torch.dot(flat_kl_grads, v)
        kl_2nd_grads = torch.autograd.grad(kl_v_product, self.actor.parameters())
        flat_kl_2nd_grads = self._flat_grad(kl_2nd_grads).detach()

        return flat_kl_2nd_grads + self.cfg.cg_damping * v

    def conjugate_gradient(self, b: torch.Tensor, states: torch.Tensor, old_dist: torch.distributions.Categorical) -> torch.Tensor:
        """Approximately solve Hx = b, returning x = H^{-1}b."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for _ in range(self.cfg.cg_iters):
            Avp = self.compute_fvp(p, states, old_dist)
            denom = torch.dot(p, Avp).clamp_min(1e-10)
            alpha = rdotr / denom

            x += alpha * p
            r -= alpha * Avp

            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break

            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def line_search(
        self,
        full_step: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_dist: torch.distributions.Categorical,
    ) -> dict[str, float]:
        """Backtracking line search with KL constraint."""
        old_params = self.get_flat_params(self.actor)

        with torch.no_grad():
            old_obj = self.compute_surrogate_obj(states, actions, advantages, old_log_probs)
            old_obj_value = float(old_obj.item())

        for i in range(self.cfg.backtrack_iters):
            step_fraction = self.cfg.backtrack_coeff ** i
            new_params = old_params + step_fraction * full_step
            self.set_flat_params(self.actor, new_params)

            with torch.no_grad():
                new_obj = self.compute_surrogate_obj(states, actions, advantages, old_log_probs)
                new_kl = self.compute_kl(states, old_dist)

            obj_improvement = float((new_obj - old_obj).item())
            kl_value = float(new_kl.item())

            if obj_improvement > 0.0 and kl_value <= self.cfg.max_kl:
                return {
                    "line_search_success": 1.0,
                    "line_search_steps": float(i + 1),
                    "surrogate_improvement": obj_improvement,
                    "kl": kl_value,
                    "actor_objective": float(new_obj.item()),
                }

        self.set_flat_params(self.actor, old_params)
        return {
            "line_search_success": 0.0,
            "line_search_steps": float(self.cfg.backtrack_iters),
            "surrogate_improvement": 0.0,
            "kl": 0.0,
            "actor_objective": old_obj_value,
        }

    def update_critic(self, states: torch.Tensor, returns: torch.Tensor) -> dict[str, float]:
        dataset = TensorDataset(states, returns)
        loader = DataLoader(dataset, batch_size=self.cfg.critic_batch_size, shuffle=True)

        last_loss = torch.tensor(float("nan"), device=self.device)
        for _ in range(self.cfg.critic_epochs):
            for batch_states, batch_returns in loader:
                values = self.critic(batch_states).squeeze(-1)
                loss = F.mse_loss(values, batch_returns)

                self.critic_optim.zero_grad()
                loss.backward()
                if self.cfg.critic_grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        self.cfg.critic_grad_clip_norm,
                    )
                self.critic_optim.step()
                last_loss = loss.detach()

        with torch.no_grad():
            values_pred = self.critic(states).squeeze(-1)
            explained_variance = self.compute_explained_variance(values_pred, returns)

        return {
            "critic_loss": float(last_loss.item()),
            "explained_variance": float(explained_variance),
        }

    @staticmethod
    def compute_explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """1 - Var(y_true - y_pred) / Var(y_true). Higher is better; 1 is perfect."""
        y_var = torch.var(y_true)
        diff_var = torch.var(y_true - y_pred)
        return float((1.0 - diff_var / (y_var + 1e-8)).item())

    def update_batch(self, rollout_batch: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        states = rollout_batch["states"]
        actions = rollout_batch["actions"]
        advantages = rollout_batch["advantages"]
        returns = rollout_batch["returns"]

        # get old log probs
        with torch.no_grad():
            old_dist = self.get_dist(states)
            old_log_probs = old_dist.log_prob(actions)

        # compute grads of surrogate objective
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantages, old_log_probs)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grads = self._flat_grad(grads).detach()

        # step_dir = H^-1 g
        step_dir = self.conjugate_gradient(obj_grads, states, old_dist)
        H_step_dir = self.compute_fvp(step_dir, states, old_dist)
        sHs = torch.dot(step_dir, H_step_dir)

        if not torch.isfinite(sHs) or sHs <= 0:
            line_search_info = {
                "line_search_success": 0.0,
                "line_search_steps": 0.0,
                "surrogate_improvement": 0.0,
                "kl": 0.0,
                "actor_objective": float(surrogate_obj.item()),
            }
        else:
            step_scale = torch.sqrt(2.0 * cfg.max_kl / (sHs + 1e-8))
            full_step = step_scale * step_dir
            line_search_info = self.line_search(
                full_step,
                states,
                actions,
                advantages,
                old_log_probs,
                old_dist,
            )

        critic_info = self.update_critic(states, returns)
        return {**line_search_info, **critic_info}

    def train(self) -> list[dict[str, float]]:
        cfg = self.cfg
        metrics: list[dict[str, float]] = []
        episode_returns: list[float] = []
        episode_lengths: list[int] = []

        reset_seed = cfg.seed
        state, _ = self.env.reset(seed=reset_seed)

        current_episode_return = 0.0
        current_episode_length = 0

        iterator = tqdm(range(cfg.total_epochs), desc="TRPO")
        for epoch in iterator:
            # on-policy training
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_dones = []
            batch_values = []

            for _ in range(cfg.batch_size):
                state_tensor = self._as_state_tensor(state)

                with torch.no_grad():
                    value = self.critic(state_tensor).item()

                action = self.select_action(state_tensor, explore=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append(float(reward))
                batch_dones.append(float(done))
                batch_values.append(float(value))

                current_episode_return += float(reward)
                current_episode_length += 1
                self.global_step += 1
                state = next_state

                if done:
                    episode_returns.append(current_episode_return)
                    episode_lengths.append(current_episode_length)
                    self.episode_count += 1

                    current_episode_return = 0.0
                    current_episode_length = 0

                    reset_seed = None if cfg.seed is None else cfg.seed + self.episode_count
                    state, _ = self.env.reset(seed=reset_seed)

            with torch.no_grad():
                last_value = self.critic(self._as_state_tensor(state)).item()

            states = torch.as_tensor(np.asarray(batch_states, dtype=np.float32), device=self.device)
            actions = torch.as_tensor(np.asarray(batch_actions, dtype=np.int64), device=self.device)
            rewards = torch.as_tensor(np.asarray(batch_rewards, dtype=np.float32), device=self.device)
            dones = torch.as_tensor(np.asarray(batch_dones, dtype=np.float32), device=self.device)
            values = torch.as_tensor(np.asarray(batch_values, dtype=np.float32), device=self.device)
            next_values = torch.cat([values[1:], torch.tensor([last_value], device=self.device)])

            advantages, returns = compute_gae(
                rewards,
                values,
                next_values,
                dones,
                cfg.gamma,
                cfg.gae_lambda,
            )
            if cfg.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            rollout_batch = {
                "states": states,
                "actions": actions,
                "advantages": advantages,
                "returns": returns,
            }
            update_info = self.update_batch(rollout_batch)

            if episode_returns:
                mean_return_20 = float(np.mean(episode_returns[-20:]))
                last_episode_return = float(episode_returns[-1])
                last_episode_length = float(episode_lengths[-1])
            else:
                mean_return_20 = float("nan")
                last_episode_return = float("nan")
                last_episode_length = float("nan")

            row = {
                "epoch": float(epoch + 1),
                "episode": float(self.episode_count),
                "global_step": float(self.global_step),
                "batch_reward_mean": float(np.mean(batch_rewards)),
                "last_episode_return": last_episode_return,
                "last_episode_length": last_episode_length,
                "mean_return_20": mean_return_20,
                **update_info,
            }
            metrics.append(row)

            iterator.set_postfix(
                {
                    "mean20": f"{mean_return_20:.1f}",
                    "episodes": self.episode_count,
                    "kl": f"{row['kl']:.4f}",
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
                "critic": self.critic.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "global_step": self.global_step,
                "episode_count": self.episode_count,
            },
            path,
        )

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> None:
        print(f"Loading model from {path}.")
        checkpoint = torch.load(path, map_location=map_location or self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.episode_count = int(checkpoint.get("episode_count", 0))
