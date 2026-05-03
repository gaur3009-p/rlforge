"""
Proximal Policy Optimization (PPO)
===================================
Schulman et al., 2017 — https://arxiv.org/abs/1707.06347

Key ideas implemented here:
  • Clipped surrogate objective  L^CLIP
  • Generalised Advantage Estimation (GAE-λ)
  • Shared actor-critic network with separate heads
  • Entropy bonus for exploration
  • Value-function loss clipping (optional)
  • Gradient norm clipping
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from rlforge.networks.actor_critic import ActorCritic
from rlforge.utils.rollout_buffer import RolloutBuffer
from rlforge.utils.logger import Logger


# ─── Hyperparameter dataclass ────────────────────────────────────────────────

@dataclass
class PPOConfig:
    # Core PPO
    clip_eps: float = 0.2          # ε in L^CLIP
    gamma: float = 0.99            # discount factor
    gae_lambda: float = 0.95       # λ in GAE
    n_epochs: int = 10             # gradient steps per rollout
    batch_size: int = 64           # mini-batch size
    rollout_steps: int = 2048      # steps collected before each update

    # Losses
    value_coef: float = 0.5        # c₁ — value loss weight
    entropy_coef: float = 0.01     # c₂ — entropy bonus weight
    clip_value_loss: bool = True   # clip V-loss like policy loss

    # Optimisation
    learning_rate: float = 3e-4
    lr_anneal: bool = True         # linearly anneal LR to 0
    max_grad_norm: float = 0.5     # gradient clipping

    # Misc
    seed: int = 42
    device: str = "auto"           # "cpu" | "cuda" | "auto"
    log_interval: int = 1          # episodes between log writes
    save_interval: int = 50        # episodes between checkpoints
    total_timesteps: int = 1_000_000

    # Network
    hidden_sizes: tuple = (64, 64)
    activation: str = "tanh"       # "tanh" | "relu"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ─── PPO Trainer ─────────────────────────────────────────────────────────────

class PPO:
    """
    Full PPO trainer.

    Usage
    -----
    >>> import gymnasium as gym
    >>> from rlforge.algorithms.ppo import PPO, PPOConfig
    >>> env = gym.make("CartPole-v1")
    >>> agent = PPO(env, PPOConfig())
    >>> agent.learn(total_timesteps=500_000)
    """

    def __init__(self, env, config: Optional[PPOConfig] = None, logger: Optional[Logger] = None):
        self.env = env
        self.cfg = config or PPOConfig()
        self.device = torch.device(self.cfg.device)
        self.logger = logger or Logger()

        # Reproducibility
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # Network
        obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, "n"):
            act_dim = env.action_space.n
            self.discrete = True
        else:
            act_dim = env.action_space.shape[0]
            self.discrete = False

        self.policy = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=self.cfg.hidden_sizes,
            activation=self.cfg.activation,
            discrete=self.discrete,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate, eps=1e-5)
        self.buffer = RolloutBuffer(
            rollout_steps=self.cfg.rollout_steps,
            obs_dim=obs_dim,
            act_dim=1 if self.discrete else act_dim,
            device=self.device,
        )

        self._timestep = 0
        self._episode = 0
        self._best_reward = -np.inf

    # ── Core update ───────────────────────────────────────────────────────────

    def _compute_gae(self, rewards, values, dones, last_value):
        """
        Generalised Advantage Estimation (GAE-λ)

          δₜ = rₜ + γ · V(sₜ₊₁) · (1 − dₜ) − V(sₜ)
          Âₜ = δₜ + γλ · Âₜ₊₁ · (1 − dₜ)

        Returns advantages and discounted returns.
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_val * mask - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def _update(self, progress: float):
        """Single PPO update step over the collected rollout."""
        obs, acts, old_logprobs, values, rewards, dones = self.buffer.get()

        with torch.no_grad():
            _, last_val = self.policy(obs[-1].unsqueeze(0))

        advantages, returns = self._compute_gae(rewards, values, dones, last_val.squeeze())

        # Normalise advantages (zero-mean, unit-variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # LR annealing
        if self.cfg.lr_anneal:
            lr = self.cfg.learning_rate * (1.0 - progress)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        n = len(obs)
        indices = np.arange(n)
        clip_fracs = []

        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.cfg.batch_size):
                mb = indices[start:start + self.cfg.batch_size]
                mb_obs = obs[mb]
                mb_acts = acts[mb]
                mb_old_lp = old_logprobs[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]
                mb_val = values[mb]

                # Forward pass
                logprobs, entropy, new_values = self.policy.evaluate(mb_obs, mb_acts)

                # ─ Policy loss (clipped surrogate) ─────────────────────────
                # L^CLIP = E[min(r·Â, clip(r, 1−ε, 1+ε)·Â)]
                ratio = (logprobs - mb_old_lp).exp()           # πθ / πθ_old
                clip_frac = ((ratio - 1).abs() > self.cfg.clip_eps).float().mean().item()
                clip_fracs.append(clip_frac)

                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ─ Value loss ─────────────────────────────────────────────
                if self.cfg.clip_value_loss:
                    v_clipped = mb_val + (new_values - mb_val).clamp(-self.cfg.clip_eps, self.cfg.clip_eps)
                    vf_loss = torch.max((new_values - mb_ret) ** 2, (v_clipped - mb_ret) ** 2).mean()
                else:
                    vf_loss = ((new_values - mb_ret) ** 2).mean()

                # ─ Total loss ─────────────────────────────────────────────
                # L = L^CLIP − c₁·L^VF + c₂·S[πθ]
                entropy_loss = -entropy.mean()
                loss = (policy_loss
                        + self.cfg.value_coef * vf_loss
                        + self.cfg.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": vf_loss.item(),
            "entropy": -entropy_loss.item(),
            "clip_frac": np.mean(clip_fracs),
            "approx_kl": ((ratio - 1) - (logprobs - mb_old_lp)).mean().item(),
        }

    # ── Rollout collection ────────────────────────────────────────────────────

    def _collect_rollout(self):
        """Collect `rollout_steps` transitions from the environment."""
        obs, _ = self.env.reset()
        ep_rewards, ep_lengths = [], []
        ep_ret, ep_len = 0.0, 0

        for step in range(self.cfg.rollout_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_dist, value = self.policy(obs_t)
                action = action_dist.sample()
                logprob = action_dist.log_prob(action)
                if not self.discrete:
                    logprob = logprob.sum(-1)

            act_np = action.cpu().numpy().squeeze()
            next_obs, reward, terminated, truncated, _ = self.env.step(act_np)
            done = terminated or truncated

            self.buffer.add(
                obs=obs_t.squeeze(0),
                action=action.squeeze(0),
                logprob=logprob.squeeze(0),
                value=value.squeeze(0),
                reward=torch.tensor(reward, device=self.device),
                done=torch.tensor(float(done), device=self.device),
            )

            obs = next_obs
            ep_ret += reward
            ep_len += 1
            self._timestep += 1

            if done:
                ep_rewards.append(ep_ret)
                ep_lengths.append(ep_len)
                ep_ret, ep_len = 0.0, 0
                self._episode += 1
                obs, _ = self.env.reset()

        return ep_rewards, ep_lengths

    # ── Main training loop ────────────────────────────────────────────────────

    def learn(self, total_timesteps: Optional[int] = None):
        """Run the full PPO training loop."""
        total_steps = total_timesteps or self.cfg.total_timesteps
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"  RLForge · PPO")
        print(f"  Device : {self.device}")
        print(f"  Env    : {self.env.spec.id if self.env.spec else 'custom'}")
        print(f"  Steps  : {total_steps:,}")
        print(f"{'='*60}\n")

        while self._timestep < total_steps:
            progress = self._timestep / total_steps

            # Collect rollout
            ep_rewards, ep_lengths = self._collect_rollout()

            # Update policy
            metrics = self._update(progress)

            # Log
            if ep_rewards:
                mean_r = np.mean(ep_rewards)
                mean_l = np.mean(ep_lengths)
                fps = int(self._timestep / (time.time() - start_time))

                self.logger.log({
                    "timestep": self._timestep,
                    "episode": self._episode,
                    "mean_reward": mean_r,
                    "mean_ep_len": mean_l,
                    "fps": fps,
                    **metrics,
                })

                print(
                    f"  ep={self._episode:>5}  "
                    f"step={self._timestep:>8,}  "
                    f"reward={mean_r:>8.2f}  "
                    f"loss={metrics['policy_loss']:>7.4f}  "
                    f"ent={metrics['entropy']:>5.3f}  "
                    f"fps={fps}"
                )

                if mean_r > self._best_reward:
                    self._best_reward = mean_r
                    self.save("best_model.pt")

            self.buffer.reset()

        print(f"\n✓ Training complete · best reward: {self._best_reward:.2f}\n")
        self.save("final_model.pt")

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 10, render: bool = False) -> dict:
        """Evaluate the current policy deterministically."""
        rewards, lengths = [], []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done, ep_ret, ep_len = False, 0.0, 0
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                dist, _ = self.policy(obs_t)
                if self.discrete:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.mean
                obs, r, term, trunc, _ = self.env.step(action.cpu().numpy().squeeze())
                done = term or trunc
                ep_ret += r
                ep_len += 1
            rewards.append(ep_ret)
            lengths.append(ep_len)
        return {"mean_reward": np.mean(rewards), "std_reward": np.std(rewards),
                "mean_length": np.mean(lengths), "n_episodes": n_episodes}

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "timestep": self._timestep,
            "episode": self._episode,
            "best_reward": self._best_reward,
            "config": self.cfg,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._timestep = ckpt["timestep"]
        self._episode = ckpt["episode"]
        self._best_reward = ckpt["best_reward"]
        print(f"Loaded checkpoint: ep={self._episode}, reward={self._best_reward:.2f}")
