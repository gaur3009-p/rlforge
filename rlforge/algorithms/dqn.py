"""
Deep Q-Network (DQN)  +  Double DQN
=====================================
Mnih et al., 2015 (DQN)  — https://arxiv.org/abs/1312.5602
van Hasselt et al., 2016 (DDQN) — https://arxiv.org/abs/1509.06461

Implements:
  • Experience replay buffer
  • Target network with periodic hard updates
  • ε-greedy exploration with linear decay
  • Double DQN to reduce overestimation bias (optional)
"""

from __future__ import annotations
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rlforge.utils.logger import Logger


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Uniform experience replay buffer.

    Stores (s, a, r, s', done) tuples and returns random mini-batches.
    """

    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self._ptr = 0
        self._size = 0
        self._obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._acts     = np.zeros(capacity, dtype=np.int64)
        self._rewards  = np.zeros(capacity, dtype=np.float32)
        self._dones    = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        i = self._ptr
        self._obs[i] = obs
        self._acts[i] = action
        self._rewards[i] = reward
        self._next_obs[i] = next_obs
        self._dones[i] = float(done)
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._size, size=batch_size)
        d = self.device
        return (
            torch.FloatTensor(self._obs[idx]).to(d),
            torch.LongTensor(self._acts[idx]).to(d),
            torch.FloatTensor(self._rewards[idx]).to(d),
            torch.FloatTensor(self._next_obs[idx]).to(d),
            torch.FloatTensor(self._dones[idx]).to(d),
        )

    def __len__(self): return self._size


# ─── Q-Network ────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple = (128, 128)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)
        # Xavier init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class DQNConfig:
    # Core
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    train_freq: int = 4           # update every N env steps
    target_update_freq: int = 500  # hard target network update

    # Exploration (linear ε decay)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

    # Architecture
    hidden_sizes: tuple = (128, 128)
    double_dqn: bool = True

    # Misc
    seed: int = 42
    device: str = "auto"
    total_timesteps: int = 200_000

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ─── DQN Trainer ─────────────────────────────────────────────────────────────

class DQN:
    """
    DQN / Double-DQN trainer.

    Usage
    -----
    >>> env = gym.make("CartPole-v1")
    >>> agent = DQN(env, DQNConfig())
    >>> agent.learn()
    """

    def __init__(self, env, config: Optional[DQNConfig] = None, logger: Optional[Logger] = None):
        self.env = env
        self.cfg = config or DQNConfig()
        self.device = torch.device(self.cfg.device)
        self.logger = logger or Logger()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.q_net     = QNetwork(obs_dim, n_actions, self.cfg.hidden_sizes).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, self.cfg.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim, self.device)

        self._t = 0
        self._ep = 0

    def _eps(self):
        """Linear ε-greedy schedule."""
        frac = min(1.0, self._t / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def _select_action(self, obs: np.ndarray) -> int:
        if random.random() < self._eps():
            return self.env.action_space.sample()
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return int(self.q_net(obs_t).argmax(dim=1).item())

    def _update(self):
        obs, acts, rewards, next_obs, dones = self.buffer.sample(self.cfg.batch_size)

        with torch.no_grad():
            if self.cfg.double_dqn:
                # DDQN: select with online, evaluate with target
                next_actions = self.q_net(next_obs).argmax(1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(1)[0]

            target_q = rewards + self.cfg.gamma * next_q * (1 - dones)

        current_q = self.q_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(current_q, target_q)  # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def learn(self, total_timesteps: Optional[int] = None):
        total_steps = total_timesteps or self.cfg.total_timesteps
        obs, _ = self.env.reset()
        ep_ret, ep_len = 0.0, 0
        start = time.time()

        print(f"\n{'='*60}")
        print(f"  RLForge · {'Double ' if self.cfg.double_dqn else ''}DQN")
        print(f"  Device : {self.device}")
        print(f"{'='*60}\n")

        for t in range(total_steps):
            self._t = t
            action = self._select_action(obs)
            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            self.buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_ret += reward
            ep_len += 1

            if done:
                fps = int(t / max(1, time.time() - start))
                print(f"  ep={self._ep:>5}  step={t:>7,}  reward={ep_ret:>7.1f}  ε={self._eps():.3f}  fps={fps}")
                self.logger.log({"timestep": t, "episode": self._ep,
                                 "reward": ep_ret, "ep_len": ep_len, "epsilon": self._eps()})
                self._ep += 1
                ep_ret, ep_len = 0.0, 0
                obs, _ = self.env.reset()

            if t >= self.cfg.learning_starts and t % self.cfg.train_freq == 0:
                self._update()

            if t % self.cfg.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        print("\n✓ DQN training complete\n")
