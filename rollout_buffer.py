"""
Rollout Buffer
==============
Stores a single on-policy rollout of fixed length.
All tensors live on the target device from the start.
"""

from __future__ import annotations
import torch


class RolloutBuffer:
    """Fixed-length circular buffer for PPO rollouts."""

    def __init__(self, rollout_steps: int, obs_dim: int, act_dim: int, device: torch.device):
        self.T = rollout_steps
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._ptr = 0
        self._full = False
        self._alloc()

    def _alloc(self):
        T, d = self.T, self.device
        self._obs      = torch.zeros(T, self.obs_dim, device=d)
        self._acts     = torch.zeros(T, self.act_dim, device=d)
        self._logprobs = torch.zeros(T, device=d)
        self._values   = torch.zeros(T, device=d)
        self._rewards  = torch.zeros(T, device=d)
        self._dones    = torch.zeros(T, device=d)

    def add(self, obs, action, logprob, value, reward, done):
        i = self._ptr
        self._obs[i]      = obs
        self._acts[i]     = action.unsqueeze(0) if action.dim() == 0 else action
        self._logprobs[i] = logprob
        self._values[i]   = value
        self._rewards[i]  = reward
        self._dones[i]    = done
        self._ptr = (i + 1) % self.T
        if self._ptr == 0:
            self._full = True

    def get(self):
        """Return all stored tensors in insertion order."""
        n = self.T if self._full else self._ptr
        return (
            self._obs[:n],
            self._acts[:n],
            self._logprobs[:n],
            self._values[:n],
            self._rewards[:n],
            self._dones[:n],
        )

    def reset(self):
        self._ptr = 0
        self._full = False
        self._alloc()
