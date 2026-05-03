"""
Actor-Critic Network
=====================
Shared MLP backbone with two heads:
  • Actor  — outputs action distribution (Categorical or Gaussian)
  • Critic — outputs scalar value estimate V(s)

Orthogonal initialisation (Hubert et al.) is used throughout,
with the policy head scaled by 0.01 and the value head by 1.0.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def _make_mlp(in_dim: int, hidden_sizes: tuple, activation: str) -> nn.Sequential:
    act = nn.Tanh if activation == "tanh" else nn.ReLU
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    return nn.Sequential(*layers)


def _ortho_init(layer: nn.Linear, gain: float = np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic network.

    Parameters
    ----------
    obs_dim      : dimension of observation vector
    act_dim      : number of actions (discrete) or action dimensions (continuous)
    hidden_sizes : MLP hidden layer sizes
    activation   : "tanh" or "relu"
    discrete     : True for Categorical policy, False for diagonal Gaussian
    log_std_init : initial log-std for Gaussian policy (continuous only)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple = (64, 64),
        activation: str = "tanh",
        discrete: bool = True,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.discrete = discrete
        self.act_dim = act_dim

        # ── Shared trunk ──────────────────────────────────────────────────────
        self.trunk = _make_mlp(obs_dim, hidden_sizes, activation)
        for mod in self.trunk.modules():
            if isinstance(mod, nn.Linear):
                _ortho_init(mod)

        trunk_out = hidden_sizes[-1]

        # ── Actor head ────────────────────────────────────────────────────────
        self.actor_head = _ortho_init(nn.Linear(trunk_out, act_dim), gain=0.01)

        if not discrete:
            # Learnable per-action log-std (state-independent)
            self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

        # ── Critic head ───────────────────────────────────────────────────────
        self.critic_head = _ortho_init(nn.Linear(trunk_out, 1), gain=1.0)

    def _distribution(self, obs: torch.Tensor):
        feat = self.trunk(obs)
        logits_or_mean = self.actor_head(feat)
        if self.discrete:
            return Categorical(logits=logits_or_mean)
        else:
            std = self.log_std.exp().expand_as(logits_or_mean)
            return Normal(logits_or_mean, std)

    def forward(self, obs: torch.Tensor):
        """
        Returns (distribution, value_estimate).
        Used during rollout collection.
        """
        feat = self.trunk(obs)
        dist = self._distribution(obs)
        value = self.critic_head(feat).squeeze(-1)
        return dist, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Returns (log_probs, entropy, values).
        Used during the PPO update step.
        """
        dist = self._distribution(obs)
        if self.discrete:
            log_probs = dist.log_prob(actions.squeeze(-1).long())
        else:
            log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy()
        if not self.discrete:
            entropy = entropy.sum(-1)

        feat = self.trunk(obs)
        values = self.critic_head(feat).squeeze(-1)
        return log_probs, entropy, values

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
