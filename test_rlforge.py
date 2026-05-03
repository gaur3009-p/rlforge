"""
Tests for RLForge — fast smoke tests, no GPU required.
"""
import torch
import numpy as np
import gymnasium as gym
import pytest

from rlforge.networks.actor_critic import ActorCritic
from rlforge.utils.rollout_buffer import RolloutBuffer
from rlforge.algorithms.ppo import PPO, PPOConfig
from rlforge.algorithms.dqn import DQN, DQNConfig, ReplayBuffer


# ─── Network tests ────────────────────────────────────────────────────────────

class TestActorCritic:
    def test_discrete_forward(self):
        net = ActorCritic(obs_dim=4, act_dim=2, discrete=True)
        obs = torch.randn(8, 4)
        dist, val = net(obs)
        assert val.shape == (8,)
        assert dist.probs.shape == (8, 2)

    def test_continuous_forward(self):
        net = ActorCritic(obs_dim=8, act_dim=3, discrete=False)
        obs = torch.randn(4, 8)
        dist, val = net(obs)
        assert val.shape == (4,)
        assert dist.mean.shape == (4, 3)

    def test_evaluate_returns_correct_shapes(self):
        net = ActorCritic(obs_dim=4, act_dim=2, discrete=True)
        obs = torch.randn(16, 4)
        acts = torch.randint(0, 2, (16, 1))
        lp, ent, val = net.evaluate(obs, acts)
        assert lp.shape == (16,)
        assert ent.shape == (16,)
        assert val.shape == (16,)

    def test_param_count(self):
        net = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=(64, 64))
        assert net.count_params() > 0


# ─── Buffer tests ─────────────────────────────────────────────────────────────

class TestRolloutBuffer:
    def test_add_and_get(self):
        buf = RolloutBuffer(10, 4, 1, torch.device("cpu"))
        for _ in range(5):
            buf.add(
                obs=torch.randn(4),
                action=torch.tensor([0]),
                logprob=torch.tensor(0.5),
                value=torch.tensor(1.0),
                reward=torch.tensor(1.0),
                done=torch.tensor(0.0),
            )
        obs, acts, lp, val, rew, done = buf.get()
        assert obs.shape == (5, 4)
        assert rew.shape == (5,)

    def test_reset(self):
        buf = RolloutBuffer(10, 4, 1, torch.device("cpu"))
        buf.add(torch.randn(4), torch.tensor([0]),
                torch.tensor(0.0), torch.tensor(0.0),
                torch.tensor(0.0), torch.tensor(0.0))
        buf.reset()
        obs, *_ = buf.get()
        assert obs.shape[0] == 0


class TestReplayBuffer:
    def test_add_sample(self):
        buf = ReplayBuffer(100, 4, torch.device("cpu"))
        for _ in range(50):
            buf.add(np.random.randn(4), 0, 1.0, np.random.randn(4), False)
        assert len(buf) == 50
        obs, acts, rew, nobs, done = buf.sample(16)
        assert obs.shape == (16, 4)


# ─── PPO smoke test ───────────────────────────────────────────────────────────

class TestPPO:
    def test_short_train(self):
        env = gym.make("CartPole-v1")
        cfg = PPOConfig(
            total_timesteps=2048,
            rollout_steps=512,
            n_epochs=2,
            batch_size=64,
            device="cpu",
        )
        agent = PPO(env, cfg)
        agent.learn(total_timesteps=512)
        env.close()

    def test_evaluate(self):
        env = gym.make("CartPole-v1")
        cfg = PPOConfig(device="cpu")
        agent = PPO(env, cfg)
        results = agent.evaluate(n_episodes=2)
        assert "mean_reward" in results
        env.close()


# ─── DQN smoke test ───────────────────────────────────────────────────────────

class TestDQN:
    def test_short_train(self):
        env = gym.make("CartPole-v1")
        cfg = DQNConfig(
            total_timesteps=1100,
            learning_starts=100,
            device="cpu",
        )
        agent = DQN(env, cfg)
        agent.learn(total_timesteps=1100)
        env.close()
