#!/usr/bin/env python3
"""
train.py — RLForge training entrypoint
========================================
Train any supported algorithm on any Gymnasium environment.

Examples
--------
# PPO on CartPole (default)
python train.py

# DQN on LunarLander
python train.py --algo dqn --env LunarLander-v2 --total-timesteps 300000

# PPO on MuJoCo HalfCheetah
python train.py --algo ppo --env HalfCheetah-v4 --total-timesteps 2000000 --hidden 256 256
"""

import argparse
import gymnasium as gym

from rlforge.algorithms.ppo import PPO, PPOConfig
from rlforge.algorithms.dqn import DQN, DQNConfig
from rlforge.utils.logger import Logger


ALGOS = {
    "ppo": (PPO, PPOConfig),
    "dqn": (DQN, DQNConfig),
}


def parse_args():
    p = argparse.ArgumentParser(description="RLForge trainer")
    p.add_argument("--algo", default="ppo", choices=list(ALGOS), help="Algorithm")
    p.add_argument("--env", default="CartPole-v1", help="Gymnasium environment ID")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=None, help="Learning rate override")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64], help="Hidden layer sizes")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-episodes", type=int, default=10, help="Episodes to eval after training")
    return p.parse_args()


def main():
    args = parse_args()
    env = gym.make(args.env)
    AlgoClass, ConfigClass = ALGOS[args.algo]

    # Build config from CLI overrides
    cfg_kwargs = dict(
        seed=args.seed,
        gamma=args.gamma,
        device=args.device,
        total_timesteps=args.total_timesteps,
        hidden_sizes=tuple(args.hidden),
    )
    if args.lr is not None:
        cfg_kwargs["learning_rate"] = args.lr

    cfg = ConfigClass(**cfg_kwargs)
    logger = Logger(log_dir="runs")
    agent = AlgoClass(env, cfg, logger)
    agent.learn()

    # Final evaluation
    if hasattr(agent, "evaluate"):
        print("Running final evaluation...")
        results = agent.evaluate(n_episodes=args.eval_episodes)
        print(f"  Mean reward : {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean length : {results['mean_length']:.1f}")

    env.close()


if __name__ == "__main__":
    main()
