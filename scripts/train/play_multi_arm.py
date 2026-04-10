import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs.ur_dual_arm_env import URDualArmEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a multi-arm SAC policy.")
    parser.add_argument("--model", required=True, help="Path to multi-arm SAC model zip.")
    parser.add_argument("--arms", type=int, default=8, help="Even number of arms to evaluate.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--curriculum", choices=["none", "easy_grasp"], default="easy_grasp")
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer.")
    return parser.parse_args()


def main():
    args = parse_args()
    env = URDualArmEnv(
        render_mode="human" if args.viewer else None,
        arm_count=args.arms,
        curriculum_mode=args.curriculum,
    )
    model = SAC.load(args.model, device=args.device)

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False
            steps = 0

            while not terminated and not truncated:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1

                if args.viewer:
                    env.render()

            print(
                f"episode={episode + 1} steps={steps} reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated}",
                flush=True,
            )
    finally:
        env.close()

if __name__ == "__main__":
    main()
