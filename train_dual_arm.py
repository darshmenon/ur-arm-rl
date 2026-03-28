import sys
sys.path.insert(0, "/home/asimov/mujoco-ur-arm-rl")

from envs.ur_dual_arm_env import URDualArmEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import os

LOG_DIR = "logs/dual_arm"
MODEL_DIR = "models/dual_arm"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

env = URDualArmEnv()
check_env(env, warn=True)

eval_env = URDualArmEnv()
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    buffer_size=500_000,
    batch_size=512,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
)

print("Starting SAC training on dual UR5e handover task...")
model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)
model.save(f"{MODEL_DIR}/ur5e_dual_arm_final")
print("Done. Model saved.")
