import sys
sys.path.insert(0, "/home/asimov/mujoco-ur-arm-rl")

from envs.ur_reach_env import URReachEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import os

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

env = URReachEnv(render_mode=None)
check_env(env, warn=True)

eval_env = URReachEnv(render_mode=None)
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
    buffer_size=200_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
)

print("Starting SAC training on UR5e reach task...")
model.learn(total_timesteps=500_000, callback=eval_callback, progress_bar=True)
model.save(f"{MODEL_DIR}/ur5e_reach_final")
print("Training done. Model saved.")
