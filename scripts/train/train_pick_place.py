import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs.ur_pick_place_env import URPickPlaceEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import os

LOG_DIR = "logs/pick_place"
MODEL_DIR = "models/pick_place"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

env = URPickPlaceEnv()
check_env(env, warn=True)

eval_env = URPickPlaceEnv()
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
    buffer_size=300_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
)

print("Starting SAC training on UR5e pick-and-place task...")
model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)
model.save(f"{MODEL_DIR}/ur5e_pick_place_final")
print("Done. Model saved.")
