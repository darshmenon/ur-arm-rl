import sys
sys.path.insert(0, "/home/asimov/mujoco-ur-arm-rl")

from envs.ur_dual_arm_env import URDualArmEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import mujoco.viewer
import os

LOG_DIR = "logs/dual_arm_live"
MODEL_DIR = "models/dual_arm"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class LiveViewerCallback(BaseCallback):
    def __init__(self, env, render_every=5):
        super().__init__()
        self._env = env
        self._render_every = render_every
        self._viewer = None

    def _on_training_start(self):
        self._viewer = mujoco.viewer.launch_passive(
            self._env.model, self._env.data
        )

    def _on_step(self):
        if self.n_calls % self._render_every == 0 and self._viewer is not None:
            self._viewer.sync()
        return True

    def _on_training_end(self):
        if self._viewer is not None:
            self._viewer.close()


# Use render_mode=None for speed, viewer syncs directly to env data
env = URDualArmEnv(render_mode=None)
eval_env = URDualArmEnv(render_mode=None)

live_cb = LiveViewerCallback(env, render_every=10)
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=5,
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
    target_entropy=-14,       # half of action dim — keeps exploration alive longer
    learning_starts=5_000,    # fill buffer with random experience first
    train_freq=4,
    gradient_steps=4,
)

print("Starting SAC training with live viewer...")
model.learn(
    total_timesteps=1_000_000,
    callback=[live_cb, eval_cb],
    progress_bar=True,
)
model.save(f"{MODEL_DIR}/ur5e_4arm_final")
print("Done.")
