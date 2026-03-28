# mujoco-ur-arm-rl

Reinforcement learning training environment for the Universal Robots UR5e arm using MuJoCo, with a Robotiq 2F-85 gripper and graspable objects.

![Simulation Preview](assets/sim_preview.png)

## Overview

- **Robot**: UR5e + Robotiq 2F-85 gripper
- **Task**: Reach / pick-and-place / cooperative handover
- **Algorithm**: SAC (Stable-Baselines3)
- **Simulator**: MuJoCo 3.x

## Environments

| Env | Task | Arms |
|-----|------|------|
| `URReachEnv` | Move end-effector to target | 1 |
| `URPickPlaceEnv` | Pick box and place at target | 1 |
| `URDualArmEnv` | Cooperative 4-arm handover | 4 |

## 4-Arm Training

![4-Arm Training](assets/4arm_training.png)

Four UR5e arms with Robotiq 2F-85 grippers arranged around a central table. Each arm has its own items to pick, with a shared handover object in the center.

```bash
python train_dual_arm_live.py
```

## Setup

```bash
pip install mujoco gymnasium stable-baselines3
```

Clone [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) to `/home/asimov/mujoco_menagerie`.

## Train

```bash
# Single arm reach
python train.py

# Single arm pick-and-place
python train_pick_place.py

# 4-arm cooperative handover (with live viewer)
python train_dual_arm_live.py
```

Best model saved to `models/`, logs to `logs/`.

## Visualize

```bash
python ur5e_with_gripper.py
```
