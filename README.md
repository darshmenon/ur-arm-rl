# mujoco-ur-arm-rl

Reinforcement learning training environment for the Universal Robots UR5e arm using MuJoCo, with a Robotiq 2F-85 gripper and graspable objects.

![Simulation Preview](assets/sim_preview.png)

## Overview

- **Robot**: UR5e + Robotiq 2F-85 gripper
- **Task**: Reach / pick-and-place
- **Algorithm**: SAC (Stable-Baselines3)
- **Simulator**: MuJoCo 3.x

## Setup

```bash
pip install mujoco gymnasium stable-baselines3
```

Clone [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) to `/home/asimov/mujoco_menagerie`.

## Train

```bash
python train.py
```

Best model saved to `models/`, logs to `logs/`.

## Visualize

```bash
python ur5e_with_gripper.py
```
