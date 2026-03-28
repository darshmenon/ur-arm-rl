# Project Concepts

## What This Project Is

This project trains robotic arms to perform manipulation tasks using reinforcement learning inside a physics simulator. The idea is to teach robots useful skills — reaching, picking, placing, and handing objects between arms — entirely in simulation, then deploy the learned policies to real hardware.

## The Simulator: MuJoCo

MuJoCo (Multi-Joint dynamics with Contact) is a physics engine built for robotics research. It simulates rigid body dynamics, contacts, and actuator physics accurately and fast. We use it to run thousands of episodes of the robot attempting tasks, which would be impossible (and dangerous) to do on real hardware.

## The Robot: UR5e + Robotiq 2F-85

The Universal Robots UR5e is a 6-DOF collaborative robot arm widely used in research and industry. We attach a Robotiq 2F-85 two-finger parallel gripper to the end-effector, giving the arm the ability to grasp objects.

All robot models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — a collection of high-quality MuJoCo models.

## The Algorithm: SAC

We use Soft Actor-Critic (SAC), an off-policy deep RL algorithm well suited for continuous control. Key properties:

- **Off-policy**: learns from a replay buffer of past experience, making it sample efficient
- **Entropy regularization**: encourages exploration by rewarding the agent for acting randomly early on, then gradually committing to good behaviors
- **Continuous actions**: directly outputs joint velocity commands, no discretization needed

SAC is implemented via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

## Environments

### URReachEnv
The simplest task — move the end-effector to a randomly placed 3D target. The reward is the negative distance to the target. Good for verifying the training pipeline works.

### URPickPlaceEnv
A single arm must pick a red box off a table and place it in a green target zone. The reward combines:
- Distance from end-effector to object (reach)
- Distance from object to target (place)
- Height bonus for lifting

### URDualArmEnv (4-arm)
Four UR5e arms are arranged around a central table. Each arm has its own set of colored items nearby. A central orange box must be handed from one side to the other and placed in a drop zone. This tests whether agents can learn cooperative behavior across a shared workspace.

## Observation Space

Each agent observes:
- Joint positions and velocities of its arm(s)
- End-effector position(s) in world frame
- Object position
- Target/drop zone position
- Gripper state

## Action Space

Normalized joint velocity commands in [-1, 1], scaled before being sent to the simulator. One extra action per arm controls the gripper (open/close).

## Training Loop

1. Agent takes an action based on current observation
2. Simulator steps forward (5 physics steps per action)
3. Reward is computed from the new state
4. Transition is stored in a replay buffer
5. SAC samples a batch and updates the actor and critic networks
6. Repeat for 500k–1M timesteps

## Live Viewer

During training, a passive MuJoCo viewer syncs with the environment every few steps so you can watch the arms move in real time as the policy improves.

## ROS2 Integration

A ROS2 inference node (`ros2/ur_policy_node.py`) loads a trained SAC model and publishes joint trajectory commands to a real UR5e over ROS2. It subscribes to `/joint_states` and publishes to the UR's trajectory controller at 10Hz.

## Future Work

- Curriculum learning: start with easy targets, gradually increase difficulty
- Multi-task learning: single policy for reach, pick, place, handover
- Isaac Lab: GPU-accelerated parallel training (1000x faster than CPU MuJoCo)
- Sim-to-real transfer: domain randomization to close the gap between sim and real robot
