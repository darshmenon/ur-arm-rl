"""
ROS2 node that loads a trained SAC policy and publishes UR5e joint commands.

Subscribes:
  /joint_states  (sensor_msgs/JointState)  — current arm state

Publishes:
  /scaled_joint_trajectory_controller/joint_trajectory
  (trajectory_msgs/JointTrajectory)        — target joint positions
"""

import sys
sys.path.insert(0, "/home/asimov/mujoco-ur-arm-rl")

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from stable_baselines3 import SAC

UR5E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class URPolicyNode(Node):
    def __init__(self, model_path: str, target: list):
        super().__init__("ur_policy_node")

        self.target = np.array(target, dtype=np.float32)
        self.qpos = np.zeros(6, dtype=np.float32)
        self.qvel = np.zeros(6, dtype=np.float32)
        self._prev_pos = np.zeros(6, dtype=np.float32)
        self._prev_time = None

        self.get_logger().info(f"Loading model from {model_path}")
        self.model = SAC.load(model_path)
        self.get_logger().info("Model loaded.")

        self.sub = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10
        )
        self.pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10,
        )
        self.timer = self.create_timer(0.1, self._step)  # 10 Hz

    def _joint_cb(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        now = self.get_clock().now().nanoseconds * 1e-9

        for i, jname in enumerate(UR5E_JOINTS):
            if jname in name_to_idx:
                idx = name_to_idx[jname]
                self.qpos[i] = msg.position[idx]
                if self._prev_time is not None and (now - self._prev_time) > 0:
                    self.qvel[i] = (msg.position[idx] - self._prev_pos[i]) / (now - self._prev_time)
                self._prev_pos[i] = msg.position[idx]

        self._prev_time = now

    def _step(self):
        obs = np.concatenate([self.qpos, self.qvel, self.target]).astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)

        # Convert velocity action to position delta
        dt = 0.1
        target_pos = self.qpos + np.clip(action, -1, 1) * 0.5 * dt

        msg = JointTrajectory()
        msg.joint_names = UR5E_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = target_pos.tolist()
        pt.time_from_start = Duration(sec=0, nanosec=100_000_000)
        msg.points = [pt]
        self.pub.publish(msg)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/asimov/mujoco-ur-arm-rl/models/best_model.zip")
    parser.add_argument("--target", nargs=3, type=float, default=[0.4, 0.0, 0.4],
                        metavar=("X", "Y", "Z"), help="Target end-effector position")
    args = parser.parse_args(sys.argv[1:])

    rclpy.init()
    node = URPolicyNode(args.model, args.target)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
