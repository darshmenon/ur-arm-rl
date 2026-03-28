import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

TABLE_Z = 0.02   # table surface z
LIFT_Z  = 0.10   # z threshold to count as "lifted"

# Each arm: name, base pos, euler_z, object start pos, object color, drop zone pos
ARM_CFG = [
    ("left1",  [-0.9, -0.5, 0.0],   0.0, [-0.45, -0.45, 0.045], [0.9, 0.1, 0.1, 1], [-0.2, -0.35, TABLE_Z]),
    ("left2",  [-0.9,  0.5, 0.0],   0.0, [-0.45,  0.45, 0.045], [0.1, 0.1, 0.9, 1], [-0.2,  0.35, TABLE_Z]),
    ("right1", [ 0.9, -0.5, 0.0], 180.0, [ 0.45, -0.45, 0.045], [0.1, 0.8, 0.1, 1], [ 0.2, -0.35, TABLE_Z]),
    ("right2", [ 0.9,  0.5, 0.0], 180.0, [ 0.45,  0.45, 0.045], [0.9, 0.8, 0.1, 1], [ 0.2,  0.35, TABLE_Z]),
]


class URDualArmEnv(gym.Env):
    """
    4x UR5e arms each assigned their own object to pick and place.
    Staged reward: reach -> grasp -> lift -> place.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        spec = mujoco.MjSpec()
        spec.option.gravity = [0, 0, -9.81]

        tex = spec.add_texture()
        tex.name = "groundplane"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        tex.rgb1 = [0.2, 0.3, 0.4]
        tex.rgb2 = [0.1, 0.2, 0.3]
        tex.width = 300
        tex.height = 300

        mat = spec.add_material()
        mat.name = "groundplane"
        mat.textures = ["groundplane"] + [""] * 9
        mat.texuniform = True

        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0, 0, 0.05]
        floor.material = "groundplane"

        spec.worldbody.add_light().pos = [0, 0, 2.5]

        def attach_arm(arm_name, pos, euler_z):
            arm_spec = mujoco.MjSpec.from_file(
                "/home/asimov/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
            )
            grip_spec = mujoco.MjSpec.from_file(
                "/home/asimov/mujoco_menagerie/robotiq_2f85/2f85.xml"
            )
            arm_spec.sites[0].attach_body(grip_spec.worldbody.first_body(), f"{arm_name}_gr-", "")
            mount = spec.worldbody.add_body()
            mount.name = f"{arm_name}_mount"
            mount.pos = pos
            mount.alt.euler = [0, 0, euler_z]
            base = mount.add_geom()
            base.name = f"{arm_name}_base"
            base.type = mujoco.mjtGeom.mjGEOM_BOX
            base.size = [0.1, 0.1, 0.15]
            base.pos = [0, 0, -0.15]
            base.rgba = [0.2, 0.2, 0.2, 1]
            frame = mount.add_frame()
            frame.attach_body(arm_spec.worldbody.first_body(), f"{arm_name}_", "")

        for arm_name, pos, euler_z, _, _, _ in ARM_CFG:
            attach_arm(arm_name, pos, euler_z)

        # Central table
        table = spec.worldbody.add_body()
        table.name = "table"
        table.pos = [0.0, 0.0, 0.0]
        tg = table.add_geom()
        tg.name = "table_geom"
        tg.type = mujoco.mjtGeom.mjGEOM_BOX
        tg.size = [0.6, 0.8, TABLE_Z]
        tg.rgba = [0.8, 0.6, 0.4, 1]

        # Extra static obstacle boxes in center of table
        extra_boxes = [
            ([0.0,  0.0,  TABLE_Z+0.025], [0.6, 0.3, 0.3, 1]),
            ([0.1,  0.2,  TABLE_Z+0.025], [0.3, 0.6, 0.3, 1]),
            ([-0.1, -0.2, TABLE_Z+0.025], [0.3, 0.3, 0.6, 1]),
            ([0.0, -0.15, TABLE_Z+0.025], [0.6, 0.6, 0.2, 1]),
            ([0.15, 0.0,  TABLE_Z+0.025], [0.2, 0.6, 0.6, 1]),
            ([-0.15, 0.1, TABLE_Z+0.025], [0.6, 0.2, 0.6, 1]),
        ]
        for ei, (epos, ecol) in enumerate(extra_boxes):
            eb = spec.worldbody.add_body()
            eb.name = f"extra_{ei}"
            eb.pos = epos
            efj = eb.add_freejoint()
            efj.name = f"extra_{ei}_joint"
            eg = eb.add_geom()
            eg.name = f"extra_{ei}_geom"
            eg.type = mujoco.mjtGeom.mjGEOM_BOX
            eg.size = [0.02, 0.02, 0.02]
            eg.rgba = ecol
            eg.mass = 0.05
            eg.friction = [2.0, 0.01, 0.001]

        # One object per arm + one drop zone per arm
        for arm_name, _, _, obj_pos, obj_color, drop_pos in ARM_CFG:
            obj = spec.worldbody.add_body()
            obj.name = f"obj_{arm_name}"
            obj.pos = obj_pos
            fj = obj.add_freejoint()
            fj.name = f"obj_{arm_name}_joint"
            og = obj.add_geom()
            og.name = f"obj_{arm_name}_geom"
            og.type = mujoco.mjtGeom.mjGEOM_BOX
            og.size = [0.025, 0.025, 0.025]
            og.rgba = obj_color
            og.mass = 0.1
            og.friction = [2.0, 0.01, 0.001]

            dz = spec.worldbody.add_body()
            dz.name = f"drop_{arm_name}"
            dz.pos = drop_pos
            dzg = dz.add_geom()
            dzg.name = f"drop_{arm_name}_geom"
            dzg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            dzg.size = [0.055, 0.002, 0]
            dzg.rgba = [0.1, 0.9, 0.1, 0.5]
            dzg.contype = 0
            dzg.conaffinity = 0

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self._n_arm   = 6
        self._n_arms  = 4
        self._n_ctrl  = 7 * self._n_arms  # 6 arm + 1 gripper actuator per arm (nu=28)

        arm_names = [c[0] for c in ARM_CFG]

        # Get exact qpos address for each arm's shoulder_pan joint
        self._arm_qpos_adr = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_shoulder_pan_joint")
            ]
            for n in arm_names
        ]
        # Get exact qvel address for each arm's shoulder_pan joint
        self._arm_qvel_adr = [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_shoulder_pan_joint")
            ]
            for n in arm_names
        ]
        # Gripper finger qpos (driver joint)
        self._grip_qpos_adr = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_{n}_gr-right_driver_joint")
            ]
            for n in arm_names
        ]

        self._ee_sites = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{n}_attachment_site")
            for n in arm_names
        ]
        self._obj_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{n}")
            for n in arm_names
        ]
        self._drop_positions = [np.array(c[5]) for c in ARM_CFG]
        self._obj_init_pos   = [np.array(c[3]) for c in ARM_CFG]

        # Find qpos start of first arm object (after all joints)
        self._obj_qpos_adr = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{n}_joint")
            ]
            for n in arm_names
        ]

        # obs per arm: qpos(6) + qvel(6) + ee(3) + obj(3) + drop(3) + gripper(1) = 22
        obs_dim = self._n_arms * 22
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._n_ctrl,), dtype=np.float32)
        self._viewer = None

    def _get_obs(self):
        parts = []
        for i in range(self._n_arms):
            qa = self._arm_qpos_adr[i]
            va = self._arm_qvel_adr[i]
            parts.append(self.data.qpos[qa:qa+self._n_arm].astype(np.float32))
            parts.append(self.data.qvel[va:va+self._n_arm].astype(np.float32))
            parts.append(self.data.site_xpos[self._ee_sites[i]].astype(np.float32))
            parts.append(self.data.xpos[self._obj_ids[i]].astype(np.float32))
            parts.append(self._drop_positions[i].astype(np.float32))
            parts.append(np.array([self.data.qpos[self._grip_qpos_adr[i]]], dtype=np.float32))
        return np.concatenate(parts)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Set arms to a "ready" pose — elbow bent forward, EE pointing down toward table
        # UR5e joints: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        # Left arms (euler_z=0) need pan=pi to face +x (toward table)
        # Right arms (euler_z=180) need pan=0 to face -x (toward table)
        ready_poses = [
            np.array([np.pi, -1.0, 1.5, -1.57, -1.57, 0.0]),  # left1
            np.array([np.pi, -1.0, 1.5, -1.57, -1.57, 0.0]),  # left2
            np.array([0.0,   -1.0, 1.5, -1.57, -1.57, 0.0]),  # right1
            np.array([0.0,   -1.0, 1.5, -1.57, -1.57, 0.0]),  # right2
        ]
        for i in range(self._n_arms):
            qa = self._arm_qpos_adr[i]
            self.data.qpos[qa:qa + self._n_arm] = ready_poses[i]

        # Reset each object using its exact qpos address
        for i, init_pos in enumerate(self._obj_init_pos):
            base = self._obj_qpos_adr[i]
            self.data.qpos[base:base+3] = init_pos
            self.data.qpos[base+3:base+7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # Apply arm + gripper controls (ctrl layout: 7 per arm = 6 arm + 1 gripper)
        for i in range(self._n_arms):
            s = i * 7
            self.data.ctrl[s:s+self._n_arm] = action[s:s+self._n_arm] * 0.5
            grip_idx = s + self._n_arm
            if grip_idx < self.model.nu:
                self.data.ctrl[grip_idx] = (action[grip_idx] + 1) / 2 * 0.8

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        reward = 0.0
        info = {}
        all_placed = True

        for i in range(self._n_arms):
            arm_name = ARM_CFG[i][0]
            ee       = self.data.site_xpos[self._ee_sites[i]]
            obj      = self.data.xpos[self._obj_ids[i]]
            drop     = self._drop_positions[i]
            grip_q   = self.data.qpos[self._grip_qpos_adr[i]]

            dist_ee_obj  = float(np.linalg.norm(ee - obj))
            dist_obj_drop = float(np.linalg.norm(obj[:2] - drop[:2]))
            obj_height   = float(obj[2])
            lifted       = obj_height > LIFT_Z

            # --- Staged reward ---
            # 1. Reach: always active, pull EE toward object
            reward -= 0.3 * dist_ee_obj

            # 2. Grasp: bonus when EE is close AND gripper is closing
            if dist_ee_obj < 0.10 and grip_q > 0.15:
                reward += 0.5

            # 3. Lift: reward proportional to height above table
            lift_bonus = max(0.0, obj_height - (TABLE_Z + 0.03))
            reward += 3.0 * lift_bonus

            # 4. Place: only penalise drop distance once object is lifted
            if lifted:
                reward -= 0.8 * dist_obj_drop

            # 5. Success per arm
            placed = dist_obj_drop < 0.06 and obj_height < TABLE_Z + 0.05
            if placed:
                reward += 5.0
            else:
                all_placed = False

            info[f"{arm_name}_dist_ee"]   = dist_ee_obj
            info[f"{arm_name}_dist_drop"] = dist_obj_drop
            info[f"{arm_name}_lifted"]    = lifted
            info[f"{arm_name}_placed"]    = placed

        terminated = all_placed
        truncated  = self.data.time > 20.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
