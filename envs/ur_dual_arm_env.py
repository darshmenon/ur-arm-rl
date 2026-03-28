import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces


class URDualArmEnv(gym.Env):
    """
    Two UR5e arms doing a cooperative handover task:
    - Arm 1 (left):  picks the object
    - Arm 2 (right): receives and places it at target
    Both arms are controlled simultaneously.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        spec = mujoco.MjSpec()
        spec.option.gravity = [0, 0, -9.81]

        # Skybox + ground
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

        light = spec.worldbody.add_light()
        light.pos = [0, 0, 2]

        def attach_arm(parent_spec, arm_name, pos, euler_z):
            arm_spec = mujoco.MjSpec.from_file(
                "/home/asimov/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
            )
            gripper_spec = mujoco.MjSpec.from_file(
                "/home/asimov/mujoco_menagerie/robotiq_2f85/2f85.xml"
            )
            arm_spec.sites[0].attach_body(
                gripper_spec.worldbody.first_body(), f"{arm_name}_gr-", ""
            )
            mount = parent_spec.worldbody.add_body()
            mount.name = f"{arm_name}_mount"
            mount.pos = pos
            mount.alt.euler = [0, 0, euler_z]
            frame = mount.add_frame()
            frame.attach_body(arm_spec.worldbody.first_body(), f"{arm_name}_", "")

        attach_arm(spec, "left1",  [-0.9, -0.5, 0.0],  0.0)
        attach_arm(spec, "left2",  [-0.9,  0.5, 0.0],  0.0)
        attach_arm(spec, "right1", [ 0.9, -0.5, 0.0], 180.0)
        attach_arm(spec, "right2", [ 0.9,  0.5, 0.0], 180.0)

        # Large central table between all 4 arms
        table = spec.worldbody.add_body()
        table.name = "table"
        table.pos = [0.0, 0.0, 0.0]
        tg = table.add_geom()
        tg.name = "table_geom"
        tg.type = mujoco.mjtGeom.mjGEOM_BOX
        tg.size = [0.6, 0.8, 0.02]
        tg.rgba = [0.8, 0.6, 0.4, 1]

        # Items near each arm base (2 per arm = 8 total)
        items = [
            # left1 items
            ("object_l1a", [-0.5, -0.5, 0.045], [0.9, 0.1, 0.1, 1]),
            ("object_l1b", [-0.4, -0.6, 0.045], [0.9, 0.5, 0.1, 1]),
            # left2 items
            ("object_l2a", [-0.5,  0.5, 0.045], [0.1, 0.1, 0.9, 1]),
            ("object_l2b", [-0.4,  0.6, 0.045], [0.5, 0.1, 0.9, 1]),
            # right1 items
            ("object_r1a", [ 0.5, -0.5, 0.045], [0.1, 0.8, 0.1, 1]),
            ("object_r1b", [ 0.4, -0.6, 0.045], [0.1, 0.8, 0.5, 1]),
            # right2 items
            ("object_r2a", [ 0.5,  0.5, 0.045], [0.8, 0.8, 0.1, 1]),
            ("object_r2b", [ 0.4,  0.6, 0.045], [0.8, 0.4, 0.1, 1]),
        ]
        for name, pos, color in items:
            obj = spec.worldbody.add_body()
            obj.name = name
            obj.pos = pos
            fj = obj.add_freejoint()
            fj.name = f"{name}_joint"
            og = obj.add_geom()
            og.name = f"{name}_geom"
            og.type = mujoco.mjtGeom.mjGEOM_BOX
            og.size = [0.025, 0.025, 0.025]
            og.rgba = color
            og.mass = 0.1
            og.friction = [1.5, 0.005, 0.0001]

        # Central handover object (orange box — no rolling)
        obj = spec.worldbody.add_body()
        obj.name = "object"
        obj.pos = [0.0, 0.0, 0.045]
        fj = obj.add_freejoint()
        fj.name = "object_joint"
        og = obj.add_geom()
        og.name = "object_geom"
        og.type = mujoco.mjtGeom.mjGEOM_BOX
        og.size = [0.03, 0.03, 0.03]
        og.rgba = [1.0, 0.5, 0.0, 1]
        og.mass = 0.1
        og.friction = [2.0, 0.01, 0.001]

        # Drop zones for each right arm
        for dname, dpos in [("drop1", [0.3, -0.3, 0.022]), ("drop2", [0.3, 0.3, 0.022])]:
            dz = spec.worldbody.add_body()
            dz.name = dname
            dz.pos = dpos
            dzg = dz.add_geom()
            dzg.name = f"{dname}_geom"
            dzg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            dzg.size = [0.05, 0.002, 0]
            dzg.rgba = [0.1, 0.9, 0.1, 0.4]
            dzg.contype = 0
            dzg.conaffinity = 0

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self._n_arm = 6
        self._n_grip = 1
        self._n_per_arm = self._n_arm + self._n_grip
        self._n_arms = 4
        self._n_ctrl = self._n_per_arm * self._n_arms

        self._ee_sites = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{n}_attachment_site")
            for n in ["left1", "left2", "right1", "right2"]
        ]
        self._obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._drop_pos = np.array([0.3, 0.0, 0.022])

        # obs: 4x(qpos6+qvel6) + 4x ee_pos(3) + obj(3) + drop(3) + grippers(4)
        obs_dim = self._n_arms * 12 + self._n_arms * 3 + 3 + 3 + self._n_arms
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._n_ctrl,), dtype=np.float32)

        self._viewer = None

    def _get_obs(self):
        parts = []
        for i in range(self._n_arms):
            s = i * self._n_per_arm
            parts.append(self.data.qpos[s:s+self._n_arm].astype(np.float32))
            parts.append(self.data.qvel[s:s+self._n_arm].astype(np.float32))
        for site_id in self._ee_sites:
            parts.append(self.data.site_xpos[site_id].astype(np.float32))
        parts.append(self.data.xpos[self._obj_id].astype(np.float32))
        parts.append(self._drop_pos.astype(np.float32))
        grips = np.array([
            self.data.qpos[i * self._n_per_arm + self._n_arm] for i in range(self._n_arms)
        ], dtype=np.float32)
        parts.append(grips)
        return np.concatenate(parts)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        obj_start = self._n_per_arm * self._n_arms
        self.data.qpos[obj_start:obj_start+3] = [0.0, 0.0, 0.045]
        self.data.qpos[obj_start+3:obj_start+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        for i in range(self._n_arms):
            s = i * self._n_per_arm
            self.data.ctrl[s:s+self._n_arm] = action[s:s+self._n_arm] * 0.5
            grip_idx = s + self._n_arm
            if grip_idx < self.model.nu:
                self.data.ctrl[grip_idx] = (action[grip_idx] + 1) / 2 * 0.8

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obj_pos = self.data.xpos[self._obj_id]
        ee_positions = [self.data.site_xpos[sid] for sid in self._ee_sites]
        dist_ee_obj = min(np.linalg.norm(ee - obj_pos) for ee in ee_positions)
        dist_obj_drop = np.linalg.norm(obj_pos[:2] - self._drop_pos[:2])

        reward = -0.5 * dist_ee_obj - 1.0 * dist_obj_drop
        if dist_obj_drop < 0.06 and obj_pos[2] < 0.06:
            reward += 20.0

        terminated = bool(dist_obj_drop < 0.05 and obj_pos[2] < 0.06)
        truncated = bool(self.data.time > 20.0)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {
            "dist_ee_obj": dist_ee_obj,
            "dist_obj_drop": dist_obj_drop,
        }

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
