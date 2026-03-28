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

        attach_arm(spec, "left",  [-0.35, 0.0, 0.0],  0.0)
        attach_arm(spec, "right", [ 0.35, 0.0, 0.0], 180.0)

        # Object (handover target)
        obj = spec.worldbody.add_body()
        obj.name = "object"
        obj.pos = [-0.3, 0.0, 0.3]
        fj = obj.add_freejoint()
        fj.name = "object_joint"
        og = obj.add_geom()
        og.name = "object_geom"
        og.type = mujoco.mjtGeom.mjGEOM_SPHERE
        og.size = [0.03, 0, 0]
        og.rgba = [1.0, 0.5, 0.0, 1]
        og.mass = 0.1
        og.friction = [1.5, 0.005, 0.0001]

        # Drop zone (right side)
        dz = spec.worldbody.add_body()
        dz.name = "drop_zone"
        dz.pos = [0.3, 0.0, 0.02]
        dzg = dz.add_geom()
        dzg.name = "drop_zone_geom"
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
        self._n_ctrl = self._n_per_arm * 2  # both arms

        self._left_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_attachment_site")
        self._right_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_attachment_site")
        self._obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._drop_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drop_zone")
        self._drop_pos = np.array([0.3, 0.0, 0.02])

        # obs: left_qpos(6)+qvel(6) + right_qpos(6)+qvel(6) + left_ee(3) + right_ee(3) + obj(3) + drop(3) + grippers(2)
        obs_dim = 6+6 + 6+6 + 3+3+3+3 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._n_ctrl,), dtype=np.float32)

        self._viewer = None

    def _get_obs(self):
        lq = self.data.qpos[:self._n_arm].astype(np.float32)
        lv = self.data.qvel[:self._n_arm].astype(np.float32)
        rq = self.data.qpos[self._n_per_arm:self._n_per_arm + self._n_arm].astype(np.float32)
        rv = self.data.qvel[self._n_per_arm:self._n_per_arm + self._n_arm].astype(np.float32)
        lee = self.data.site_xpos[self._left_ee].astype(np.float32)
        ree = self.data.site_xpos[self._right_ee].astype(np.float32)
        obj = self.data.xpos[self._obj_id].astype(np.float32)
        drop = self._drop_pos.astype(np.float32)
        grips = np.array([
            self.data.qpos[self._n_arm],
            self.data.qpos[self._n_per_arm + self._n_arm],
        ], dtype=np.float32)
        return np.concatenate([lq, lv, rq, rv, lee, ree, obj, drop, grips])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Place object near left arm ee
        obj_start = self._n_per_arm * 2
        self.data.qpos[obj_start:obj_start+3] = [-0.3, 0.0, 0.3]
        self.data.qpos[obj_start+3:obj_start+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        left_arm  = action[:self._n_arm] * 0.5
        left_grip = (action[self._n_arm] + 1) / 2 * 0.8
        right_arm  = action[self._n_per_arm:self._n_per_arm + self._n_arm] * 0.5
        right_grip = (action[self._n_per_arm + self._n_arm] + 1) / 2 * 0.8

        self.data.ctrl[:self._n_arm] = left_arm
        self.data.ctrl[self._n_arm] = left_grip
        self.data.ctrl[self._n_per_arm:self._n_per_arm + self._n_arm] = right_arm
        self.data.ctrl[self._n_per_arm + self._n_arm] = right_grip

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obj_pos = self.data.xpos[self._obj_id]
        lee = self.data.site_xpos[self._left_ee]
        ree = self.data.site_xpos[self._right_ee]

        dist_l_obj = np.linalg.norm(lee - obj_pos)
        dist_r_obj = np.linalg.norm(ree - obj_pos)
        dist_obj_drop = np.linalg.norm(obj_pos[:2] - self._drop_pos[:2])

        reward = -0.5 * dist_l_obj - 0.5 * dist_r_obj - 1.0 * dist_obj_drop
        if dist_obj_drop < 0.06 and obj_pos[2] < 0.06:
            reward += 20.0

        terminated = bool(dist_obj_drop < 0.05 and obj_pos[2] < 0.06)
        truncated = bool(self.data.time > 20.0)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {
            "dist_l_obj": dist_l_obj,
            "dist_r_obj": dist_r_obj,
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
