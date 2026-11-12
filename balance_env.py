import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BalanceEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path("balance.xml")
        self.data = mujoco.MjData(self.model)

        self.leg = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "foot"
        )
        # Joint IDs
        self.pitch_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pitch"
        )
        self.roll_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "roll"
        )

        # Actuator IDs
        self.pitch_motor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor"
        )
        self.roll_motor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll_motor"
        )

        # Upper body ID
        self.upper_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "upper_body"
        )

        # Joint addresses
        self.pitch_qpos = self.model.jnt_qposadr[self.pitch_joint]
        self.roll_qpos = self.model.jnt_qposadr[self.roll_joint]

        self.pitch_dof = self.model.jnt_dofadr[self.pitch_joint]
        self.roll_dof = self.model.jnt_dofadr[self.roll_joint]

        # Action: Pitch / Roll target angles
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Observation:
        # body up vector (3)
        # body angular velocity (3)
        # joint angles (2)
        # joint velocities (2)
        # ball angular velocity (3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        self.max_angle = 0.785  # 45 degrees
        self.frame_skip = 10
        self.max_steps = 2000
        self.steps = 0

    def get_obs(self):

        # Upper body orientation
        rotation = self.data.xmat[self.upper_body].reshape(3, 3)

        # Upper body's local Z axis in world coordinates
        up = rotation[:, 2]

        # Upper body angular velocity
        angular_velocity = self.data.cvel[self.upper_body][:3]

        # Joint states
        pitch = self.data.qpos[self.pitch_qpos]
        roll = self.data.qpos[self.roll_qpos]

        pitch_vel = self.data.qvel[self.pitch_dof]
        roll_vel = self.data.qvel[self.roll_dof]

        # Ball angular velocity
        ball_angular_velocity = self.data.qvel[3:6]

        obs = np.concatenate([
            up,
            angular_velocity,
            [pitch, roll],
            [pitch_vel, roll_vel],
            ball_angular_velocity
        ])

        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Small random initial joint angles
        self.data.qpos[self.pitch_qpos] = self.np_random.uniform(
            -0.03, 0.03
        )
        self.data.qpos[self.roll_qpos] = self.np_random.uniform(
            -0.03, 0.03
        )

        self.data.ctrl[self.pitch_motor] = 0.0
        self.data.ctrl[self.roll_motor] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.steps = 0

        return self.get_obs(), {}

    def step(self, action):

        action = np.clip(action, -1.0, 1.0)

        # Convert normalized action to target angles
        self.data.ctrl[self.pitch_motor] = action[0] * self.max_angle
        self.data.ctrl[self.roll_motor] = action[1] * self.max_angle

        # Advance simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1

        obs = self.get_obs()

        # Upright score
        # upright = obs[2]
        rotation = self.data.xmat[self.leg].reshape(3, 3)
        leg_up = rotation[:, 2]
        upright = leg_up[2]

        # Angular velocity penalty
        angular_penalty = np.sum(obs[3:6] ** 2)

        # Control effort penalty
        action_penalty = np.sum(action ** 2)

        reward = (
            2.0 * upright
            - 0.05 * angular_penalty
            - 0.01 * action_penalty
        )
        #reward = 1.0

        # Fall detection
        terminated = bool(upright < 0.707)
        #if terminated:
        #  reward = -10.0

        # Time limit
        truncated = bool(self.steps >= self.max_steps)

        return obs, float(reward), terminated, truncated, {}

