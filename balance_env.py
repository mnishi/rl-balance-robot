import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class BalanceEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path("balance.xml")
        self.data = mujoco.MjData(self.model)

        # Body IDs
        self.ball = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        self.foot = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "foot"
        )
        self.upper_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "upper_body"
        )

        # Joint IDs
        pitch_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pitch"
        )
        roll_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "roll"
        )

        # Actuator IDs
        self.pitch_motor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor"
        )
        self.roll_motor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll_motor"
        )

        # Joint addresses
        self.pitch_qpos = self.model.jnt_qposadr[pitch_joint]
        self.roll_qpos = self.model.jnt_qposadr[roll_joint]

        self.pitch_dof = self.model.jnt_dofadr[pitch_joint]
        self.roll_dof = self.model.jnt_dofadr[roll_joint]

        # Action space: normalized pitch and roll target angles
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Observation space:
        # Upper body orientation (3)
        # Upper body angular velocity (3)
        # Joint angles (2)
        # Joint velocities (2)
        # Ball angular velocity (3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        # Simulation settings
        self.max_angle = np.deg2rad(30.0)
        self.frame_skip = 5
        self.max_steps = 2000
        self.steps = 0

        # Gravity torque settings
        self.total_mass = self.model.body_subtreemass[self.ball]
        self.gravity_force = self.total_mass * self.model.opt.gravity

    def get_obs(self):

        # Upper body orientation
        rotation = self.data.xmat[self.upper_body].reshape(3, 3)
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

        # Randomize initial joint angles
        self.data.qpos[self.pitch_qpos] = self.np_random.uniform(
            -0.03, 0.03
        )
        self.data.qpos[self.roll_qpos] = self.np_random.uniform(
            -0.03, 0.03
        )

        # Reset actuator targets
        self.data.ctrl[self.pitch_motor] = 0.0
        self.data.ctrl[self.roll_motor] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.steps = 0

        return self.get_obs(), {}

    def step(self, action):

        action = np.clip(action, -1.0, 1.0)

        # Convert normalized actions to target angles
        self.data.ctrl[self.pitch_motor] = action[0] * self.max_angle
        self.data.ctrl[self.roll_motor] = action[1] * self.max_angle

        # Advance simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1

        obs = self.get_obs()

        # Calculate system center of mass
        com_position = self.data.subtree_com[self.ball]

        # Calculate contact point
        ball_position = self.data.xpos[self.ball]

        contact_point = ball_position + np.array([
            0.0,
            0.0,
            -0.025
        ])

        # Calculate gravity torque about the contact point
        gravity_torque = np.cross(
            com_position - contact_point,
            self.gravity_force
        )

        torque_magnitude = np.linalg.norm(gravity_torque)

        # Gravity torque reward
        torque_scale = 0.01

        torque_reward = 1.0 / (
            1.0 + torque_magnitude / torque_scale
        )

        # Calculate total angular momentum
        angular_momentum = self.get_total_angular_momentum(
            contact_point
        )

        momentum_magnitude = np.linalg.norm(angular_momentum)

        # Angular momentum reward
        momentum_scale = 0.001

        momentum_reward = 1.0 / (
            1.0 + (momentum_magnitude / momentum_scale) ** 2
        )

        # Combined reward
        reward = torque_reward * (
            1.0 + 0.5 * momentum_reward
        )

        # Detect falling using foot orientation
        rotation = self.data.xmat[self.foot].reshape(3, 3)
        foot_up = rotation[:, 2]

        terminated = bool(foot_up[2] < 0.707)

        # Check episode time limit
        truncated = bool(self.steps >= self.max_steps)

        return obs, float(reward), terminated, truncated, {}

    def get_total_angular_momentum(self, contact_point):
        """Calculate total angular momentum about the contact point."""

        total_momentum = np.zeros(3, dtype=np.float64)

        for body_id in range(1, self.model.nbody):

            mass = self.model.body_mass[body_id]

            if mass <= 0.0:
                continue

            # Body center of mass
            com = self.data.xipos[body_id]

            # Spatial velocity in world coordinates
            velocity = np.zeros(6, dtype=np.float64)

            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                velocity,
                0
            )

            angular_velocity = velocity[:3]
            linear_velocity = velocity[3:]

            # Inertia tensor in world coordinates
            rotation = self.data.ximat[body_id].reshape(3, 3)

            inertia_world = (
                rotation
                @ np.diag(self.model.body_inertia[body_id])
                @ rotation.T
            )

            # Rotational angular momentum
            rotational_momentum = (
                inertia_world @ angular_velocity
            )

            # Orbital angular momentum
            r = com - contact_point

            orbital_momentum = np.cross(
                r,
                mass * linear_velocity
            )

            total_momentum += (
                rotational_momentum + orbital_momentum
            )

        return total_momentum
