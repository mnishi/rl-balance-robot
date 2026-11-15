from collections import deque

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

        # Joint movement penalty settings
        self.movement_window = 2000
        self.movement_weight = 0.01

        self.angle_changes = deque(maxlen=self.movement_window)
        self.angle_change_sum = 0.0

        self.prev_pitch = 0.0
        self.prev_roll = 0.0

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

        # Reset joint movement history
        self.angle_changes.clear()
        self.angle_change_sum = 0.0

        self.prev_pitch = float(self.data.qpos[self.pitch_qpos])
        self.prev_roll = float(self.data.qpos[self.roll_qpos])

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

        # Calculate gravitational torque around the contact point
        com_position = self.data.subtree_com[self.ball]
        ball_position = self.data.xpos[self.ball]

        contact_point = ball_position + np.array([0.0, 0.0, -0.025])

        gravity_torque = np.cross(
            com_position - contact_point,
            self.gravity_force
        )

        # Calculate gravity torque reward
        torque_error = np.sum(gravity_torque ** 2) / (0.01 ** 2)
        torque_reward = 1.0 / (1.0 + torque_error)

        # Get actual joint angles
        pitch = float(self.data.qpos[self.pitch_qpos])
        roll = float(self.data.qpos[self.roll_qpos])

        # Calculate actual joint movement
        angle_change = (
            abs(pitch - self.prev_pitch)
            + abs(roll - self.prev_roll)
        )

        # Update moving average using a running sum
        if len(self.angle_changes) == self.movement_window:
            self.angle_change_sum -= self.angle_changes.popleft()

        self.angle_changes.append(angle_change)
        self.angle_change_sum += angle_change

        mean_angle_change = (
            self.angle_change_sum / len(self.angle_changes)
        )

        # Calculate joint movement penalty
        movement_penalty = self.movement_weight * mean_angle_change

        # Calculate total reward
        reward = torque_reward - movement_penalty

        # Update previous joint angles
        self.prev_pitch = pitch
        self.prev_roll = roll

        # Detect falling using foot orientation
        rotation = self.data.xmat[self.foot].reshape(3, 3)
        foot_up = rotation[:, 2]

        terminated = bool(foot_up[2] < 0.707)

        # Check episode time limit
        truncated = bool(self.steps >= self.max_steps)

        return obs, float(reward), terminated, truncated, {}
