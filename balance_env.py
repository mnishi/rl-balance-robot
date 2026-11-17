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
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "ball"
        )

        self.foot = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "foot"
        )

        self.upper_body = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "upper_body"
        )

        # Joint IDs
        pitch_joint = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "pitch"
        )

        roll_joint = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "roll"
        )

        # Actuator IDs
        self.pitch_motor = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "pitch_motor"
        )

        self.roll_motor = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "roll_motor"
        )

        # Joint addresses
        self.pitch_qpos = self.model.jnt_qposadr[pitch_joint]
        self.roll_qpos = self.model.jnt_qposadr[roll_joint]

        self.pitch_dof = self.model.jnt_dofadr[pitch_joint]
        self.roll_dof = self.model.jnt_dofadr[roll_joint]

        # Action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Observation:
        # COM x
        # COM y
        # COM vx
        # COM vy
        # pitch
        # roll
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        # Simulation settings
        self.max_angle = np.deg2rad(45.0)
        self.frame_skip = 5
        self.max_steps = 2000
        self.steps = 0

        # Gravity torque settings
        self.total_mass = self.model.body_subtreemass[self.ball]
        self.gravity_force = (
            self.total_mass * self.model.opt.gravity
        )

    def get_obs(self):
        com_position = self.data.subtree_com[self.ball]

        ball_position = self.data.xpos[self.ball]

        contact_point = ball_position + np.array([
            0.0,
            0.0,
            -0.02
        ])

        com_offset = (
            com_position[:2]
            - contact_point[:2]
        )

        # COM velocity relative to the ball center
        if self.prev_com_offset is None:
            com_velocity = np.zeros(2)
        else:
            dt = self.model.opt.timestep * self.frame_skip

            com_velocity = (
                com_offset - self.prev_com_offset
            ) / dt

        self.prev_com_offset = com_offset.copy()    

        pitch = self.data.qpos[self.pitch_qpos]
        roll = self.data.qpos[self.roll_qpos]

        obs = np.array([
            com_offset[0],
            com_offset[1],
            com_velocity[0],
            com_velocity[1],
            pitch,
            roll,
        ], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        # Randomize initial joint angles
        self.data.qpos[self.pitch_qpos] = (
            self.np_random.uniform(
                -0.08,
                0.08
            )
        )

        self.data.qpos[self.roll_qpos] = (
            self.np_random.uniform(
                -0.08,
                0.08
            )
        )

        # Reset actuator targets
        self.data.ctrl[self.pitch_motor] = 0.0
        self.data.ctrl[self.roll_motor] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.steps = 0

        # Initialize previous COM offset
        self.prev_com_offset = None

        return self.get_obs(), {}

    def step(self, action):
        action = np.clip(
            action,
            -1.0,
            1.0
        )

        self.data.ctrl[self.pitch_motor] = (
            action[0] * self.max_angle
        )

        self.data.ctrl[self.roll_motor] = (
            action[1] * self.max_angle
        )

        for _ in range(self.frame_skip):
            mujoco.mj_step(
                self.model,
                self.data
            )

        self.steps += 1

        obs = self.get_obs()

        com_position = self.data.subtree_com[self.ball]

        ball_position = self.data.xpos[self.ball]

        contact_point = ball_position + np.array([
            0.0,
            0.0,
            -0.02
        ])

        gravity_torque = np.cross(
            com_position - contact_point,
            self.gravity_force
        )

        torque_magnitude = np.linalg.norm(
            gravity_torque
        )

        torque_scale = 0.01

        torque_reward = 1.0 / (
            1.0 +
            torque_magnitude / torque_scale
        )

        angular_momentum = (
            self.get_total_angular_momentum(
                contact_point
            )
        )

        momentum_magnitude = np.linalg.norm(
            angular_momentum
        )

        momentum_scale = 0.001

        momentum_reward = 1.0 / (
            1.0 +
            (momentum_magnitude /
             momentum_scale) ** 2
        )

        #reward = torque_reward * (
        #    1.0 +
        #    0.5 * momentum_reward
        #)

        reward = torque_reward * momentum_reward
        
        com_offset = np.linalg.norm(
            com_position[:2]
            - contact_point[:2]
        )

        terminated = bool(
            com_offset > 0.1
        )

        truncated = bool(
            self.steps >= self.max_steps
        )

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            {}
        )

    def get_total_angular_momentum(
        self,
        contact_point
    ):
        total_momentum = np.zeros(
            3,
            dtype=np.float64
        )

        for body_id in range(
            1,
            self.model.nbody
        ):
            mass = self.model.body_mass[
                body_id
            ]

            if mass <= 0.0:
                continue

            com = self.data.xipos[
                body_id
            ]

            velocity = np.zeros(
                6,
                dtype=np.float64
            )

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

            rotation = (
                self.data.ximat[body_id]
                .reshape(3, 3)
            )

            inertia_world = (
                rotation
                @ np.diag(
                    self.model.body_inertia[
                        body_id
                    ]
                )
                @ rotation.T
            )

            rotational_momentum = (
                inertia_world
                @ angular_velocity
            )

            r = com - contact_point

            orbital_momentum = np.cross(
                r,
                mass * linear_velocity
            )

            total_momentum += (
                rotational_momentum
                + orbital_momentum
            )

        return total_momentum
