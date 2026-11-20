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

        # Foot body ID
        self.foot = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "foot"
        )

        # Upper body ID
        self.upper_body = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "upper_body"
        )

        # Joint ID
        roll_joint = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "roll"
        )

        # Actuator ID
        self.roll_motor = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "roll_motor"
        )

        # Joint addresses
        self.roll_qpos = self.model.jnt_qposadr[roll_joint]
        self.roll_dof = self.model.jnt_dofadr[roll_joint]

        # Action: Roll target angle
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation:
        # COM Y offset
        # COM Y velocity
        # Roll joint angle
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

        # Simulation settings
        self.max_angle = np.deg2rad(80.0)
        self.frame_skip = 5
        self.max_steps = 2000
        self.steps = 0

        # Cylinder radius
        self.contact_radius = 0.02

        # Gravity settings
        self.total_mass = self.model.body_subtreemass[self.ball]

        self.gravity_force = (
            self.total_mass * self.model.opt.gravity
        )

        self.prev_com_offset = None
        
        # Initial ground-contact body position in world coordinates
        self.initial_ball_xy = None

        # Previous oscillation energy
        self.prev_osc_energy = None
        
        # Oscillation energy scales
        self.com_position_scale = 0.05
        self.com_velocity_scale = 0.2
        
        # Energy change reward coefficient
        self.energy_reward_weight = 0.1

    def get_contact_point(self):
        # Approximate the cylinder contact point.
        # This assumes the cylinder remains approximately horizontal.
        ball_position = self.data.xpos[self.ball]

        return ball_position + np.array([
            0.0,
            0.0,
            -self.contact_radius
        ])

    def get_obs(self):
        com_position = self.data.subtree_com[self.ball]
        contact_point = self.get_contact_point()

        # COM offset along the controlled direction
        com_offset = (
            com_position[1] - contact_point[1]
        )

        # COM offset velocity
        if self.prev_com_offset is None:
            com_velocity = 0.0
        else:
            dt = self.model.opt.timestep * self.frame_skip

            com_velocity = (
                com_offset - self.prev_com_offset
            ) / dt

        self.prev_com_offset = float(com_offset)

        # Roll joint angle
        roll = self.data.qpos[self.roll_qpos]

        obs = np.array([
            com_offset,
            com_velocity,
            roll
        ], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        # Randomize initial roll joint angle
        self.data.qpos[self.roll_qpos] = (
            self.np_random.uniform(-0.08, 0.08)
        )

        # Reset actuator target
        self.data.ctrl[self.roll_motor] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        # Save the initial XY position in world coordinates
        self.initial_ball_xy = self.data.xpos[self.ball][:2].copy()

        self.steps = 0
        self.prev_com_offset = None

        # Previous oscillation energy
        self.prev_osc_energy = None
        
        # Oscillation energy scales
        self.com_position_scale = 0.05
        self.com_velocity_scale = 0.2
        
        # Energy change reward coefficient
        self.energy_reward_weight = 0.1

        return self.get_obs(), {}

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0
        )

        # Convert normalized action to roll target angle
        self.data.ctrl[self.roll_motor] = (
            float(action[0]) * self.max_angle
        )

        # Advance simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(
                self.model,
                self.data
            )

        self.steps += 1

        obs = self.get_obs()

        com_position = self.data.subtree_com[self.ball]
        contact_point = self.get_contact_point()

        # Gravity torque around the contact point
        gravity_torque = np.cross(
            com_position - contact_point,
            self.gravity_force
        )

        # Only evaluate torque around the Roll axis
        torque_magnitude = abs(gravity_torque[0])

        torque_scale = 0.01

        torque_reward = 1.0 / (
            1.0 + torque_magnitude / torque_scale
        )

        # Total angular momentum around the contact point
        angular_momentum = (
            self.get_total_angular_momentum(contact_point)
        )

        # Only evaluate angular momentum around the Roll axis
        momentum_magnitude = abs(angular_momentum[0])

        momentum_scale = 0.001

        momentum_reward = 1.0 / (
            1.0 +
            (momentum_magnitude / momentum_scale) ** 2
        )

        # Oscillation energy based on COM offset and velocity
        com_y = float(obs[0])
        com_vy = float(obs[1])
        
        osc_energy = 0.5 * (
            (com_y / self.com_position_scale) ** 2
            + (com_vy / self.com_velocity_scale) ** 2
        )
        
        # Calculate the change in oscillation energy
        if self.prev_osc_energy is None:
            energy_change = 0.0
        else:
            energy_change = osc_energy - self.prev_osc_energy
        
        self.prev_osc_energy = osc_energy
        
        # Reward energy reduction and penalize energy increase
        energy_reward = -self.energy_reward_weight * np.clip(
            energy_change, -1.0, 1.0
        )
        
        # Combined reward
        reward = (
            torque_reward * momentum_reward
            + energy_reward
        )

        # Termination based on COM offset along Y
        com_offset = abs(
            com_position[1] - contact_point[1]
        )

        # Upper body height above the ground
        upper_height = self.data.xpos[self.upper_body][2]
        
        # Foot tilt angle
        foot_rotation = self.data.xmat[self.foot].reshape(3, 3)
        foot_up = foot_rotation[2, 2]
        
        # Calculate horizontal displacement from the initial position
        ball_xy = self.data.xpos[self.ball][:2]
        
        travel_distance = np.linalg.norm(
            ball_xy - self.initial_ball_xy
        )
        
        # Terminate if:
        # - COM offset exceeds 5 cm
        # - Foot tilts more than 80 degrees
        # - Upper body height falls to 5 cm or below
        # - Robot moves more than 3 cm from its initial position
        terminated = bool(
            com_offset > 0.05
            or foot_up < np.cos(np.deg2rad(80.0))
            or upper_height <= 0.05
            or travel_distance > 0.1
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

    def get_total_angular_momentum(self, contact_point):
        total_momentum = np.zeros(
            3,
            dtype=np.float64
        )

        for body_id in range(1, self.model.nbody):
            mass = self.model.body_mass[body_id]

            if mass <= 0.0:
                continue

            com = self.data.xipos[body_id]

            velocity = np.zeros(
                6,
                dtype=np.float64
            )

            # Get world-frame spatial velocity
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

            # Convert velocity at the body origin to COM velocity
            body_origin = self.data.xpos[body_id]

            com_velocity = (
                linear_velocity +
                np.cross(
                    angular_velocity,
                    com - body_origin
                )
            )

            # Body inertia in world coordinates
            rotation = (
                self.data.ximat[body_id].reshape(3, 3)
            )

            inertia_world = (
                rotation
                @ np.diag(self.model.body_inertia[body_id])
                @ rotation.T
            )

            rotational_momentum = (
                inertia_world @ angular_velocity
            )

            # Orbital angular momentum
            r = com - contact_point

            orbital_momentum = np.cross(
                r,
                mass * com_velocity
            )

            total_momentum += (
                rotational_momentum +
                orbital_momentum
            )

        return total_momentum
