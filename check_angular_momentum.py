import argparse

import numpy as np
import mujoco
from stable_baselines3 import PPO

from balance_env import BalanceEnv


def calculate_angular_momentum(model, data, contact_point):
    """Calculate total angular momentum about the contact point."""

    total_momentum = np.zeros(3, dtype=np.float64)

    for body_id in range(1, model.nbody):

        mass = model.body_mass[body_id]

        if mass <= 0.0:
            continue

        # Body center of mass in world coordinates
        com = data.xipos[body_id]

        # Spatial velocity in world coordinates at body COM
        velocity = np.zeros(6, dtype=np.float64)

        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0
        )

        angular_velocity = velocity[:3]
        linear_velocity = velocity[3:]

        # Transform inertia tensor into world coordinates
        rotation = data.ximat[body_id].reshape(3, 3)

        inertia_world = (
            rotation
            @ np.diag(model.body_inertia[body_id])
            @ rotation.T
        )

        # Rotational angular momentum
        rotational_momentum = (
            inertia_world @ angular_velocity
        )

        # Orbital angular momentum about the contact point
        r = com - contact_point

        orbital_momentum = np.cross(
            r,
            mass * linear_velocity
        )

        total_momentum += (
            rotational_momentum + orbital_momentum
        )

    return total_momentum


def calculate_gravity_torque(model, data, ball_id, contact_point):
    """Calculate gravity torque about the contact point."""

    # Center of mass of the entire robot
    com = data.subtree_com[ball_id]

    # Total mass of the robot
    total_mass = model.body_subtreemass[ball_id]

    gravity_force = total_mass * model.opt.gravity

    gravity_torque = np.cross(
        com - contact_point,
        gravity_force
    )

    return gravity_torque, com


def calculate_com_tilt(com, contact_point):
    """Calculate COM tilt angle relative to vertical."""

    com_vector = com - contact_point

    horizontal_distance = np.linalg.norm(
        com_vector[:2]
    )

    vertical_distance = com_vector[2]

    tilt_rad = np.arctan2(
        horizontal_distance,
        vertical_distance
    )

    return np.degrees(tilt_rad)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="balance_ppo.zip",
        help="Path to trained PPO model"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Maximum evaluation steps"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes"
    )

    args = parser.parse_args()

    # Initialize environment
    env = BalanceEnv()

    # Load trained PPO model
    agent = PPO.load(args.model)

    # Get ball body ID
    ball_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "ball"
    )

    if ball_id < 0:
        raise RuntimeError("Body 'ball' not found")

    # Ball radius
    ball_radius = 0.025

    # CSV header
    print(
        "Episode,Step,"
        "GravityTorque,"
        "AngularMomentum,"
        "COMTiltDeg"
    )

    try:

        for episode in range(1, args.episodes + 1):

            obs, info = env.reset()

            for step in range(args.steps):

                # Predict action
                action, _ = agent.predict(
                    obs,
                    deterministic=True
                )

                # Execute simulation step
                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                # Calculate sphere-ground contact point
                ball_position = env.data.xpos[ball_id]

                contact_point = (
                    ball_position
                    + np.array([0.0, 0.0, -ball_radius])
                )

                # Calculate gravity torque
                gravity_torque, com = (
                    calculate_gravity_torque(
                        env.model,
                        env.data,
                        ball_id,
                        contact_point
                    )
                )

                torque_magnitude = np.linalg.norm(
                    gravity_torque
                )

                # Calculate total angular momentum
                angular_momentum = (
                    calculate_angular_momentum(
                        env.model,
                        env.data,
                        contact_point
                    )
                )

                momentum_magnitude = np.linalg.norm(
                    angular_momentum
                )

                # Calculate COM tilt angle
                com_tilt_deg = calculate_com_tilt(
                    com,
                    contact_point
                )

                # Output CSV data
                print(
                    f"{episode},"
                    f"{step},"
                    f"{torque_magnitude:.6f},"
                    f"{momentum_magnitude:.6f},"
                    f"{com_tilt_deg:.3f}"
                )

                if terminated or truncated:
                    break

    finally:
        env.close()


if __name__ == "__main__":
    main()
