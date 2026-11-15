import argparse
import csv

import numpy as np
from stable_baselines3 import PPO

from balance_env import BalanceEnv


def calculate_reward_components(env):
    """Calculate gravity torque and angular momentum rewards."""

    # Calculate system center of mass
    com_position = env.data.subtree_com[env.ball]

    # Calculate contact point
    ball_position = env.data.xpos[env.ball]

    contact_point = ball_position + np.array([
        0.0, 0.0, -0.025
    ])

    # Calculate gravity torque
    gravity_torque = np.cross(
        com_position - contact_point,
        env.gravity_force
    )

    torque = np.linalg.norm(gravity_torque)

    # Gravity torque reward
    torque_reward = 1.0 / (
        1.0 + torque / 0.01
    )

    # Calculate total angular momentum
    angular_momentum = env.get_total_angular_momentum(
        contact_point
    )

    momentum = np.linalg.norm(angular_momentum)

    # Angular momentum reward
    momentum_reward = 1.0 / (
        1.0 + (momentum / 0.001) ** 2
    )

    # Calculate COM tilt angle
    com_vector = com_position - contact_point

    horizontal_distance = np.linalg.norm(
        com_vector[:2]
    )

    com_tilt_deg = np.degrees(
        np.arctan2(horizontal_distance, com_vector[2])
    )

    return (
        torque,
        torque_reward,
        momentum,
        momentum_reward,
        com_tilt_deg
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        nargs="?",
        default="balance_ppo.zip",
        help="PPO model file"
    )

    args = parser.parse_args()

    env = BalanceEnv()
    model = PPO.load(args.model)

    print(f"Loading model: {args.model}")

    results = []

    # Open CSV file once for all episodes
    with open("evaluation.csv", "w", newline="") as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "episode",
            "step",
            "pitch",
            "roll",
            "pitch_target",
            "roll_target",
            "pitch_vel",
            "roll_vel",
            "pitch_torque",
            "roll_torque",
            "foot_up",
            "upper_up",
            "gravity_torque",
            "torque_reward",
            "angular_momentum",
            "momentum_reward",
            "com_tilt_deg",
            "reward"
        ])

        for episode in range(10):

            obs, info = env.reset()
            total_reward = 0.0

            for step in range(env.max_steps):

                # Predict action
                action, _ = model.predict(
                    obs,
                    deterministic=True
                )

                # Execute simulation
                obs, reward, terminated, truncated, info = env.step(
                    action
                )

                # Get joint states
                pitch = float(env.data.qpos[env.pitch_qpos])
                roll = float(env.data.qpos[env.roll_qpos])

                pitch_vel = float(env.data.qvel[env.pitch_dof])
                roll_vel = float(env.data.qvel[env.roll_dof])

                # Get target angles
                pitch_target = float(action[0] * env.max_angle)
                roll_target = float(action[1] * env.max_angle)

                # Get actuator torques
                pitch_torque = float(
                    env.data.actuator_force[env.pitch_motor]
                )

                roll_torque = float(
                    env.data.actuator_force[env.roll_motor]
                )

                # Get body orientations
                foot_rotation = env.data.xmat[env.foot].reshape(3, 3)
                upper_rotation = env.data.xmat[env.upper_body].reshape(3, 3)

                foot_up = float(foot_rotation[2, 2])
                upper_up = float(upper_rotation[2, 2])

                # Get reward components
                (
                    torque,
                    torque_reward,
                    momentum,
                    momentum_reward,
                    com_tilt_deg
                ) = calculate_reward_components(env)

                total_reward += reward

                # Record evaluation data
                writer.writerow([
                    episode + 1,
                    step,
                    pitch,
                    roll,
                    pitch_target,
                    roll_target,
                    pitch_vel,
                    roll_vel,
                    pitch_torque,
                    roll_torque,
                    foot_up,
                    upper_up,
                    torque,
                    torque_reward,
                    momentum,
                    momentum_reward,
                    com_tilt_deg,
                    reward
                ])

                # Print evaluation log
                if step % 20 == 0:

                    print(
                        f"Episode={episode + 1}, "
                        f"step={step:4d}, "
                        f"foot_up={foot_up:.3f}, "
                        f"upper_up={upper_up:.3f}, "
                        f"pitch={pitch:.3f}, "
                        f"roll={roll:.3f}, "
                        f"torque={torque:.5f}, "
                        f"momentum={momentum:.6f}, "
                        f"com_tilt={com_tilt_deg:.3f}, "
                        f"torque_reward={torque_reward:.5f}, "
                        f"momentum_reward={momentum_reward:.5f}, "
                        f"reward={reward:.5f}"
                    )

                if terminated or truncated:
                    break

            steps = step + 1
            results.append(steps)

            print(
                f"Episode {episode + 1} ended: "
                f"steps={steps}, "
                f"reward={total_reward:.2f}"
            )

    print("\n===== Evaluation Results =====")

    print(f"Average survival: {np.mean(results):.1f} steps")
    print(f"Maximum survival: {np.max(results)} steps")
    print(f"Minimum survival: {np.min(results)} steps")

    print(
        f"{env.max_steps}-step completion: "
        f"{sum(s >= env.max_steps for s in results)}/{len(results)}"
    )

    print("Survival per episode:", results)
    print("Evaluation data saved to evaluation.csv")

    env.close()


if __name__ == "__main__":
    main()
