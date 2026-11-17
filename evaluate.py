import argparse
import csv

import mujoco
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
        0.0,
        0.0,
        -0.02
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
        help="PPO model file or 'zero' for zero-action evaluation"
    )

    args = parser.parse_args()

    env = BalanceEnv()

    # Load PPO model or select zero-action evaluation
    if args.model == "zero":
        model = None
        print("Evaluation mode: zero action")
    else:
        model = PPO.load(args.model)
        print(f"Loading model: {args.model}")

        # Check observation compatibility
        model_obs_shape = model.observation_space.shape
        env_obs_shape = env.observation_space.shape

        if model_obs_shape != env_obs_shape:
            raise ValueError(
                f"Observation shape mismatch: "
                f"model={model_obs_shape}, "
                f"environment={env_obs_shape}"
            )

    results = []

    # Open CSV file once for all episodes
    with open("evaluation.csv", "w", newline="") as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "episode",
            "step",

            "com_x",
            "com_y",
            "com_vx",
            "com_vy",

            "pitch",
            "roll",

            "pitch_target",
            "roll_target",

            "pitch_torque",
            "roll_torque",

            "gravity_torque",
            "torque_reward",

            "angular_momentum",
            "momentum_reward",

            "com_tilt_deg",

            "reward"
        ])

        for episode in range(100):

            obs, info = env.reset()
            total_reward = 0.0

            for step in range(env.max_steps):

                # Predict action or use zero action
                if model is None:
                    action = np.zeros(2, dtype=np.float32)
                else:
                    action, _ = model.predict(
                        obs,
                        deterministic=True
                    )

                # Execute simulation
                obs, reward, terminated, truncated, info = env.step(
                    action
                )

                # Get joint states
                pitch = float(
                    env.data.qpos[env.pitch_qpos]
                )

                roll = float(
                    env.data.qpos[env.roll_qpos]
                )

                # Get target angles
                pitch_target = float(
                    action[0] * env.max_angle
                )

                roll_target = float(
                    action[1] * env.max_angle
                )

                # Get actuator torques
                pitch_torque = float(
                    env.data.actuator_force[env.pitch_motor]
                )

                roll_torque = float(
                    env.data.actuator_force[env.roll_motor]
                )

                # Calculate reward components
                (
                    torque,
                    torque_reward,
                    momentum,
                    momentum_reward,
                    com_tilt_deg
                ) = calculate_reward_components(env)

                total_reward += reward

                # Use the same COM state as the training environment
                com_x, com_y, com_vx, com_vy = map(
                    float,
                    obs[:4]
                )

                # Get joint angles from observation
                pitch = float(obs[4])
                roll = float(obs[5])

                # Record evaluation data
                writer.writerow([
                    episode + 1,
                    step,

                    com_x,
                    com_y,
                    com_vx,
                    com_vy,

                    pitch,
                    roll,

                    pitch_target,
                    roll_target,

                    pitch_torque,
                    roll_torque,

                    torque,
                    torque_reward,

                    momentum,
                    momentum_reward,

                    com_tilt_deg,

                    reward
                ])

                # Print reward components every 10 steps
                if env.steps <= 200 and env.steps % 10 == 0:
                    print(
                        f"Episode={episode + 1}, "
                        f"Step={env.steps:4d}, "
                        f"Torque={torque:.6f} N*m, "
                        f"Momentum={momentum:.8f} kg*m^2/s, "
                        f"TorqueReward={torque_reward:.4f}, "
                        f"MomentumReward={momentum_reward:.4f}, "
                        f"Reward={reward:.4f}"
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

    print(
        f"Average survival: "
        f"{np.mean(results):.1f} steps"
    )

    print(
        f"Maximum survival: "
        f"{np.max(results)} steps"
    )

    print(
        f"Minimum survival: "
        f"{np.min(results)} steps"
    )

    print(
        f"{env.max_steps}-step completion: "
        f"{sum(s >= env.max_steps for s in results)}/{len(results)}"
    )

    print("Survival per episode:", results)
    print("Evaluation data saved to evaluation.csv")

    env.close()


if __name__ == "__main__":
    main()
