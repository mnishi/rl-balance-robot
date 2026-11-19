import argparse
import csv

import numpy as np
from stable_baselines3 import PPO

from balance_env import BalanceEnv


def calculate_reward_components(env):
    """Calculate roll-axis gravity torque and angular momentum rewards."""

    # Calculate system center of mass
    com_position = env.data.subtree_com[env.ball]
    ball_position = env.data.xpos[env.ball]

    # Approximate the ground contact point for the cylindrical base
    contact_point = ball_position + np.array([0.0, 0.0, -0.02])

    # Calculate gravity torque around the roll axis
    gravity_torque = np.cross(
        com_position - contact_point,
        env.gravity_force
    )

    roll_gravity_torque = abs(float(gravity_torque[0]))

    torque_reward = 1.0 / (
        1.0 + roll_gravity_torque / 0.01
    )

    # Calculate total angular momentum around the roll axis
    angular_momentum = env.get_total_angular_momentum(
        contact_point
    )

    roll_momentum = abs(float(angular_momentum[0]))

    momentum_reward = 1.0 / (
        1.0 + (roll_momentum / 0.001) ** 2
    )

    # Calculate COM tilt angle
    com_vector = com_position - contact_point

    horizontal_distance = np.linalg.norm(
        com_vector[:2]
    )

    com_tilt_deg = float(
        np.degrees(
            np.arctan2(
                horizontal_distance,
                com_vector[2]
            )
        )
    )

    return (
        roll_gravity_torque,
        torque_reward,
        roll_momentum,
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

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )

    parser.add_argument(
        "--csv",
        default="evaluation.csv",
        help="CSV output path"
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    env = BalanceEnv()

    try:

        # Load PPO model or select zero-action evaluation
        if args.model == "zero":
            model = None
            print("Evaluation mode: zero action")

        else:
            model = PPO.load(args.model)
            print(f"Loading model: {args.model}")

            # Check observation compatibility
            if model.observation_space.shape != env.observation_space.shape:
                raise ValueError(
                    f"Observation shape mismatch: "
                    f"model={model.observation_space.shape}, "
                    f"environment={env.observation_space.shape}"
                )

            # Check action compatibility
            if model.action_space.shape != env.action_space.shape:
                raise ValueError(
                    f"Action shape mismatch: "
                    f"model={model.action_space.shape}, "
                    f"environment={env.action_space.shape}"
                )

        results = []

        # Open CSV file once for all episodes
        with open(args.csv, "w", newline="") as csv_file:

            writer = csv.writer(csv_file)

            writer.writerow([
                "episode",
                "step",
                "com_y",
                "com_vy",
                "roll",
                "roll_target",
                "roll_torque",
                "gravity_torque",
                "torque_reward",
                "angular_momentum",
                "momentum_reward",
                "com_tilt_deg",
                "reward"
            ])

            for episode in range(1, args.episodes + 1):

                obs, _ = env.reset()
                total_reward = 0.0
                steps = 0

                for step in range(env.max_steps):

                    # Predict action or use zero action
                    if model is None:
                        action = np.zeros(
                            env.action_space.shape,
                            dtype=np.float32
                        )
                    else:
                        action, _ = model.predict(
                            obs,
                            deterministic=True
                        )

                    # Match the action clipping performed by the environment
                    clipped_action = np.clip(
                        action,
                        -1.0,
                        1.0
                    )

                    # Execute simulation
                    obs, reward, terminated, truncated, _ = env.step(
                        action
                    )

                    steps = step + 1
                    total_reward += float(reward)

                    # Get observation values
                    com_y, com_vy, roll = map(
                        float,
                        obs[:3]
                    )

                    # Get target angle
                    roll_target = float(
                        clipped_action[0] * env.max_angle
                    )

                    # Get actuator torque
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

                    # Record evaluation data
                    writer.writerow([
                        episode,
                        step,
                        com_y,
                        com_vy,
                        roll,
                        roll_target,
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
                            f"Episode={episode}, "
                            f"Step={env.steps:4d}, "
                            f"Torque={torque:.6f} N*m, "
                            f"Momentum={momentum:.8f} kg*m^2/s, "
                            f"TorqueReward={torque_reward:.4f}, "
                            f"MomentumReward={momentum_reward:.4f}, "
                            f"Reward={reward:.4f}"
                        )

                    if terminated or truncated:
                        break

                results.append(steps)

                print(
                    f"Episode {episode} ended: "
                    f"steps={steps}, "
                    f"reward={total_reward:.2f}"
                )

        # Print evaluation summary
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
        print(f"Evaluation data saved to {args.csv}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
