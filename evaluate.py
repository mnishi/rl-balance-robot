import argparse
import csv

import mujoco
import numpy as np
from stable_baselines3 import PPO

from balance_env import BalanceEnv


def calculate_reward_components(env):

    # Calculate gravitational torque
    com_position = env.data.subtree_com[env.ball]
    ball_position = env.data.xpos[env.ball]

    contact_point = ball_position + np.array([0.0, 0.0, -0.025])

    gravity_torque = np.cross(
        com_position - contact_point,
        env.gravity_force
    )

    # Calculate gravity torque reward
    torque = np.linalg.norm(gravity_torque)
    torque_error = np.sum(gravity_torque ** 2) / (0.01 ** 2)
    torque_reward = 1.0 / (1.0 + torque_error)

    # Calculate joint movement penalty
    mean_angle_change = (
        env.angle_change_sum / len(env.angle_changes)
    )

    movement_penalty = env.movement_weight * mean_angle_change

    return torque, torque_reward, mean_angle_change, movement_penalty


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
            "torque",
            "torque_reward",
            "mean_angle_change",
            "movement_penalty",
            "reward"
        ])

        for episode in range(10):

            obs, info = env.reset()
            total_reward = 0.0

            for step in range(env.max_steps):

                action, _ = model.predict(
                    obs,
                    deterministic=True
                )

                obs, reward, terminated, truncated, info = env.step(
                    action
                )

                # Get joint states
                pitch = float(env.data.qpos[env.pitch_qpos])
                roll = float(env.data.qpos[env.roll_qpos])

                pitch_vel = float(env.data.qvel[env.pitch_dof])
                roll_vel = float(env.data.qvel[env.roll_dof])

                pitch_target = action[0] * env.max_angle
                roll_target = action[1] * env.max_angle

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
                    mean_angle_change,
                    movement_penalty
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
                    mean_angle_change,
                    movement_penalty,
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
                        f"torque_reward={torque_reward:.5f}, "
                        f"movement_penalty={movement_penalty:.6f}, "
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
