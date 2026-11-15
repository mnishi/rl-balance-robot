import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_plot(filename, title, ylabel, time, series):

    plt.figure(figsize=(12, 5))

    for values, label, style in series:
        plt.plot(time, values, style, label=label)

    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episode",
        type=int,
        default=10,
        help="Episode number to plot (default: 10)"
    )

    parser.add_argument(
        "--csv",
        default="evaluation.csv",
        help="Evaluation CSV file"
    )

    args = parser.parse_args()

    # Load evaluation data
    df = pd.read_csv(args.csv)

    # Select episode
    df = df[df["episode"] == args.episode].copy()

    if df.empty:
        raise ValueError(
            f"Episode {args.episode} not found in {args.csv}"
        )

    # Simulation time: timestep=0.002, frame_skip=5
    dt = 0.01
    time = df["step"].to_numpy() * dt

    # Convert joint angles to degrees
    pitch = np.degrees(df["pitch"])
    roll = np.degrees(df["roll"])

    pitch_target = np.degrees(df["pitch_target"])
    roll_target = np.degrees(df["roll_target"])

    # Plot pitch angle
    save_plot(
        "evaluation_pitch.png",
        "Pitch: Target vs Actual",
        "Pitch Angle [deg]",
        time,
        [
            (pitch, "Actual", "-"),
            (pitch_target, "Target", "--")
        ]
    )

    # Plot roll angle
    save_plot(
        "evaluation_roll.png",
        "Roll: Target vs Actual",
        "Roll Angle [deg]",
        time,
        [
            (roll, "Actual", "-"),
            (roll_target, "Target", "--")
        ]
    )

    # Plot gravitational torque
    save_plot(
        "evaluation_torque.png",
        "Gravitational Torque",
        "Torque [N m]",
        time,
        [
            (df["torque"], "Gravity torque", "-")
        ]
    )

    # Plot reward components
    save_plot(
        "evaluation_reward.png",
        "Reward Components",
        "Reward",
        time,
        [
            (df["torque_reward"], "Torque reward", "-"),
            (df["movement_penalty"], "Movement penalty", "--"),
            (df["reward"], "Total reward", "-.")
        ]
    )

    # Plot joint movement
    save_plot(
        "evaluation_movement.png",
        "Average Joint Movement",
        "Mean Angle Change [rad/step]",
        time,
        [
            (
                df["mean_angle_change"],
                "Mean joint movement",
                "-"
            )
        ]
    )


if __name__ == "__main__":
    main()
