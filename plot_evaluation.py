import argparse
from pathlib import Path

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

    # CSV step starts at zero
    time = (df["step"].to_numpy() + 1) * dt

    # Convert joint angles to degrees
    pitch = np.degrees(df["pitch"])
    roll = np.degrees(df["roll"])

    pitch_target = np.degrees(df["pitch_target"])
    roll_target = np.degrees(df["roll_target"])

    # Convert COM position to millimeters
    com_x = df["com_x"] * 1000.0
    com_y = df["com_y"] * 1000.0

    # COM relative velocity
    com_vx = df["com_vx"]
    com_vy = df["com_vy"]

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

    # Plot COM position
    save_plot(
        "evaluation_com_position.png",
        "COM Position Relative to Ball Center",
        "COM Position [mm]",
        time,
        [
            (com_x, "COM X", "-"),
            (com_y, "COM Y", "--")
        ]
    )

    # Plot COM velocity
    save_plot(
        "evaluation_com_velocity.png",
        "COM Velocity Relative to Ball Center",
        "COM Velocity [m/s]",
        time,
        [
            (com_vx, "COM Vx", "-"),
            (com_vy, "COM Vy", "--")
        ]
    )

    # Plot actuator torques
    save_plot(
        "evaluation_actuator_torque.png",
        "Actuator Torques",
        "Torque [N m]",
        time,
        [
            (df["pitch_torque"], "Pitch torque", "-"),
            (df["roll_torque"], "Roll torque", "--")
        ]
    )

    # Plot gravitational torque
    save_plot(
        "evaluation_torque.png",
        "Gravitational Torque",
        "Torque [N m]",
        time,
        [
            (df["gravity_torque"], "Gravity torque", "-")
        ]
    )

    # Plot angular momentum
    save_plot(
        "evaluation_momentum.png",
        "Total Angular Momentum",
        "Angular Momentum [kg m^2/s]",
        time,
        [
            (df["angular_momentum"], "Angular momentum", "-")
        ]
    )

    # Plot center of mass tilt
    save_plot(
        "evaluation_com_tilt.png",
        "Center of Mass Tilt",
        "COM Tilt [deg]",
        time,
        [
            (df["com_tilt_deg"], "COM tilt", "-")
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
            (df["momentum_reward"], "Momentum reward", "--"),
            (df["reward"], "Total reward", "-.")
        ]
    )


if __name__ == "__main__":
    main()
