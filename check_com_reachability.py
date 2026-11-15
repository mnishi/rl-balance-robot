"""Check horizontal center-of-mass controllability of a MuJoCo model."""

import argparse
import numpy as np
import mujoco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="balance.xml")
    parser.add_argument("--range-deg", type=float, default=30.0)
    parser.add_argument("--grid", type=int, default=61)
    parser.add_argument("--delta-deg", type=float, default=0.01)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    ball_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "ball"
    )
    pitch_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "pitch"
    )
    roll_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "roll"
    )

    if min(ball_id, pitch_id, roll_id) < 0:
        raise ValueError("Required names: ball, pitch, roll")

    pitch_q = model.jnt_qposadr[pitch_id]
    roll_q = model.jnt_qposadr[roll_id]

    baseline = data.qpos.copy()

    def com_xy(pitch, roll):
        data.qpos[:] = baseline
        data.qpos[pitch_q] = pitch
        data.qpos[roll_q] = roll

        mujoco.mj_forward(model, data)

        return data.subtree_com[ball_id, :2].copy()

    # Calculate the COM Jacobian using central differences.
    h = np.deg2rad(args.delta_deg)

    center = com_xy(0.0, 0.0)

    jacobian = np.column_stack((
        (com_xy(h, 0) - com_xy(-h, 0)) / (2 * h),
        (com_xy(0, h) - com_xy(0, -h)) / (2 * h),
    ))

    singular_values = np.linalg.svd(
        jacobian, compute_uv=False
    )

    rank = np.linalg.matrix_rank(jacobian)

    print("COM at pitch=roll=0 [m]:", center)

    print("Jacobian [m/rad] (rows X,Y; columns Pitch,Roll):")
    print(jacobian)

    print("Singular values [m/rad]:", singular_values)
    print("Rank:", rank)
    print("Condition number:", np.linalg.cond(jacobian))

    print(
        "Local 2D movement:",
        "YES" if rank == 2 else "NO (singular)"
    )

    # Sample the reachable COM positions over the joint-angle range.
    angles = np.deg2rad(
        np.linspace(
            -args.range_deg,
            args.range_deg,
            args.grid
        )
    )

    points = np.array([
        com_xy(p, r)
        for p in angles
        for r in angles
    ])

    distances = np.linalg.norm(points, axis=1)

    best = int(np.argmin(distances))
    pitch_best, roll_best = divmod(best, len(angles))

    print(
        f"Grid: {args.grid} x {args.grid}, "
        f"range: +/-{args.range_deg:g} deg"
    )

    print(
        "COM X range [m]:",
        points[:, 0].min(),
        points[:, 0].max()
    )

    print(
        "COM Y range [m]:",
        points[:, 1].min(),
        points[:, 1].max()
    )

    print(
        "Closest sampled COM to ball center XY [m]:",
        points[best]
    )

    print(
        "Distance to ball center XY [m]:",
        distances[best]
    )

    print(
        "Angles at closest sample [deg]:",
        np.rad2deg(angles[pitch_best]),
        np.rad2deg(angles[roll_best])
    )

    print(
        "Note: Grid coverage and local Jacobian do not "
        "prove global reachability or dynamic stability."
    )


if __name__ == "__main__":
    main()
