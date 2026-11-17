import argparse
import time

import mujoco.viewer
from stable_baselines3 import PPO

from balance_env import BalanceEnv


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

    try:
        # Load PPO model
        model = PPO.load(args.model)

        # Check observation compatibility
        if model.observation_space.shape != env.observation_space.shape:
            raise ValueError(
                f"Observation shape mismatch: "
                f"model={model.observation_space.shape}, "
                f"environment={env.observation_space.shape}"
            )

        obs, _ = env.reset()

        print(f"Loading model: {args.model}")
        print("Starting viewer...")

        # Simulation time per control step
        sim_time = env.frame_skip * env.model.opt.timestep

        with mujoco.viewer.launch_passive(
            env.model,
            env.data
        ) as viewer:

            step = 0

            while viewer.is_running():

                start = time.perf_counter()

                # Predict the next action
                action, _ = model.predict(
                    obs,
                    deterministic=True
                )

                # Advance simulation
                obs, reward, terminated, truncated, _ = env.step(
                    action
                )

                step += 1

                # Print evaluation log every 10 steps
                if step % 10 == 0:
                    print(
                        f"step={step:4d}, "
                        f"com_x={obs[0]:+.4f}, "
                        f"com_y={obs[1]:+.4f}, "
                        f"com_vx={obs[2]:+.4f}, "
                        f"com_vy={obs[3]:+.4f}, "
                        f"pitch={obs[4]:+.3f}, "
                        f"roll={obs[5]:+.3f}, "
                        f"action={action}, "
                        f"reward={reward:.4f}"
                    )

                # Update viewer
                viewer.sync()

                # Stop when the episode ends
                if terminated or truncated:

                    print(
                        f"Episode ended: "
                        f"steps={step}, "
                        f"terminated={terminated}, "
                        f"truncated={truncated}"
                    )

                    # Keep the viewer open until manually closed
                    while viewer.is_running():
                        time.sleep(0.1)

                    break

                # Match simulation time
                elapsed = time.perf_counter() - start

                time.sleep(
                    max(0.0, sim_time - elapsed)
                )

    finally:
        env.close()


if __name__ == "__main__":
    main()
