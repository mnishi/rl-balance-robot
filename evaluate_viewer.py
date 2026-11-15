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
        model = PPO.load(args.model)
        obs, _ = env.reset()

        print(f"Loading model: {args.model}")
        print("Starting viewer...")

        sim_time = env.frame_skip * env.model.opt.timestep

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

            step = 0

            while viewer.is_running():

                start = time.perf_counter()

                # Predict the next action
                action, _ = model.predict(
                    obs,
                    deterministic=True
                )

                # Advance simulation
                obs, reward, terminated, truncated, _ = env.step(action)

                step += 1

                # Print evaluation log
                if step % 10 == 0:
                    print(
                        f"step={step:4d}, "
                        f"pitch={obs[6]:+.3f}, "
                        f"roll={obs[7]:+.3f}, "
                        f"action={action}"
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

                    # Wait until the viewer window is closed manually
                    while viewer.is_running():
                      time.sleep(0.1)

                    break

                # Match simulation time
                elapsed = time.perf_counter() - start
                time.sleep(max(0.0, sim_time - elapsed))

    finally:
        env.close()


if __name__ == "__main__":
    main()
