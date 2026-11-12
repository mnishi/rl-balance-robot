import time
import numpy as np
import mujoco
import mujoco.viewer

from stable_baselines3 import PPO
from balance_env import BalanceEnv


env = BalanceEnv()
model = PPO.load("balance_ppo")

obs, info = env.reset()

print("Starting viewer...")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    step = 0

    while viewer.is_running():

        start = time.time()

        # PPOによる行動決定
        action, _ = model.predict(obs, deterministic=True)

        # 環境を1ステップ進める
        obs, reward, terminated, truncated, info = env.step(action)

        step += 1

        rotation = env.data.xmat[env.leg].reshape(3, 3)
        leg_up = rotation[:, 2]
        upright = leg_up[2]

        if step % 10 == 0:
            print(
                f"step={step:4d}, "
                f"upright={upright:.3f}, "
                f"pitch={obs[6]:.3f}, "
                f"roll={obs[7]:.3f}, "
                f"action={action}"
            )

        # ビューワー更新
        viewer.sync()

        # シミュレーション時間に合わせる
        elapsed = time.time() - start
        sim_time = env.frame_skip * env.model.opt.timestep
        time.sleep(max(0, sim_time - elapsed))

        if terminated or truncated:
            print("Episode ended")
            break

env.close()
