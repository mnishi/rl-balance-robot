import numpy as np
from stable_baselines3 import PPO
from balance_env import BalanceEnv
import mujoco

env = BalanceEnv()

model = PPO.load("balance_ppo")

obs, info = env.reset()

for i in range(env.model.nbody):
    name = mujoco.mj_id2name(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        i
    )
    print(f"BODY ID={i}: {name}")

results = []

for episode in range(10):
    obs, info = env.reset()
    total_reward = 0.0

    for i in range(2000):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if i % 20 == 0:
            # Leg orientation
            leg_rotation = env.data.xmat[env.leg].reshape(3, 3)
            leg_up = leg_rotation[:, 2]
            leg_upright = leg_up[2]

            # Upper body orientation
            upper_rotation = env.data.xmat[env.upper_body].reshape(3, 3)
            upper_up = upper_rotation[:, 2]
            upper_upright = upper_up[2]

            print(
                f"Episode={episode + 1}, "
                f"step={i:3d}, "
                f"leg_up={leg_upright:.3f}, "
                f"upper_up={upper_upright:.3f}, "
                f"pitch={obs[6]:.3f}, "
                f"roll={obs[7]:.3f}, "
                f"action={action}"
            )

        if terminated or truncated:
            break

    steps = i + 1
    results.append(steps)

    print(
        f"Episode {episode + 1} ended: "
        f"steps={steps}, reward={total_reward:.1f}"
    )

print("\n===== Evaluation Results =====")
print(f"Average survival: {np.mean(results):.1f} steps")
print(f"Maximum survival: {np.max(results)} steps")
print(f"Minimum survival: {np.min(results)} steps")
print(f"2000-step completion: {sum(s >= 2000 for s in results)}/10")
print("Survival per episode:", results)
env.close()
