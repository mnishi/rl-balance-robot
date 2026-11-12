from balance_env import BalanceEnv
import numpy as np

env = BalanceEnv()

obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Initial observation:", obs)

for i in range(1000):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if i % 50 == 0:
        print(
            f"step={i}, "
            f"reward={reward:.3f}, "
            f"upright={obs[2]:.3f}, "
            f"pitch={obs[6]:.3f}, "
            f"roll={obs[7]:.3f}"
        )

    if terminated or truncated:
        print("Episode ended:", i)
        obs, info = env.reset()

print("Test completed")
