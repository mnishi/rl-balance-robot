from balance_env import BalanceEnv
import numpy as np

env = BalanceEnv()

obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Initial observation:", obs)

for i in range(1000):

    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)

    if i < 20 or i % 50 == 0:
      print(
          f"step={env.steps}, "
          f"reward={reward:.3f}, "
          f"com_y={obs[0]:.4f}, "
          f"com_vy={obs[1]:.4f}, "
          f"roll={obs[2]:.3f}" 
      )

    if terminated or truncated:
        print("Episode ended:", i)
        obs, info = env.reset()

print("Test completed")
