from balance_env import BalanceEnv
import numpy as np

env = BalanceEnv()

obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Initial observation:", obs)

for i in range(1000):

    #action = env.action_space.sample()
    #action = np.array([0.0, 0.0], dtype=np.float32) 
    #action = np.array([1.0, -1.0], dtype=np.float32)
    #action = np.array([0.5, -0.5], dtype=np.float32)
    action = np.array([0.1, -0.1], dtype=np.float32)
    
    obs, reward, terminated, truncated, info = env.step(action)

    if i < 20 or i % 50 == 0:
        print(
            f"step={i}, "
            f"reward={reward:.3f}, "
            f"upright={obs[2]:.3f}, "
            f"pitch={obs[6]:.3f}, "
            f"roll={obs[7]:.3f}, "
            f"ball_omega={np.linalg.norm(obs[10:13]):.3f}, "
            f"upper_omega={np.linalg.norm(obs[3:6]):.3f}"
        )

    if terminated or truncated:
        print("Episode ended:", i)
        obs, info = env.reset()

print("Test completed")
