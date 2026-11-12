import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from balance_env import BalanceEnv

MODEL_PATH = "balance_ppo"
TOTAL_TIMESTEPS = 1_000_000

env = BalanceEnv()

# Save checkpoints every 100,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="balance_ppo"
)

# Load an existing model or create a new one
if os.path.exists(MODEL_PATH + ".zip"):
    print("Loading existing model...")
    model = PPO.load(MODEL_PATH, env=env)
    reset_num_timesteps = False
else:
    print("Creating new model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/"
    )
    reset_num_timesteps = True

# Continue training for 1,000,000 steps
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=reset_num_timesteps,
    callback=checkpoint_callback
)

# Save the final model
model.save(MODEL_PATH)

env.close()

print("Training completed.")
