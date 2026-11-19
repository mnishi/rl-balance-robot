#!/usr/bin/env bash

set -euo pipefail

MODEL="balance_ppo.zip"
BACKUP_DIR="models"
TARGET_STEPS=100000000

mkdir -p "$BACKUP_DIR"

# Read cumulative training steps from the saved PPO model.
if [ -f "$MODEL" ]; then
    TOTAL_STEPS=$(python - "$MODEL" <<'PY'
import sys
from stable_baselines3 import PPO

model = PPO.load(sys.argv[1], device="cpu")
print(model.num_timesteps)
PY
)
else
    TOTAL_STEPS=0
fi

echo "========================================"
echo "Current training steps: $TOTAL_STEPS"
echo "Target training steps:  $TARGET_STEPS"
echo "========================================"

while [ "$TOTAL_STEPS" -lt "$TARGET_STEPS" ]; do

    # Select the checkpoint interval.
    if [ "$TOTAL_STEPS" -lt 10000000 ]; then
        INTERVAL=1000000
    else
        INTERVAL=10000000
    fi

    # Calculate the next interval boundary.
    NEXT_STEPS=$(( (TOTAL_STEPS / INTERVAL + 1) * INTERVAL ))

    # Do not exceed the final target.
    if [ "$NEXT_STEPS" -gt "$TARGET_STEPS" ]; then
        NEXT_STEPS=$TARGET_STEPS
    fi

    INCREMENT=$((NEXT_STEPS - TOTAL_STEPS))

    echo "========================================"
    echo "Training: $TOTAL_STEPS -> $NEXT_STEPS"
    echo "Additional steps: $INCREMENT"
    echo "========================================"

    # Continue training from the existing model.
    python train.py "$INCREMENT"

    if [ ! -f "$MODEL" ]; then
        echo "Error: Model file not found after training."
        exit 1
    fi

    # Read the actual saved step count.
    TOTAL_STEPS=$(python - "$MODEL" <<'PY'
import sys
from stable_baselines3 import PPO

model = PPO.load(sys.argv[1], device="cpu")
print(model.num_timesteps)
PY
)

    if [ "$TOTAL_STEPS" -lt "$NEXT_STEPS" ]; then
        echo "Error: Training did not reach the expected step count."
        exit 1
    fi

    # Save a checkpoint using the actual cumulative step count.
    BACKUP_FILE="$BACKUP_DIR/balance_ppo_${TOTAL_STEPS}.zip"
    cp -- "$MODEL" "$BACKUP_FILE"

    echo "Checkpoint saved: $BACKUP_FILE"
    echo
done

echo "========================================"
echo "Training completed: $TOTAL_STEPS steps"
echo "Final model: $MODEL"
echo "========================================"
