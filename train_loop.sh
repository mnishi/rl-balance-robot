#!/usr/bin/env bash

set -euo pipefail

MODEL="balance_ppo.zip"
BACKUP_DIR="models"

# Final cumulative training steps (100 million)
TARGET_STEPS=100000000

# Reset training artifacts
echo "========================================"
echo "Resetting training environment"
echo "========================================"

# Delete the existing model and checkpoints
rm -f -- "$MODEL"
rm -rf -- "$BACKUP_DIR"

# Create a new checkpoint directory
mkdir -p "$BACKUP_DIR"

# Start training from scratch
TOTAL_STEPS=0

while [ "$TOTAL_STEPS" -lt "$TARGET_STEPS" ]; do

    # Use 1M increments until 10M, then 10M increments
    if [ "$TOTAL_STEPS" -lt 10000000 ]; then
        INCREMENT=1000000
    else
        INCREMENT=10000000
    fi

    # Calculate the next cumulative step count
    NEXT_STEPS=$((TOTAL_STEPS + INCREMENT))

    echo "========================================"
    echo "Training: $TOTAL_STEPS -> $NEXT_STEPS"
    echo "========================================"

    # Train a new model or continue training the existing model
    python train.py "$INCREMENT"

    # Verify that the trained model was saved
    if [ ! -f "$MODEL" ]; then
        echo "Error: Model file not found after training."
        exit 1
    fi

    # Update cumulative training steps
    TOTAL_STEPS=$NEXT_STEPS

    # Save a checkpoint
    MILLIONS=$((TOTAL_STEPS / 1000000))
    BACKUP_FILE="$BACKUP_DIR/balance_ppo_${MILLIONS}M.zip"

    cp -- "$MODEL" "$BACKUP_FILE"

    echo "Checkpoint saved: $BACKUP_FILE"
    echo

done

echo "========================================"
echo "Training completed: $TOTAL_STEPS steps"
echo "Final model: $MODEL"
echo "========================================"
