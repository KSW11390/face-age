#!/bin/bash
echo "🚀 Starting Face-Age training..."

DATA_ROOT="/content/drive/MyDrive/face-age/datasets/UTKFace"
SAVE_DIR="/content/drive/MyDrive/face-age/checkpoints"

python3 -m faceage.train \
  --data_root="$DATA_ROOT" \
  --epochs=10 \
  --batch_size=64 \
  --lr=1e-4 \
  --save_dir="$SAVE_DIR" \
  --augment_minority_only

echo "✅ Training finished and model saved to $SAVE_DIR"