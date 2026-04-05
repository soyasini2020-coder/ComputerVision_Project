#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_food.sh Apple.png small
#   ./run_food.sh data/external_test/banana_real.jpg medium
#   ./run_food.sh salmon-steak.webp medium --min-confidence 0.6
# Defaults:
#   image: data/external_test/Apple.png
#   portion: medium

IMAGE_INPUT="${1:-data/external_test/Apple.png}"
PORTION="${2:-medium}"
EXTRA_ARGS="${*:3}"

if [[ "$IMAGE_INPUT" != */* ]]; then
  IMAGE_PATH="data/external_test/$IMAGE_INPUT"
else
  IMAGE_PATH="$IMAGE_INPUT"
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Image not found: $IMAGE_PATH"
  echo "Tip: put image in data/external_test and pass filename only."
  exit 1
fi

if [[ "$PORTION" != "small" && "$PORTION" != "medium" && "$PORTION" != "large" ]]; then
  echo "Invalid portion: $PORTION"
  echo "Use: small | medium | large"
  exit 1
fi

BASE_NAME="$(basename "$IMAGE_PATH")"
STEM="${BASE_NAME%.*}"
OUT_PATH="outputs/${STEM}_${PORTION}.jpg"

if [[ -x "/opt/anaconda3/envs/ocv/bin/python" ]]; then
  PY_CMD="/opt/anaconda3/envs/ocv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
  PY_CMD="python3"
else
  echo "Python not found. Install Python or activate your environment."
  exit 1
fi

"$PY_CMD" src/main.py --image "$IMAGE_PATH" --portion "$PORTION" --output-image "$OUT_PATH" $EXTRA_ARGS

echo "Saved output image: $OUT_PATH"
