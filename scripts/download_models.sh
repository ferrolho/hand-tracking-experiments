#!/usr/bin/env bash
# Download the MediaPipe HandLandmarker model into the repo root.
# The model is gitignored (binary weight blob); fetch it once after cloning.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/hand_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if [[ -f "$DEST" ]]; then
  echo "Model already present: $DEST"
  exit 0
fi

echo "Downloading hand_landmarker.task ..."
curl -fSL -o "$DEST" "$URL"
echo "Saved to $DEST"
