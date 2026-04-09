#!/bin/bash
# ScriptToReel — always starts with the correct Python (Homebrew 3.14)
# which has torch, diffusers, moviepy, kokoro-onnx etc. installed.
#
# Usage:  ./start.sh
#         ./start.sh --port 9090   (optional custom port)

PYTHON="/opt/homebrew/bin/python3"

if ! "$PYTHON" -c "import torch" &>/dev/null; then
  echo "❌ torch not found in $PYTHON — check your installation"
  exit 1
fi

echo "✅ Using $("$PYTHON" --version)"
exec "$PYTHON" server.py "$@"
