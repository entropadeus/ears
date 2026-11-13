#!/usr/bin/env bash
# One-shot setup. Safe to re-run.
set -e
cd "$(dirname "$0")"
PIP="python3 -m pip install -q"
$PIP numpy scipy matplotlib 2>/dev/null || $PIP --break-system-packages numpy scipy matplotlib
# Layer 3 (recognition). CPU wheels are enough. Skip with EARS_NO_TORCH=1.
if [ -z "$EARS_NO_TORCH" ]; then
  python3 -c "import torch, torchaudio" 2>/dev/null || \
    $PIP torch torchaudio --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    $PIP --break-system-packages torch torchaudio --index-url https://download.pytorch.org/whl/cpu || \
    echo "torch install failed: layers 1 and 2 still work, judge/tune/taste will not"
fi
# Renderer (only needed to get audio out of Web Audio code)
command -v node >/dev/null && npm install --silent --no-fund --no-audit || echo "node missing: render.mjs unavailable"
python3 selftest.py | tail -1
