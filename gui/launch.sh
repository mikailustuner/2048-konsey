#!/usr/bin/env bash
# Starts the 2048-konsey GUI gateway (if not already running) and opens it in the browser.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${KONSEY_PORT:-8048}
URL="http://127.0.0.1:$PORT"
if [ ! -x build/play ]; then
    PATH="$HOME/.local/bin:$PATH" cmake -S . -B build -G Ninja >/dev/null && cmake --build build --target play
fi
if ! curl -sf "$URL/api/models" >/dev/null; then
    mkdir -p runs
    nohup python3 gui/server.py --port "$PORT" > runs/gateway.log 2>&1 &
    for _ in $(seq 50); do curl -sf "$URL/api/models" >/dev/null && break; sleep 0.1; done
fi
xdg-open "$URL" >/dev/null 2>&1 &
echo "$URL"
