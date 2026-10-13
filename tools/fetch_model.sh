#!/usr/bin/env bash
# Copies trained networks from Google Drive (written by Colab) into models/, where the GUI
# finds them.  usage: tools/fetch_model.sh [drive_dir=2048/final] [remote=gdrive]
# One-time setup (opens the browser for Google sign-in, read-only access):
#   rclone config create gdrive drive scope=drive.readonly
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=${1:-2048/final}
REMOTE=${2:-gdrive}
export PATH="$HOME/.local/bin:$PATH"
if ! rclone listremotes | grep -qx "$REMOTE:"; then
    echo "rclone uzak '$REMOTE' yok. Bir kez calistirin:" >&2
    echo "  rclone config create $REMOTE drive scope=drive.readonly" >&2
    exit 1
fi
mkdir -p models
rclone copy "$REMOTE:$SRC" models/ --include "*.bin" --progress
ls -lh models/*.bin
