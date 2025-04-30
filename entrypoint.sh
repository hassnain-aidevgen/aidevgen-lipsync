#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Entrypoint started"

# Run weight downloader if present
if [ -x "./download_weights.sh" ]; then
  echo "🛠️  Running download_weights.sh..."
  ./download_weights.sh
else
  echo "⚠️  download_weights.sh not found or not executable"
fi

# Start the main process (e.g., CMD in Dockerfile)
echo "🎬 Launching: $@"
exec "$@"
