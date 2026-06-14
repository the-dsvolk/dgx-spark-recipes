#!/usr/bin/env bash
set -euo pipefail

# Step 5 (runtime): Hugging Face auth via HF_TOKEN (passed by runme.sh).
if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -f /root/.cache/huggingface/token ]]; then
  echo "Warning: HF_TOKEN not set and no cached token in /root/.cache/huggingface." >&2
  echo "Run ./runme.sh from the host, or: hf auth login" >&2
fi

exec "$@"
