#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-dgx-spark-pytorch-ft:26.05}"
WORKSPACE="${WORKSPACE:-${PWD}}"

if [ -f "${HOME}/.config/secrets/hf_token" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/.config/secrets/hf_token"
fi

mkdir -p "${HOME}/.cache/huggingface" "${WORKSPACE}"

exec docker run --gpus all -it --rm --ipc=host \
  -e HF_TOKEN \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${WORKSPACE}:/workspace" \
  "${IMAGE}"
