#!/usr/bin/env bash
# launch-single-spark.sh — run Nemotron-Super (NVFP4) on ONE DGX Spark (TP=1),
# on the WORKER / second node, leaving the primary (head) node free — e.g. so
# qmx's Ollama/reranker can keep running there.
#
# Nemotron-Super NVFP4 (~75 GB) fits on a single 128 GB box, so this needs NO
# RoCE / NCCL / multi-node flags — none of the dual-node foot-guns apply.
# The two boxes don't contend for memory: Nemotron uses the worker, qmx the head.
#
# Config: reads cluster.env (copy from cluster.env.example). Overrides:
#   SINGLE_HOST  target node   (default: WORKER_HOST)
#   NAME         container     (default: vllm-solo)
#
# Usage:
#   ./launch-single-spark.sh up            # built-in Nemotron-Super recipe, TP=1
#   ./launch-single-spark.sh up -- <serve args...>
#   ./launch-single-spark.sh down
#   ./launch-single-spark.sh status
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${CLUSTER_ENV:-$SCRIPT_DIR/cluster.env}"
# shellcheck disable=SC1090
[ -f "$CONF" ] && . "$CONF"

IMAGE="${IMAGE:-vllm-spark-nccl:cu130}"
# Target the WORKER (second) node so the primary/head stays free for qmx etc.
HOST="${SINGLE_HOST:-${WORKER_HOST:-spark-b.local}}"
PORT="${SERVE_PORT:-8000}"
NAME="${NAME:-vllm-solo}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

# Same recipe as the dual launcher, but TP=1 and no distributed/multi-node flags.
DEFAULT_SERVE_ARGS="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --served-model-name nemotron-3-super --trust-remote-code --tensor-parallel-size 1 --max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-seqs 4 --load-format fastsafetensors --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"

cache_args() { echo "-v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.cache/vllm:/root/.cache/vllm -v ~/.cache/flashinfer:/root/.cache/flashinfer -v ~/.triton:/root/.triton -v ~/.tilelang:/root/.tilelang"; }

cmd_up() {
  local serve="$*"; [ -z "$serve" ] && serve="$DEFAULT_SERVE_ARGS"
  case " $serve " in
    *" --nnodes "*|*" --node-rank "*|*" --headless "*|*" --distributed-executor-backend "*|*" --tensor-parallel-size 2 "*)
      echo "ERROR: single-node serve args must not include multi-node flags or TP=2." >&2; exit 2;;
  esac
  local cmd="docker rm -f $NAME >/dev/null 2>&1; mkdir -p ~/.cache/vllm ~/.cache/flashinfer ~/.triton ~/.tilelang; \
docker run -d --name $NAME --network host --gpus all --ipc=host \
  --restart unless-stopped \
  --log-opt max-size=50m --log-opt max-file=3 \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  $(cache_args) \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint vllm $IMAGE serve $serve --host 0.0.0.0 --port $PORT"
  echo ">> [$HOST] launching $NAME (single node, TP=1)"
  $SSH "$HOST" "$cmd" >/dev/null
  echo "Launched single-node Nemotron on $HOST. Endpoint: http://$HOST:$PORT/v1 (first load can take a few minutes). Primary node left free."
}

cmd_down() { echo ">> [$HOST] removing $NAME"; $SSH "$HOST" "docker rm -f $NAME >/dev/null 2>&1 || true"; }

cmd_status() {
  echo -n "[$HOST] "; $SSH "$HOST" "docker ps --filter name=^/$NAME\$ --format '{{.Names}} {{.Status}}' || true"
  echo -n "endpoint: "; $SSH "$HOST" "curl -s -m 5 http://localhost:$PORT/v1/models >/dev/null 2>&1 && echo UP || echo 'not ready'"
}

action="${1:-up}"; shift || true
[ "${1:-}" = "--" ] && shift || true
case "$action" in
  up) cmd_up "$@";;
  down) cmd_down;;
  status) cmd_status;;
  *) echo "usage: $0 {up [-- <serve args>]|down|status}" >&2; exit 1;;
esac
