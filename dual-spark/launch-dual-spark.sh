#!/usr/bin/env bash
# launch-dual-spark.sh — launch identical vLLM (TP across nodes, no Ray) on two+
# DGX Spark boxes from the OFFICIAL vllm/vllm-openai image, with RoCE/RDMA on.
#
# It removes the two easy-to-get-wrong-by-hand foot-guns:
#   1) SAME engine args on every node   (mismatched args -> CUDA-graph deadlock)
#   2) RDMA verbs devices + memlock in the container
#      (missing -> NCCL silently falls back to TCP, i.e. no RoCE)
#
# The launcher only differs per node by: --node-rank, --headless (workers),
# --host/--port (head), and VLLM_HOST_IP. Everything else is identical.
#
# Usage:
#   ./launch-dual-spark.sh up                     # built-in Nemotron-Super recipe
#   ./launch-dual-spark.sh up -- <serve args...>  # custom (identical on all nodes)
#   ./launch-dual-spark.sh down
#   ./launch-dual-spark.sh status
#
# Config via env (defaults match this repo's dual-Spark setup):
#   NODES, ROCE_IPS (head first, same order), IMAGE, PORT, MASTER_PORT,
#   IB_HCA, IB_GID_INDEX, IB_IFACES, GLOO_IF, UVERBS
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai:cu130-nightly}"
read -r -a NODES     <<< "${NODES:-spark-0e81.local spark-0f50.local}"   # SSH hosts, head first
read -r -a ROCE_IPS  <<< "${ROCE_IPS:-192.168.100.1 192.168.100.2}"      # RoCE IP per node
MASTER_PORT="${MASTER_PORT:-29500}"
PORT="${PORT:-8000}"
IB_HCA="${IB_HCA:-rocep1s0f0,roceP2p1s0f0}"
IB_GID_INDEX="${IB_GID_INDEX:-3}"
IB_IFACES="${IB_IFACES:-enp1s0f0np0,enP2p1s0f0np0}"
GLOO_IF="${GLOO_IF:-enp1s0f0np0}"
read -r -a UVERBS    <<< "${UVERBS:-uverbs0 uverbs1 uverbs2 uverbs3}"     # RDMA verbs devices

# Identical serve args for every node. Do NOT put --node-rank/--headless/--host/
# --port/--nnodes/--master-*/--distributed-executor-backend here (added per node).
DEFAULT_SERVE_ARGS="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --served-model-name nemotron-3-super --trust-remote-code --tensor-parallel-size 2 --max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-seqs 4 --load-format fastsafetensors --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"

NNODES="${#NODES[@]}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
name_for() { [ "$1" -eq 0 ] && echo vllm-head || { [ "$NNODES" -gt 2 ] && echo "vllm-worker-$1" || echo vllm-worker; }; }

dev_args() { local a="--device=/dev/infiniband/rdma_cm"; for u in "${UVERBS[@]}"; do a="$a --device=/dev/infiniband/$u"; done; echo "$a"; }
cache_args() { echo "-v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.cache/vllm:/root/.cache/vllm -v ~/.cache/flashinfer:/root/.cache/flashinfer -v ~/.triton:/root/.triton -v ~/.tilelang:/root/.tilelang"; }

cmd_up() {
  local serve="$*"; [ -z "$serve" ] && serve="$DEFAULT_SERVE_ARGS"
  case " $serve " in
    *" --node-rank "*|*" --headless "*|*" --nnodes "*|*" --master-addr "*|*" --distributed-executor-backend "*)
      echo "ERROR: serve args must not include per-node flags (--node-rank/--headless/--nnodes/--master-*/--distributed-executor-backend)." >&2; exit 2;;
  esac
  local i
  for i in "${!NODES[@]}"; do
    local host="${NODES[$i]}" ip="${ROCE_IPS[$i]}" name; name="$(name_for "$i")"
    local role="--headless"; [ "$i" -eq 0 ] && role="--host 0.0.0.0 --port $PORT"
    local cmd="docker rm -f $name >/dev/null 2>&1; mkdir -p ~/.cache/vllm ~/.cache/flashinfer ~/.triton ~/.tilelang; \
docker run -d --name $name --network host --gpus all --ipc=host \
  --log-opt max-size=50m --log-opt max-file=3 \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  $(dev_args) $(cache_args) \
  -e VLLM_HOST_IP=$ip -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=$IB_GID_INDEX -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME=$IB_IFACES -e GLOO_SOCKET_IFNAME=$GLOO_IF -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint vllm $IMAGE serve $serve \
    --distributed-executor-backend mp --nnodes $NNODES --node-rank $i \
    --master-addr ${ROCE_IPS[0]} --master-port $MASTER_PORT $role"
    echo ">> [$host] launching $name (rank $i)"
    $SSH "$host" "$cmd" >/dev/null
  done
  echo "Launched $NNODES-node cluster. Endpoint: http://${NODES[0]}:$PORT/v1 (weights load + first compile can take a few minutes)."
}

cmd_down() {
  local i; for i in "${!NODES[@]}"; do local name; name="$(name_for "$i")"; echo ">> [${NODES[$i]}] removing $name"; $SSH "${NODES[$i]}" "docker rm -f $name >/dev/null 2>&1 || true"; done
}

cmd_status() {
  local i; for i in "${!NODES[@]}"; do local name; name="$(name_for "$i")"; echo -n "[${NODES[$i]}] "; $SSH "${NODES[$i]}" "docker ps --filter name=^/$name\$ --format '{{.Names}} {{.Status}}' || true"; done
  echo -n "endpoint: "; $SSH "${NODES[0]}" "curl -s -m 5 http://localhost:$PORT/v1/models >/dev/null 2>&1 && echo UP || echo 'not ready'"
}

action="${1:-up}"; shift || true
[ "${1:-}" = "--" ] && shift || true
case "$action" in
  up) cmd_up "$@";;
  down) cmd_down;;
  status) cmd_status;;
  *) echo "usage: $0 {up [-- <serve args>]|down|status}" >&2; exit 1;;
esac
