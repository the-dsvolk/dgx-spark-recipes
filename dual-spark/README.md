# Dual DGX Spark — Nemotron-3-Super-120B (NVFP4) over RoCE with vLLM (TP=2)

Serve NVIDIA **Nemotron-3-Super-120B-A12B (NVFP4)** across **two DGX Spark (GB10)** boxes with
tensor parallelism (**TP=2**) on **vLLM**, using the direct **ConnectX-7 200 GbE / RoCE** link
between the nodes. OpenAI-compatible endpoint, ~131K context.

This recipe is written in the order we actually built it up:

1. **SSH** between the two nodes (passwordless).
2. **RoCE / ConnectX-7** — the interconnect and how to make NCCL actually use it.
3. **Why vLLM** (vs TensorRT-LLM / SGLang, and why not Ollama) — and why Nemotron-Super + TP=2.
4. **Deployment** — the exact multi-node launch, the gotchas, validation, and teardown.

> Conventions: the two nodes are **`spark-a`** (head, rank 0) and **`spark-b`** (worker, rank 1) —
> these are *role labels only*. **Every site-specific hostname/IP lives in one file, `cluster.env`**
> (copy from [`cluster.env.example`](./cluster.env.example); it is git-ignored). The commands below
> reference `$VARIABLES` from it — `source ./cluster.env` first, or let
> [`launch-dual-spark.sh`](./launch-dual-spark.sh) read it for you. Only the `.example` (with generic
> values) is committed — **no real IPs, passwords, tokens, or keys are in this repo.**

### Cluster variables (all defined in `cluster.env`)

| Variable | Meaning | Example |
| --- | --- | --- |
| `HEAD_HOST` / `WORKER_HOST` | SSH names (rank 0 / rank 1) | `spark-a.local` / `spark-b.local` |
| `HEAD_RAIL0_IP` / `WORKER_RAIL0_IP` | RoCE rail 0 (port p0, PCIe dom 0000) | `192.168.100.1` / `192.168.100.2` |
| `HEAD_RAIL1_IP` / `WORKER_RAIL1_IP` | RoCE rail 1 (PCIe dom 0002) | `192.168.101.1` / `192.168.101.2` |
| `MASTER_IP` / `MASTER_PORT` | NCCL rendezvous (= `HEAD_RAIL0_IP`) / port | `192.168.100.1` / `29500` |
| `SERVE_PORT` | OpenAI API port (on head) | `8000` |
| `IB_HCA` / `IB_IFACES` / `IB_GID_INDEX` | RoCE HCAs / netdevs / RoCEv2 GID | `rocep1s0f0,roceP2p1s0f0` / … / `3` |

---

## Result / what you get

| Property | Value |
| --- | --- |
| Model | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (~120.6B total / 12.7B active, MoE, hybrid Mamba-Transformer) |
| Parallelism | Tensor-parallel **TP=2** across 2 nodes (1 GPU each) |
| Quant | **NVFP4** weights (Blackwell-native), **fp8** KV cache |
| Context | up to **131,072** tokens |
| Interconnect | direct **ConnectX-7 200 GbE** QSFP, **RoCEv2** (no switch) |
| Serving | vLLM OpenAI-compatible API on `$HEAD_HOST:$SERVE_PORT` |
| Decode | ~mid-20s tok/s single-stream (matches NVIDIA's single-Spark eval) |

Each box is a GB10 with **128 GB unified LPDDR5X**. The NVFP4 checkpoint is ~75 GB and *fits on a
single box*, so TP=2 here is primarily for **more KV / context / concurrency headroom**, not because
the weights don't fit. (A single-node TP=1 deployment is a valid, simpler fallback.)

---

## Hardware & topology

- 2× DGX Spark (GB10, `sm_121`, arm64, CUDA 13, 128 GB unified).
- 1× QSFP56 **200 G** direct-attach cable between the two boxes (**one** cable is enough — see
  [RoCE](#2-roce--the-connectx-7-fabric)).
- Both nodes on the same management LAN (Wi-Fi/Ethernet) for internet + control.

**Verified software/firmware levels (both nodes, kept identical):**

| Component | Version |
| --- | --- |
| NVIDIA driver | **580.173.02** (`nvidia-driver-580-open`; 580.x pinned, 590.x blocked) |
| ConnectX-7 NIC firmware | **28.45.4028** |
| DGX OS | **7.5.0** (SWBUILD 7.2.3), Ubuntu 24.04.4 LTS, kernel 6.17 |
| Serving image | `vllm/vllm-openai:cu130-nightly` + [`Dockerfile.nccl-fix`](./Dockerfile.nccl-fix) → `vllm-spark-nccl:cu130` |

```
        management LAN (mDNS: spark-a.local / spark-b.local)
   ┌────────────┐                              ┌────────────┐
   │  spark-a   │  ── ConnectX-7 QSFP 200G ──  │  spark-b   │
   │ (rank 0)   │     direct cable (RoCE)      │ (rank 1)   │
   │ vLLM head  │                              │ vLLM worker│
   └────────────┘                              └────────────┘
```

---

## 1. SSH between the nodes (passwordless)

Multi-node vLLM does **not** need inter-node SSH at runtime (the ranks talk over the RoCE fabric),
but passwordless SSH makes orchestration (launching, log tailing) far easier.

From your workstation, both boxes are reachable by their mDNS names (`$HEAD_HOST` / `$WORKER_HOST`)
and you log in as the **same user** on both (`$SSH_USER` — a same username on both nodes is required
by the NVIDIA multi-node tooling). `source ./cluster.env` first to use the variables below.

Distribute your public key once so subsequent logins need no password:

```bash
# one-time: you'll be prompted for the node's login password ONCE, then never again
ssh-copy-id "$SSH_USER@$HEAD_HOST"
ssh-copy-id "$SSH_USER@$WORKER_HOST"
```

Verify:

```bash
ssh -o BatchMode=yes "$SSH_USER@$HEAD_HOST" hostname   # must NOT prompt for a password
ssh -o BatchMode=yes "$SSH_USER@$WORKER_HOST" hostname
```

Notes / gotchas:
- mDNS `*.local` names resolve only on the same LAN; if they don't resolve, the box is off/rebooting
  or off-LAN (not an auth problem).
- If you use a per-host SSH config `Include`, keep any custom `Host` stanza **below** the `Include`.
  An `Include` placed *after* a `Host` block becomes nested inside it and only loads on match —
  silently breaking every other host.
- Docker access: add your user to the `docker` group on **both** nodes (`sudo usermod -aG docker
  <user>`; reconnect) so you don't need `sudo` for `docker`.

---

## 2. RoCE — the ConnectX-7 fabric

This is the part that makes multi-node TP viable. Get it wrong and it silently works but slowly.

### 2.1 One cable = two PCIe rails (important GB10 quirk)

On GB10 each **physical** QSFP port is exposed as **two** network interfaces on **two separate PCIe
domains** — i.e. one cable lights up two netdevs / two RoCE devices. Confirm with:

```bash
ibdev2netdev
#   rocep1s0f0   port 1 ==> enp1s0f0np0   (Up)     # physical port p0, PCIe domain 0000
#   roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)     # physical port p0, PCIe domain 0002
#   rocep1s0f1   ... enp1s0f1np1  (Down)           # physical port p1 (uncabled here)
#   roceP2p1s0f1 ... enP2p1s0f1np1 (Down)

# prove both "f0" netdevs are the SAME physical port:
cat /sys/class/net/enp1s0f0np0/phys_port_name     # p0
cat /sys/class/net/enP2p1s0f0np0/phys_port_name    # p0  (same port, two PCIe rails)
```

**Consequence:** to get the port's full bandwidth you must assign IPs to **both** rails. A *second*
physical cable does **not** increase two-node throughput (the GB10 chip's NIC bandwidth is already
reached by one port's two rails) — a second cable is only for 3–4-node ring/switch topologies.

### 2.2 Persistent addressing (netplan)

Give each rail an address on **both** nodes. Create `/etc/netplan/99-connectx.yaml`
(`chmod 600`, then `sudo netplan apply`). We used one subnet per rail:

On the **head** use `HEAD_RAIL{0,1}_IP`; on the **worker** use `WORKER_RAIL{0,1}_IP` (values from
`cluster.env`). Netplan is static YAML, so substitute the literals in — head example:

```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp1s0f0np0:      # rail 0  (IB_IFACES[0])
      optional: true
      mtu: 9000        # jumbo frames (optional; RoCE MTU is 4096 regardless)
      addresses: [<HEAD_RAIL0_IP>/24]
    enP2p1s0f0np0:    # rail 1  (IB_IFACES[1])
      optional: true
      mtu: 9000
      addresses: [<HEAD_RAIL1_IP>/24]
```

`optional: true` keeps boot from waiting on the link when the peer is off. NVIDIA's own playbook
instead puts **both rails of a port on the same subnet** — both approaches work; pick one and be
consistent.

### 2.3 Verify the fabric

```bash
# link is up at 200G, DAC cable
ethtool enp1s0f0np0 | grep -E "Speed|Duplex|Link detected"   # Speed: 200000Mb/s, Link detected: yes
# IP reachability + jumbo frames end-to-end (DF-bit, 8972B payload for MTU 9000)
ping -c3 -M do -s 8972 "$WORKER_RAIL0_IP"                     # 0% loss (jumbo frames)
# RDMA loopback bandwidth (install perftest); server on peer, client here:
#   peer:  ib_write_bw -d roceP2p1s0f0 -F -R -q 4 --report_gbits
#   here:  ib_write_bw -d roceP2p1s0f0 -F -R -q 4 --report_gbits "$WORKER_RAIL1_IP"
```

### 2.4 What to expect — the PCIe ceiling

Each ConnectX-7 port attaches to the host over **PCIe Gen5 x4** (`lspci -vv … LnkSta: Speed 32GT/s,
Width x4`), which caps a single rail at **~112 Gb/s**, *not* the 200 Gb/s Ethernet line rate.
Measured on this link:

| test | throughput |
| --- | --- |
| TCP, MTU 1500, 8 streams | ~107 Gb/s (many retransmits) |
| TCP, MTU 9000, 16 streams | ~111 Gb/s (0 retransmits) |
| RDMA (RoCEv2), 1 rail | ~112 Gb/s (near-zero CPU) |
| **RDMA, both rails (multi-rail)** | **~198 Gb/s** |

RDMA's win here isn't more Gb/s per rail — it's the same ~112 Gb/s at **near-zero CPU**, which is
what matters for the per-token TP all-reduce. Use **both rails** for the full ~200 G class.

### 2.5 NCCL / RDMA environment

Point NCCL at both RoCE HCAs and the RoCEv2 GID. (GID index 3 is the RoCEv2 IPv4 GID on these
cards — verify with `cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/3` → `RoCE v2`.)

```bash
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0     # both RoCE HCAs → multi-rail
export NCCL_IB_GID_INDEX=3                       # RoCEv2 IPv4 GID
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0   # OOB bootstrap over the ConnectX links
export GLOO_SOCKET_IFNAME=enp1s0f0np0
```

> **The single biggest RoCE gotcha (in containers):** setting these env vars is **not enough**. The
> container must also have the **RDMA verbs devices** mapped in, or NCCL silently falls back to TCP
> sockets. See [4.3](#43-critical-gotchas).

---

## 3. Why vLLM (and why Nemotron-Super + TP=2)

### Engine: vLLM vs TensorRT-LLM vs SGLang

| | vLLM | TensorRT-LLM | SGLang |
| --- | --- | --- | --- |
| Approach | Python runtime, no build | ahead-of-time **engine compile** | Python runtime |
| Ease on GB10 `sm_121` | **easy** (official image) | hard (arm64 engine build) | medium |
| Peak perf | very good | **highest** (NVFP4 on Blackwell) | strong on MoE/prefix |
| Flexibility | high (swap models instantly) | low (recompile per change) | high |
| Killer feature | PagedAttention, huge ecosystem | TensorRT kernels + quant | RadixAttention (prefix reuse) |

We chose **vLLM** because:
- It has an **official, `sm_121`-tested image** (`vllm/vllm-openai:cu130-nightly`) — no source build.
- NVFP4 auto-detects the right FlashInfer-CUTLASS kernels on GB10.
- It exposes an OpenAI-compatible API + Prometheus metrics with zero extra glue.
- **Ollama was not an option**: it can't shard one model across two nodes (no multi-node TP).
- TensorRT-LLM is the "phase 2" performance play (best NVFP4 latency) but the arm64/`sm_121` engine
  build is heavy — revisit once the serving profile is frozen. SGLang is the best *alternative* to
  benchmark for agent/prefix-heavy workloads (RadixAttention).

### Model: Nemotron-3-Super-120B, TP=2

- Nemotron-Super is NVIDIA's own **NVFP4** MoE (12.7B active) — a documented, first-class DGX Spark
  target with tested vLLM recipes.
- NVFP4 (native Blackwell 4-bit) is a better memory/perf fit here than FP8.
- It **fits on one box**; TP=2 across both boxes buys extra KV/context/concurrency headroom. If you
  only need a single box, run TP=1 and keep the second Spark free.

---

## 4. Deployment (vLLM, TP=2, multi-node, **no Ray**)

The official image does **not** ship Ray. vLLM's native multi-node path uses the **multiprocessing
(`mp`)** executor with a head/worker split — no Ray required.

> **Shortcut:** [`launch-dual-spark.sh`](./launch-dual-spark.sh) in this directory wraps everything
> below — it enforces **identical engine args on every node**, maps the **RDMA devices** in, and
> defaults to the **NCCL-fixed image** (see [4.4](#44-stability--known-issues-dgx-spark)). First
> `./launch-dual-spark.sh build` (once, builds the fixed image on both nodes), then
> `./launch-dual-spark.sh up` (built-in Nemotron-Super recipe), `up -- <serve args>` for a custom
> model, `down`, or `status`. The manual steps below document exactly what it does — but for
> multi-node you **must** use the NCCL-fixed image or it will hang (see 4.4).

### 4.0 Performance & stability flags (why the extra `docker run` bits)

The launch below adds a few DGX-Spark-specific flags beyond the bare minimum. Each earns its place:

| Flag | Why |
| --- | --- |
| `-v ~/.cache/{vllm,flashinfer} ~/.triton ~/.tilelang` | **Persist the compile caches.** Otherwise every relaunch redoes `torch.compile` (~30 s) + CUDA-graph capture + FlashInfer JIT. Mounting them makes restarts near-instant. |
| `--load-format fastsafetensors` | Spark's MMAP is slow; the fastsafetensors loader cuts cold weight-load time. **Caveat:** don't use it if a node's shard exceeds ~0.85 of RAM (risk of OOM) — fine here (~37 GB/node under TP=2). |
| `-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces CUDA allocator fragmentation on GB10 **unified** memory. |
| `--ulimit nofile=1048576:1048576` | Avoids "too many open files" when many shards are opened in parallel. |
| `--ulimit memlock=-1 --cap-add=IPC_LOCK --device=/dev/infiniband/*` | **Required for RoCE** (see [2.5](#25-nccl--rdma-environment) / [4.3](#43-critical-gotchas)). |
| `--log-opt max-size=50m --log-opt max-file=3` | Cap container logs so they can't fill the disk. |

### 4.1 Prep both nodes

```bash
source ./cluster.env                   # hostnames/IPs/images/ports as $VARS

# 1) build the NCCL-fixed image ($IMAGE) on BOTH nodes — see Dockerfile.nccl-fix / §4.4
./launch-dual-spark.sh build           # or per node: docker build -t "$IMAGE" - < Dockerfile.nccl-fix

# 2) authenticate to Hugging Face (gated NVIDIA repo) — token stays local, NOT in git
export HF_TOKEN=...                     # or: hf auth login

# 3) pre-stage weights into each node's HF cache ("download once per node")
docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint hf "$BASE_IMAGE" \
  download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

### 4.2 Launch

Both nodes run the **same engine args**; they differ only in `--node-rank` (0 vs 1), `--headless`
(worker), and `VLLM_HOST_IP` (each node's own rail-0 IP). `--master-addr` is the head's rail-0 IP.
**`source ./cluster.env`** first so the `$VARS`/`$IMAGE` below resolve — or just use
`./launch-dual-spark.sh up`, which does exactly this.

**Head (`spark-a`, rank 0):**

```bash
docker run -d --name vllm-head --network host --gpus all --ipc=host \
  --log-opt max-size=50m --log-opt max-file=3 \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  --device=/dev/infiniband/rdma_cm \
  --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/uverbs1 \
  --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs3 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -v ~/.triton:/root/.triton -v ~/.tilelang:/root/.tilelang \
  -e VLLM_HOST_IP=$HEAD_RAIL0_IP \
  -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=$IB_GID_INDEX -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME=$IB_IFACES -e GLOO_SOCKET_IFNAME=$GLOO_IF \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint vllm "$IMAGE" \
  serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super --trust-remote-code \
    --tensor-parallel-size 2 --distributed-executor-backend mp \
    --nnodes 2 --node-rank 0 --master-addr $MASTER_IP --master-port $MASTER_PORT \
    --max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-seqs 4 \
    --load-format fastsafetensors \
    --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port $SERVE_PORT
```

**Worker (`spark-b`, rank 1) — identical args, but `--node-rank 1 --headless` and `VLLM_HOST_IP=$WORKER_RAIL0_IP`:**

```bash
docker run -d --name vllm-worker --network host --gpus all --ipc=host \
  --log-opt max-size=50m --log-opt max-file=3 \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  --device=/dev/infiniband/rdma_cm \
  --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/uverbs1 \
  --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs3 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -v ~/.triton:/root/.triton -v ~/.tilelang:/root/.tilelang \
  -e VLLM_HOST_IP=$WORKER_RAIL0_IP \
  -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=$IB_GID_INDEX -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME=$IB_IFACES -e GLOO_SOCKET_IFNAME=$GLOO_IF \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint vllm "$IMAGE" \
  serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super --trust-remote-code \
    --tensor-parallel-size 2 --distributed-executor-backend mp \
    --nnodes 2 --node-rank 1 --master-addr $MASTER_IP --master-port $MASTER_PORT \
    --max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-seqs 4 \
    --load-format fastsafetensors \
    --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --headless
```

### 4.3 Critical gotchas

1. **Map the RDMA devices into the container** (`--device=/dev/infiniband/*` + `--cap-add=IPC_LOCK`
   + `--ulimit memlock=-1`). Without them NCCL can't open IB verbs and **silently falls back to TCP
   sockets** — it "works" but with no RDMA. Confirm RoCE is actually used by temporarily adding
   `-e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET` and looking for:
   ```
   NET/IB : Using [0]rocep1s0f0:1/RoCE [1]roceP2p1s0f0:1/RoCE ...
   Using network IB
   ```
   Then **remove `NCCL_DEBUG`** for steady state (it's very verbose and can fill logs — hence the
   `--log-opt` rotation above).
2. **Identical engine args on both nodes.** If scheduling/memory flags (e.g. `--max-num-seqs`,
   `--max-model-len`, `--gpu-memory-utilization`) differ, the two ranks compute **different CUDA-graph
   capture plans** and **deadlock** on the cross-node collective during startup (GPUs pinned ~100%,
   no progress). Pass the same set to both; only `--node-rank` / `--headless` / `--host`/`--port`
   differ.
3. **Ray is not in the official image** — use `--distributed-executor-backend mp` (default when Ray
   is absent). Don't try the Ray path with this image.
4. **JIT / cold start.** First boot loads ~75 GB of weights (a few minutes) + `torch.compile`
   (~30 s) + CUDA-graph capture; the first request triggers extra JIT. Fire a `max_tokens=3` warm-up
   before real traffic. Warm caches make subsequent launches much faster.

### 4.4 Stability & known issues (DGX Spark)

- **NCCL multi-node hang — FIXED here (required).** The official image's pip-bundled NCCL
  (`nvidia-nccl-cu13`, ~219 MB) **hangs multi-node TP on GB10 under sustained load** — rank 0 blocks
  in `ncclAllReduce` while a peer rank spins its GPU at ~96 % until the collective watchdog fires
  ([vllm#42354](https://github.com/vllm-project/vllm/issues/42354)). **We hit this reproducibly.**
  The image also ships the DGX-OS **system NCCL** (~190 MB, `/usr/lib/aarch64-linux-gnu/libnccl.so.2`)
  which does *not* hang (both report 2.28.9 but are different binaries). Fix = point the pip soname
  at the system lib: see [`Dockerfile.nccl-fix`](./Dockerfile.nccl-fix). Build it once with
  `./launch-dual-spark.sh build` (the launcher then runs the fixed `vllm-spark-nccl:cu130` image by
  default). **Verified:** the exact request that hung twice on the stock image ran clean afterward.
- **Driver — pin to 580.x (verified `580.173.02`).** 590.x has a CUDA-graph capture deadlock on
  GB10 unified memory, and 590 packages *are* in the repo, so pin it or `apt`/the DGX Dashboard can
  drift. Drop `/etc/apt/preferences.d/nvidia-pin-580.pref`:
  ```
  Package: *nvidia*
  Pin: version 580.*
  Pin-Priority: 1001

  Package: *nvidia*
  Pin: version 590.*
  Pin-Priority: -1
  ```
  Verify: `apt-cache policy nvidia-driver-590-open` then shows `Candidate: (none)`. Keep both nodes
  identical. **Updating from the CLI** (the shell form of the DGX Dashboard / NVIDIA Sync update):
  `sudo apt update && sudo apt dist-upgrade` (OS + driver), then `sudo fwupdmgr refresh && sudo
  fwupdmgr upgrade` (platform firmware), then `sudo reboot`. NIC firmware here is **28.45.4028**.
- **Firmware sudden-shutdown under heavy inference:** if a node power-cycles under load, cap the GPU
  clock, e.g. `sudo nvidia-smi -lgc 200,2150` (resets on reboot).
- **OOM guard:** consider running `earlyoom` alongside the container for sustained/high-concurrency
  load (guards against the host-RAM creep that can hard-crash the box).

---

## 5. Validate

```bash
source ./cluster.env    # $HEAD_HOST / $SERVE_PORT
# model listed?
curl -s "http://$HEAD_HOST:$SERVE_PORT/v1/models" | python3 -m json.tool

# chat (Nemotron emits a separate reasoning trace via the nemotron_v3 parser)
curl -s "http://$HEAD_HOST:$SERVE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron-3-super","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":256,"temperature":0}'
# -> choices[0].message.content == "4."   (and .reasoning holds the thinking trace)
```

`GPU Direct RDMA Disabled` in the NCCL log is expected without `nvidia-peermem`/dma-buf; it's a
further (separate) optimization and may not help on GB10's unified memory.

---

## 6. Teardown

```bash
docker rm -f vllm-head    # on spark-a
docker rm -f vllm-worker  # on spark-b
```

Weights stay cached in `~/.cache/huggingface`, so relaunching is fast.

---

## Appendix — parameter cheat-sheet

| Flag | Value | Why |
| --- | --- | --- |
| `--tensor-parallel-size` | `2` | shard across the two GPUs/nodes |
| `--distributed-executor-backend` | `mp` | no-Ray multi-node |
| `--nnodes` / `--node-rank` | `2` / `0`,`1` | node topology |
| `--master-addr` / `--master-port` | RoCE IP / `29500` | rendezvous over the fast link |
| `--gpu-memory-utilization` | `0.85` | leave headroom in the unified pool |
| `--max-model-len` | `131072` | context cap (sizes the KV pool) |
| `--max-num-seqs` | `4` | Spark favors small-batch; higher spikes TTFT |
| `--reasoning-parser` | `nemotron_v3` | split reasoning vs content |
| `--tool-call-parser` | `qwen3_coder` | structured tool calls |
| `--kv-cache-dtype` | (auto `fp8`) | model config enables fp8 KV |
| `--load-format` | `fastsafetensors` | faster cold weight load on Spark |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | less allocator fragmentation on unified memory |
| `--ulimit nofile` | `1048576` | avoid "too many open files" for sharded loads |

---

## Credits & further tooling

This recipe is a hand-rolled, "understand-every-flag" walkthrough. For day-to-day use, the community
**[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)** project is worth adopting:
its `launch-cluster.sh` / `run-recipe.sh` **auto-inject identical args to every node** and **expose
`/dev/infiniband` automatically** — i.e. it structurally prevents the two mistakes that cost us the
most here (mismatched per-node args → CUDA-graph deadlock; missing RDMA devices → silent TCP
fallback). It also adds node autodiscovery, parallel model distribution over the IB link
(`hf-download.sh -c --copy-parallel`), the compile-cache mounts, `earlyoom`, and the NCCL soname
fix — and defaults to the same no-Ray multi-node path used above. A `nemotron-3-super-nvfp4` recipe
exists there too (it uses Marlin kernels; we used FlashInfer-CUTLASS, which the newer `cu130-nightly`
handles cleanly).

**Recommendation:** keep this README as the from-scratch reference for *how it works*, but drive
actual launches with that launcher (or copy its patterns) to avoid the foot-guns.
