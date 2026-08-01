# Dual DGX Spark — Nemotron-3-Super-120B (NVFP4) over RoCE with vLLM (TP=2)

Serve NVIDIA **Nemotron-3-Super-120B-A12B (NVFP4)** across **two DGX Spark (GB10)** boxes with
tensor parallelism (**TP=2**) on **vLLM**, using the direct **ConnectX-7 200 GbE / RoCE** link
between the nodes. OpenAI-compatible endpoint, ~131K context.

This recipe is written in the order we actually built it up:

1. **SSH** between the two nodes (passwordless).
2. **RoCE / ConnectX-7** — the interconnect and how to make NCCL actually use it.
3. **Why vLLM** (vs TensorRT-LLM / SGLang, and why not Ollama) — and why Nemotron-Super + TP=2.
4. **Deployment** — the exact multi-node launch, the gotchas, validation, and teardown.

> Conventions: the two nodes are called **`spark-a`** (head, rank 0) and **`spark-b`**
> (worker, rank 1). Substitute your own hostnames. Example RFC-1918 addresses are used throughout —
> adapt to your LAN. **No secrets (passwords / tokens / keys) appear in this repo by design.**

---

## Result / what you get

| Property | Value |
| --- | --- |
| Model | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (~120.6B total / 12.7B active, MoE, hybrid Mamba-Transformer) |
| Parallelism | Tensor-parallel **TP=2** across 2 nodes (1 GPU each) |
| Quant | **NVFP4** weights (Blackwell-native), **fp8** KV cache |
| Context | up to **131,072** tokens |
| Interconnect | direct **ConnectX-7 200 GbE** QSFP, **RoCEv2** (no switch) |
| Serving | vLLM OpenAI-compatible API on `spark-a:8000` |
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

From your workstation, both boxes are reachable by their mDNS names (`spark-a.local`,
`spark-b.local`) and log in as the same user on both (a **same username on both nodes** is required
by the NVIDIA multi-node tooling).

Distribute your public key once so subsequent logins need no password:

```bash
# one-time: you'll be prompted for the node's login password ONCE, then never again
ssh-copy-id <user>@spark-a.local
ssh-copy-id <user>@spark-b.local
```

Verify:

```bash
ssh -o BatchMode=yes <user>@spark-a.local 'hostname'   # must NOT prompt for a password
ssh -o BatchMode=yes <user>@spark-b.local 'hostname'
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

```yaml
# spark-a (use .2 on spark-b)
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp1s0f0np0:      # rail 0
      optional: true
      mtu: 9000        # jumbo frames (optional; RoCE MTU is 4096 regardless)
      addresses: [192.168.100.1/24]
    enP2p1s0f0np0:    # rail 1
      optional: true
      mtu: 9000
      addresses: [192.168.101.1/24]
```

`optional: true` keeps boot from waiting on the link when the peer is off. NVIDIA's own playbook
instead puts **both rails of a port on the same subnet** — both approaches work; pick one and be
consistent.

### 2.3 Verify the fabric

```bash
# link is up at 200G, DAC cable
ethtool enp1s0f0np0 | grep -E "Speed|Duplex|Link detected"   # Speed: 200000Mb/s, Link detected: yes
# IP reachability + jumbo frames end-to-end (DF-bit, 8972B payload for MTU 9000)
ping -c3 -M do -s 8972 192.168.100.2                          # 0% loss
# RDMA loopback bandwidth (install perftest); server on peer, client here:
#   peer:  ib_write_bw -d roceP2p1s0f0 -F -R -q 4 --report_gbits
#   here:  ib_write_bw -d roceP2p1s0f0 -F -R -q 4 --report_gbits 192.168.101.2
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
# identical image on both (pin the digest you validated)
docker pull vllm/vllm-openai:cu130-nightly

# authenticate to Hugging Face (needed for the gated NVIDIA repo) — token stays local, NOT in git
export HF_TOKEN=...          # or: hf auth login

# pre-stage weights into each node's HF cache ("download once per node")
docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint hf vllm/vllm-openai:cu130-nightly \
  download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

### 4.2 Launch

Both nodes run the **same engine args**; they differ only in `--node-rank` (0 vs 1) and
`--headless` (worker). `master-addr` is a **RoCE** IP on the head.

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
  -e VLLM_HOST_IP=192.168.100.1 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint vllm vllm/vllm-openai:cu130-nightly \
  serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super --trust-remote-code \
    --tensor-parallel-size 2 --distributed-executor-backend mp \
    --nnodes 2 --node-rank 0 --master-addr 192.168.100.1 --master-port 29500 \
    --max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-seqs 4 \
    --load-format fastsafetensors \
    --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port 8000
```

**Worker (`spark-b`, rank 1) — identical args, `--node-rank 1 --headless`, `VLLM_HOST_IP=192.168.100.2`:**

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
  -e VLLM_HOST_IP=192.168.100.2 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint vllm vllm/vllm-openai:cu130-nightly \
  serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super --trust-remote-code \
    --tensor-parallel-size 2 --distributed-executor-backend mp \
    --nnodes 2 --node-rank 1 --master-addr 192.168.100.1 --master-port 29500 \
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

- **NCCL library load-order (official image).** The official image can ship a pip `libnccl.so.2`
  that shadows the system `libnccl2`, a known cause of **multi-node NCCL hangs** on DGX Spark
  ([vllm#42354](https://github.com/vllm-project/vllm/issues/42354)). We didn't hit it, but if the
  cluster hangs at init, redirect NCCL to the system soname (the `spark-vllm-docker`
  `use-official-vllm` mod does this automatically).
- **Driver:** use **580.x** — 590.x has a CUDA-graph capture deadlock on GB10 unified memory. Keep
  both nodes on the **same** driver version (mismatched minors are best avoided).
- **Firmware sudden-shutdown under heavy inference:** if a node power-cycles under load, cap the GPU
  clock, e.g. `sudo nvidia-smi -lgc 200,2150` (resets on reboot).
- **OOM guard:** consider running `earlyoom` alongside the container for sustained/high-concurrency
  load (guards against the host-RAM creep that can hard-crash the box).

---

## 5. Validate

```bash
# model listed?
curl -s http://spark-a.local:8000/v1/models | python3 -m json.tool

# chat (Nemotron emits a separate reasoning trace via the nemotron_v3 parser)
curl -s http://spark-a.local:8000/v1/chat/completions \
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
