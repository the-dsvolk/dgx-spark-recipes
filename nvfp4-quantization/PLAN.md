# Qwen BF16 vs FP4 vs custom NVFP4 — testing plan

## 1. Pick the Qwen variant

**Default candidate:** `Qwen3-8B` — fits single Spark, supported in the [TRT-LLM model matrix](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/trt-llm).

| Role | HF handle (candidate) | Notes |
|------|----------------------|-------|
| BF16 baseline | `Qwen/Qwen3-8B` | Confirm arch + license; use same tokenizer for all runs |
| NVIDIA FP4 | `nvidia/Qwen3-8B-FP4` | Pre-quantized NVFP4 from NVIDIA |
| Custom NVFP4 | `./output_models/saved_models_Qwen3-8B_nvfp4_hf/` | Produced by Model Optimizer (step 3) |

**Selection checklist**

- [ ] Verify `trtllm-serve` loads BF16 checkpoint on Spark (quickstart)
- [ ] Confirm HF access for all handles (`HF_TOKEN`)
- [ ] Lock one variant (base vs instruct) — do not mix across the three runs
- [ ] If 8B OOMs, fall back to a smaller Qwen; if headroom allows, try `Qwen3-14B`

## 2. Environment

Host (once):

- Docker + NVIDIA Container Toolkit (`docker run --gpus all … nvidia-smi`)
- `~/.config/secrets/hf_token`
- `mkdir -p output_models ~/.cache/huggingface`

Fixed container image for all phases:

```bash
export DOCKER_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
```

Fixed docker flags (all runs):

```bash
--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
-v "$HOME/.cache/huggingface:/root/.cache/huggingface"
-e HF_TOKEN
```

## 3. Prepare model artifacts

### 3a. BF16 — download only

```bash
hf download Qwen/Qwen3-8B --local-dir ./models/qwen3-8b-bf16
```

### 3b. FP4 — download only

```bash
hf download nvidia/Qwen3-8B-FP4 --local-dir ./models/qwen3-8b-fp4
```

### 3c. Custom NVFP4 — quantize from BF16

Follow [nvfp4-quantization playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/nvfp4-quantization) Step 4, substituting:

```bash
--model 'Qwen/Qwen3-8B' --quant nvfp4 --tp 1 --export_fmt hf
```

Validate output with `ls ./output_models/` and a quick `quickstart_advanced.py` smoke test.

## 4. Serve each variant (same config)

Use `trtllm-serve` from the [trt-llm playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/trt-llm). **Hold constant** across all three runs:

| Parameter | Value |
|-----------|-------|
| `--backend` | `pytorch` |
| `--max_batch_size` | `4` (or `1` if memory-bound) |
| `--port` | `8000` |
| `tp_size` | `1` |
| KV cache fraction | same for all |

Enable Prometheus metrics for AIPerf — add `return_perf_metrics: true` in `extra_llm_api_options.yaml` ([AIPerf TRT-LLM docs](https://docs.nvidia.com/aiperf/server-metrics/ai-perf-server-metrics-reference)).

Run order (one server at a time):

1. BF16 → serve → benchmark → stop
2. FP4 → serve → benchmark → stop
3. Custom NVFP4 → serve → benchmark → stop

## 5. Benchmark with AIPerf

Install AIPerf on host or in a sidecar container ([docs](https://docs.nvidia.com/aiperf/getting-started/ai-perf-comprehensive-llm-benchmarking)).

**Fixed workload** (same for all three models):

| Setting | Value |
|---------|-------|
| Input dataset | Shared JSON / trace file |
| Concurrency levels | e.g. `1, 4, 8` |
| Streaming | on |
| Request profile | fixed ISL/OSL (e.g. 512 in / 128 out) |

Example profile command:

```bash
aiperf profile \
  -m Qwen/Qwen3-8B \
  --url http://localhost:8000 \
  --streaming \
  --input-file ./benchmarks/prompts.json \
  --concurrency 4 \
  --artifact-dir ./results/<variant>-c4
```

Repeat per variant and concurrency; name artifacts `bf16`, `fp4-nvidia`, `nvfp4-custom`.

## 6. Compare results

Build a summary table from AIPerf artifacts:

| Metric | BF16 | FP4 (NVIDIA) | NVFP4 (custom) |
|--------|------|--------------|----------------|
| TTFT p50 / p99 | | | |
| TPOT p50 / p99 | | | |
| Throughput (tok/s) | | | |
| E2E latency p99 | | | |
| Peak GPU memory | | | |
| Model size on disk | | | |

**Optional (accuracy):** run the same prompt set through all three; note quality deltas. Quantization playbooks cite &lt;1% accuracy loss vs FP8 for NVFP4 — verify on your prompts.

## 7. Deliverables for this repo

- [ ] `runme.sh` — docker wrapper (HF token, mounts, flags)
- [ ] `quantize.sh` — Model Optimizer one-liner for Qwen
- [ ] `serve.sh` — `trtllm-serve` with pinned args
- [ ] `benchmark.sh` — AIPerf profile loop
- [ ] `results/` — artifact dirs + summary markdown

## 8. Risks

| Risk | Mitigation |
|------|------------|
| UMA memory pressure on Spark | `sync; echo 3 > /proc/sys/vm/drop_caches`; reduce batch size |
| BF16 unsupported in TRT-LLM for Qwen3 | Confirm in quickstart before full benchmark plan |
| AIPerf / `trtllm-serve` metrics mismatch | Use `/prometheus/metrics` + `return_perf_metrics: true` |
| Custom NVFP4 quality regression | Spot-check generations before perf-only conclusion |

## 9. Timeline (estimate)

| Phase | Duration |
|-------|----------|
| Model selection + smoke tests | 1–2 h |
| Downloads + custom quantization | 1–2 h |
| Three serve + AIPerf runs | 2–3 h |
| Results summary | 30 min |
