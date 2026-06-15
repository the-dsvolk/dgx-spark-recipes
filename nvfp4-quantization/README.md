# NVFP4 Quantization (DGX Spark)

Work in progress based on NVIDIA playbooks:

- [nvfp4-quantization](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/nvfp4-quantization) — quantize models with TensorRT Model Optimizer
- [trt-llm](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/trt-llm) — serve and benchmark with TensorRT-LLM

Container base: `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev`

## Goal

Benchmark a **Qwen** model on DGX Spark across three precision paths:

| Variant | Description |
|---------|-------------|
| **BF16** | Original full-precision checkpoint |
| **FP4** | NVIDIA pre-quantized NVFP4 checkpoint |
| **NVFP4 (custom)** | BF16 → NVFP4 via Model Optimizer (this recipe) |

Compare inference performance with **[NVIDIA AIPerf](https://docs.nvidia.com/aiperf/)** under identical serving settings.

Exact Qwen version TBD — see [PLAN.md](./PLAN.md).
