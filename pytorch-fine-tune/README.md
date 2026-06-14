# PyTorch Fine-Tune (DGX Spark)

Based on the NVIDIA playbook:  
https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/pytorch-fine-tune

## What this is

`assets/` contains **vendored copies** of the NVIDIA fine-tuning examples (scripts, configs, multi-node helpers), plus small DGX Spark fixes in `spark_hf_utils.py`.

The Docker image bakes in setup that the playbook does manually each time: deps, `bitsandbytes` for CUDA 13.2 aarch64, and the recipe files.

## Problems solved

- **Repeat setup** — no per-run `pip install`, `git clone`, or `hf auth login`
- **CUDA 13.2 + aarch64** — `bitsandbytes` CI wheel with `libbitsandbytes_cuda132.so` (PyPI only had up to cuda130)
- **`torchao` pin** — removed; upstream pin broke `peft` on `26.05-py3`
- **Hugging Face auth** — `HF_TOKEN` passed via `runme.sh`; scripts use `spark_hf_utils` for gated-model checks

## Llama access

**Gated meta-llama models require Hugging Face approval** before training (e.g. [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)). Request access as your HF user, then verify:

```bash
hf download meta-llama/Llama-3.2-3B-Instruct config.json --local-dir /tmp/test
```

Store your token in `~/.config/secrets/hf_token` (not `bashrc`).

## Prerequisites (host)

One-time setup on the DGX Spark machine (playbook steps 1–3):

- **NVIDIA GPU driver** — latest for your OS
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** — enables `docker run --gpus all`
- **Docker** — installed; your user in the `docker` group (`docker ps` works without sudo)
- **Hugging Face token** — in `~/.config/secrets/hf_token` (see Llama access above)

Configure Docker after toolkit install:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Quick start

```bash
docker build -t dgx-spark-pytorch-ft:26.05 .
./runme.sh
python Llama3_3B_full_finetuning.py --model_name meta-llama/Llama-3.2-3B-Instruct --dataset_size 100
```
