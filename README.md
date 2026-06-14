# dgx-spark-recipes

Personal recipes and Docker images for **NVIDIA DGX Spark** (GB10).  
Verified on this machine — **June 2026**.

## DGX Spark software stack

Official release notes:  
https://docs.nvidia.com/dgx/dgx-spark/release-notes.html

Current stack per release notes (June 2026): **DGX OS 7.5.0**, **GPU driver 580.159.03**.

### This machine (`spark-0e81`)

```
Sun Jun 14 13:32:34 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.03             Driver Version: 580.159.03     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
| N/A   38C    P8              3W /  N/A  | Not Supported          |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| GPU | NVIDIA GB10 |
| Driver | 580.159.03 |
| CUDA (driver) | 13.0 |
| Container Toolkit | 1.19.1 |

## Recipes

| Recipe | Description |
|---|---|
| [pytorch-fine-tune](pytorch-fine-tune/) | PyTorch LoRA/SFT fine-tuning image based on NVIDIA playbook |
