# Reusable image for NVIDIA DGX Spark PyTorch fine-tuning.
# Based on NVIDIA dgx-spark-playbooks pytorch-fine-tune (assets vendored in ./assets).
#
# Host setup (playbook steps 1-3) is still required once on the machine:
# - docker group membership
# - NVIDIA Container Toolkit
# - docker run --gpus all ...

FROM nvcr.io/nvidia/pytorch:26.05-py3

ARG ASSETS_DIR=/opt/pytorch-fine-tune/assets

# Step 4: install fine-tuning dependencies.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# bitsandbytes PyPI wheels only ship up to cuda130 on aarch64; this image uses CUDA 13.2.
# Install the CI aarch64 wheel that includes libbitsandbytes_cuda132.so.
ARG BNB_WHEEL_URL=https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_aarch64.whl
RUN pip install --no-cache-dir --force-reinstall --no-deps "${BNB_WHEEL_URL}" \
    && python -c "import bitsandbytes as bnb; from bitsandbytes.cextension import lib; assert lib is not None; print('bitsandbytes', bnb.__version__, 'cuda132 ok')"

# Step 6: vendored fine-tuning recipes (from NVIDIA/dgx-spark-playbooks).
COPY assets/ ${ASSETS_DIR}/

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chmod +x ${ASSETS_DIR}/pytorch-ft-entrypoint.sh \
    && chmod +x ${ASSETS_DIR}/run-multi-llama_*

ENV ASSETS_DIR="${ASSETS_DIR}"
WORKDIR ${ASSETS_DIR}

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
