FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime AS rare26-seg-aux

ENV PYTHONUNBUFFERED=1
# Grand Challenge runs the container with --network=none. Make every library fail
# loudly at build/config time rather than silently hanging on a hub lookup.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# opencv-python-headless still needs libgl/libglib on the slim runtime image.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements.txt

# SurgeNetXL's DINOv3 weights need the reference implementation: timm's variant has
# a different register-token count, split qkv bias and a different RoPE table.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git clone --depth 1 https://github.com/facebookresearch/dinov3.git /opt/app/dinov3_repo \
    && chown -R user:user /opt/app/dinov3_repo \
    && apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
USER user

COPY --chown=user:user model /opt/app/model
COPY --chown=user:user inference.py /opt/app/

# Weights are NOT baked in. They ship as the Grand Challenge algorithm model
# tarball, extracted to /opt/ml/model at runtime (see export_model_tarball.sh).
# That keeps this image ~2 GB instead of ~10 GB, so the container can be built and
# validated once while the ensemble is still being updated.

ENTRYPOINT ["python", "inference.py"]
