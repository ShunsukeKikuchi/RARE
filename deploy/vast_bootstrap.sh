#!/usr/bin/env bash
# One-shot setup for a vast.ai box (RTX 3090, sm_86).
#
#   HF_TOKEN=hf_xxx bash vast_bootstrap.sh
#
# Pulls our code + working set from ONE private HF dataset repo, the DINOv3
# reference implementation from GitHub, and SurgeNetXL weights from their origin.
# Third-party weights are never redistributed by us.
set -euo pipefail

WORK=${WORK:-/workspace/RARE}
REPO=${REPO:-negichi/rare26-work}
: "${HF_TOKEN:?set HF_TOKEN}"
export HF_TOKEN WORK REPO

apt-get update -qq && apt-get install -y -qq git curl unzip libgl1 libglib2.0-0 >/dev/null

pip install -q uv huggingface_hub
mkdir -p "$WORK" && cd "$WORK"

uv venv "$WORK/.venv" --python 3.11
export VIRTUAL_ENV="$WORK/.venv" PATH="$WORK/.venv/bin:$PATH"
# 3090 is Ampere: cu128 wheels give bf16 + FlashAttention (both absent on our Turing box).
uv pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -q timm pytorch-lightning albumentations segmentation-models-pytorch \
    SimpleITK opencv-python-headless pandas scikit-learn pyyaml matplotlib scipy tqdm \
    huggingface_hub psrcal

# DINOv3 reference implementation -- required for SurgeNetXL (timm cannot load it:
# different register-token count, fused qkv bias, different RoPE period table).
git clone --depth 1 https://github.com/facebookresearch/dinov3.git "$WORK/dinov3_repo"

python - <<'PY'
import os, tarfile
from pathlib import Path
from huggingface_hub import snapshot_download
work = Path(os.environ["WORK"])
d = Path(snapshot_download(os.environ["REPO"], repo_type="dataset", token=os.environ["HF_TOKEN"]))

with tarfile.open(d / "code.tar") as t: t.extractall(work)
(work / "data/RARE25").mkdir(parents=True, exist_ok=True)
with tarfile.open(d / "rare25.tar") as t: t.extractall(work / "data/RARE25")
with tarfile.open(d / "external.tar") as t: t.extractall(work)   # arcnames already data/BarrettsEsophagus/...

(work / "workspace/fold/v1").mkdir(parents=True, exist_ok=True)
(work / "workspace/fold/v1/folds.csv").write_bytes((d / "meta/folds.csv").read_bytes())
(work / "workspace/expA00_seg_aux_baseline/src/external_index.csv").write_bytes(
    (d / "meta/external_index.csv").read_bytes())
print("data + code ready at", work)
PY

# SurgeNetXL weights from origin (public release of the SurgeNet authors).
mkdir -p "$WORK/weights"
huggingface-cli download TimJaspers/SurgeNetXL --local-dir "$WORK/weights/surgenet" 2>/dev/null || \
  echo "WARN: fetch DINOv3_ViT{b,l}16_size336_SurgeNetXL.pth into $WORK/weights/surgenet manually"

nvidia-smi
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo "bootstrap complete: $WORK"
