"""RARE26 Grand Challenge algorithm entry point.

Source: workspace/expA00_seg_aux_baseline/ (seg-aux multi-task classifier)
See SUBMISSIONS.md for the model provenance and CV scores of the shipped folds.

I/O contract (verified against last year's real example stack):
  in : /input/inputs.json  -- socket slug "stacked-barretts-esophagus-endoscopy-images"
       /input/images/stacked-barretts-esophagus-endoscopy/*.{tif,tiff,mha}
       one file holding N frames stacked as (N, H, W, C)
  out: /output/stacked-neoplastic-lesion-likelihoods.json
       a flat JSON list of N floats in [0, 1], in input order
"""

from __future__ import annotations

import json
import logging
import sys
from glob import glob
from pathlib import Path

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODELS_PATH = Path("/opt/ml/model")
RESOURCE_PATH = Path("/opt/app/resources")

# --- inference parameters (single place to change) ---------------------------
# Architecture / input size / calibration now live in the model tarball manifest.
# A10G has 22.6 GB and the try-out run used 2.5% of it at 0% GPU utilisation -- the
# batch was far too small to keep the card busy. Peak VRAM measured locally at batch 32
# was 3.2 GB, so 128 leaves a wide margin while cutting kernel-launch overhead across
# the 30 members. Batch size does not change any prediction; frames are independent.
BATCH_SIZE = 128
CHUNK_SIZE = 256      # frames decoded per streaming read
USE_TTA = True        # identity + hflip + vflip
# -----------------------------------------------------------------------------

INPUT_SLUG = "stacked-barretts-esophagus-endoscopy-images"
INPUT_RELATIVE_PATH = "images/stacked-barretts-esophagus-endoscopy"
OUTPUT_FILENAME = "stacked-neoplastic-lesion-likelihoods.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger("inference")


def run() -> int:
    handler = {(INPUT_SLUG,): interface_0_handler}[get_interface_key()]
    return handler()


def interface_0_handler() -> int:
    _show_torch_cuda_info()
    from model.predictor import Ensemble

    input_file = find_input_file(INPUT_PATH / INPUT_RELATIVE_PATH)
    # Architecture, input size and calibration are read from manifest.json in the
    # model tarball, so the ensemble can change without rebuilding this image.
    ensemble = Ensemble(models_root=RESOURCE_PATH, batch_size=BATCH_SIZE, use_tta=USE_TTA)
    probs = ensemble.predict_file(input_file, chunk=CHUNK_SIZE)

    if not all(0.0 <= p <= 1.0 for p in probs):
        raise ValueError("likelihoods must lie in [0, 1]")
    logger.info("writing %d likelihoods (min %.4f, max %.4f)", len(probs), min(probs), max(probs))
    # A stack whose every frame scores the same means the weights did not load; a stack
    # of ONE frame trivially satisfies that and must not trip the check. Grand Challenge's
    # try-out sends a single 1024x1024 image, which is exactly that case.
    if len(probs) > 1 and len(set(round(p, 6) for p in probs)) <= 1:
        raise RuntimeError(f"all {len(probs)} likelihoods identical -- weights almost certainly failed to load")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    write_json_file(location=OUTPUT_PATH / OUTPUT_FILENAME, content=probs)
    return 0


def find_input_file(location: Path) -> Path:
    files = sorted(
        glob(str(location / "*.tif")) + glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    )
    if not files:
        raise FileNotFoundError(f"no stacked image found under {location}")
    if len(files) > 1:
        logger.warning("expected one stacked file, found %d: %s -- using the first", len(files), files)
    return Path(files[0])


def get_interface_key() -> tuple[str, ...]:
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["interface"]["slug"] for sv in inputs))


def load_json_file(*, location: Path):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location: Path, content) -> None:
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def _show_torch_cuda_info() -> None:
    import torch

    logger.info("torch %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("device: %s", torch.cuda.get_device_properties(torch.cuda.current_device()))


if __name__ == "__main__":
    raise SystemExit(run())
