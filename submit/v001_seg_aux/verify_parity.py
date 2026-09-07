"""End-to-end parity check: container output vs the same ensemble computed locally.

This is the check that catches a preprocessing drift between training and the
shipped container. It must run the *identical* member set and TTA the container
uses -- not the OOF predictions, which by construction come from one held-out
fold model per image.

Also asserts the container actually discriminates: a stack of pure negatives
would let a broken model that always returns 0 pass the I/O validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from model.net import MultiTaskNet  # noqa: E402
from model.predictor import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

logger = logging.getLogger("verify_parity")
REPO_ROOT = Path(__file__).resolve().parents[2]
GC_FRAME_SIZE = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resources", type=Path, default=Path(__file__).parent / "resources")
    ap.add_argument("--manifest", type=Path, default=Path(__file__).parent / "test/test_manifest.csv")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "test/output/interface_0/stacked-neoplastic-lesion-likelihoods.json")
    ap.add_argument("--encoder", default="tu-tf_efficientnetv2_s")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    man = pd.read_csv(args.manifest).sort_values("stack_index").reset_index(drop=True)
    container = np.asarray(json.loads(args.output.read_text()), dtype=np.float64)
    assert len(container) == len(man), f"{len(container)} scores vs {len(man)} manifest rows"

    weights = sorted(args.resources.glob("fold*.pth"))
    logger.info("replaying %d members on %s", len(weights), args.device)
    models = []
    for w in weights:
        m = MultiTaskNet(encoder_name=args.encoder, encoder_weights=None)
        m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True), strict=True)
        models.append(m.to(args.device).eval())

    s = args.image_size
    mean, std = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
    local = []
    with torch.no_grad():
        for i in range(0, len(man), 16):
            batch = []
            for path in man["path"][i : i + 16]:
                im = cv2.cvtColor(cv2.imread(str(REPO_ROOT / path)), cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, (GC_FRAME_SIZE, GC_FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
                im = cv2.resize(im, (s, s), interpolation=cv2.INTER_LINEAR)
                batch.append((im.astype(np.float32) / 255.0 - mean) / std)
            x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(args.device)
            acc = None
            for m in models:
                for v in (x, torch.flip(x, [3]), torch.flip(x, [2])):
                    lg = m.forward_cls(v).float()
                    acc = lg if acc is None else acc + lg
            local.append(torch.sigmoid(acc / (len(models) * 3)).cpu().numpy())
    local = np.concatenate(local).astype(np.float64)

    d = np.abs(local - container)
    logger.info("parity: mean|d|=%.3e  max|d|=%.3e", d.mean(), d.max())

    y = man["label"].to_numpy()
    logger.info("container discrimination on this stack: AUROC=%.4f (n=%d, pos=%d)",
                roc_auc_score(y, container), len(y), int(y.sum()))
    logger.info("  container score range: neg max=%.5f | pos max=%.5f | pos median=%.5f",
                container[y == 0].max(), container[y == 1].max(), np.median(container[y == 1]))

    assert d.max() < 2e-3, f"container/local mismatch: max|d|={d.max():.3e}"
    assert container.std() > 1e-6, "container returned a near-constant score -- not discriminating"
    assert roc_auc_score(y, container) > 0.8, "container ranking is broken"
    logger.info("PARITY OK")


if __name__ == "__main__":
    main()
