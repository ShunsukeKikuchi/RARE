"""Produce out-of-fold predictions for a trained experiment.

Each fold's ``best_model.ckpt`` scores its own held-out fold, so concatenating
them gives one prediction per RARE25 image. TTA matches what the submission
container does, so the OOF number is comparable to what ships.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from datasets import RareDataset  # noqa: E402
from lit import LitMultiTask  # noqa: E402
from transforms import build_transforms  # noqa: E402

logger = logging.getLogger("predict_oof")
REPO_ROOT = Path(__file__).resolve().parents[3]


def tta_logits(model, x: torch.Tensor, use_tta: bool) -> torch.Tensor:
    """Mean logit over identity / hflip / vflip. Classification head only."""
    views = [x]
    if use_tta:
        views += [torch.flip(x, dims=[3]), torch.flip(x, dims=[2])]
    return torch.stack([model.forward_cls(v) for v in views]).mean(0)


@torch.no_grad()
def predict_fold(ckpt: Path, df: pd.DataFrame, cfg: dict, device: str, use_tta: bool) -> np.ndarray:
    # The checkpoint overwrites every weight anyway; skip the ImageNet download.
    cfg = {**cfg, "model": {**cfg["model"], "encoder_weights": None}}
    module = LitMultiTask.load_from_checkpoint(ckpt, cfg=cfg, map_location="cpu")
    # Validation and checkpoint selection ran on the EMA weights, so scoring must too.
    model = module.eval_model.to(device).eval()
    logger.info("fold model: %s weights", "EMA" if module.ema is not None else "raw")

    loader = DataLoader(
        RareDataset(df, build_transforms(cfg["data"]["image_size"], train=False,
                                         gc_prescale=cfg["data"].get("gc_prescale"))),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    out = []
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        for batch in loader:
            logits = tta_logits(model, batch["image"].to(device, non_blocking=True), use_tta)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, required=True, help="results/{experiment_name}")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for fold in args.folds:
        fold_dir = args.exp_dir / f"fold{fold}"
        ckpt = fold_dir / "best_model.ckpt"
        if not ckpt.exists():
            logger.warning("missing %s -- skipping fold %d", ckpt, fold)
            continue
        cfg = yaml.safe_load((fold_dir / "config.yaml").read_text())
        folds_df = pd.read_csv(REPO_ROOT / cfg["data"]["folds_csv"])
        val_df = folds_df[folds_df.fold == fold].reset_index(drop=True)

        probs = predict_fold(ckpt, val_df, cfg, device, use_tta=not args.no_tta)
        val_df = val_df.assign(prob=probs)
        rows.append(val_df)
        logger.info("fold %d: %d predictions, mean prob %.4f", fold, len(val_df), probs.mean())

    if not rows:
        raise SystemExit("no folds predicted")
    oof = pd.concat(rows, ignore_index=True)
    out = args.out or (args.exp_dir / "oof.csv")
    oof.to_csv(out, index=False)
    logger.info("wrote %s (%d rows)", out, len(oof))


if __name__ == "__main__":
    main()
