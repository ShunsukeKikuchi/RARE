"""Copy trained fold weights out of results/ into resources/ for the container.

Lightning checkpoints carry the optimizer state and the whole hparams blob
(~250 MB each); the container only needs the model tensors, so we strip the
``model.`` prefix and save a plain state_dict. The decoder weights are kept --
they are ~2 MB and keeping the state_dict complete lets the container load with
``strict=True``, which catches an architecture mismatch instead of silently
producing garbage.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

logger = logging.getLogger("export_weights")
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PREFIX = "model."
EMA_PREFIX = "ema.module."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, nargs="+", required=True,
                    help="one or more results/expA00_<variant> dirs; all are ensembled")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "resources")
    ap.add_argument("--clean", action="store_true", help="remove existing fold*.pth first")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for old in args.out.glob("fold*.pth"):
            old.unlink()
            logger.info("removed stale %s", old.name)

    provenance = {"exp_dirs": [str(e) for e in args.exp_dir], "members": {}}
    for exp_dir in args.exp_dir:
        export_one(exp_dir, args.folds, args.out, provenance)

    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    logger.info("wrote %s (%d members)", args.out / "provenance.json", len(provenance["members"]))


def export_one(exp_dir: Path, folds: list[int], out: Path, provenance: dict) -> None:
    # Tag each file with the variant so a mixed-variant ensemble stays traceable;
    # the container globs fold*.pth, so any such name is picked up.
    tag = exp_dir.name.replace("expA00_", "")
    for fold in folds:
        fold_dir = exp_dir / f"fold{fold}"
        ckpt_path = fold_dir / "best_model.ckpt"
        if not ckpt_path.exists():
            logger.warning("missing %s -- skipping", ckpt_path)
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        # Validation selected the checkpoint using the EMA weights, so those are the
        # ones that must ship; falling back to the raw weights would silently
        # deploy a different model than the one the CV score describes.
        prefix = EMA_PREFIX if any(k.startswith(EMA_PREFIX) for k in sd) else RAW_PREFIX
        state = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not state:
            raise RuntimeError(f"no '{prefix}*' tensors in {ckpt_path}")

        dst = out / f"fold_{tag}_{fold}.pth"
        torch.save(state, dst)

        cfg = yaml.safe_load((fold_dir / "config.yaml").read_text())
        log = json.loads((fold_dir / "training_log.json").read_text())
        provenance["members"][dst.name] = {
            "source": str(ckpt_path),
            "weights": "ema" if prefix == EMA_PREFIX else "raw",
            "encoder_name": cfg["model"]["encoder_name"],
            "image_size": cfg["data"]["image_size"],
            "seg_weight": cfg["loss"]["seg_weight"],
            "use_evc": cfg["data"]["use_evc"],
            "best_val_auroc": log["best_score"],
        }
        logger.info("%-14s fold %d: val AUROC %.4f [%s] -> %s (%.1f MB)",
                    tag, fold, log["best_score"], "ema" if prefix == EMA_PREFIX else "raw",
                    dst.name, dst.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
