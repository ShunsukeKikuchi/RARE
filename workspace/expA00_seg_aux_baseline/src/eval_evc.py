"""Evaluate a trained fold ensemble on EVC_Barretts_FullSet as an external test set.

EVC is held out of training entirely (see run.sh): 100 images, 50 ACHD (neoplasia)
and 50 NDBT, from a third site (TU/e) at 1600x1200. That makes it a genuine
domain-shift probe, complementary to leave-one-center-out -- and it is what the
RARE25 winners used it for too.

Because EVC is balanced 50/50 rather than 1% prevalence, PPV@90Recall is not
meaningful here; AUROC/AUPRC are the readouts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

from datasets import _read_rgb  # noqa: E402
from lit import LitMultiTask  # noqa: E402
from transforms import build_transforms  # noqa: E402

logger = logging.getLogger("eval_evc")
REPO_ROOT = Path(__file__).resolve().parents[3]


class PathDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths, self.transform = paths, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = _read_rgb(self.paths[i])
        return self.transform(image=img, mask=np.zeros((*img.shape[:2], 2), dtype=np.float32))["image"]


def load_evc(index_csv: Path) -> pd.DataFrame:
    ext = pd.read_csv(index_csv)
    evc = ext[ext.dataset == "evc"].reset_index(drop=True)
    if evc.empty:
        raise RuntimeError(f"no EVC rows in {index_csv}")
    return evc


@torch.no_grad()
def score(exp_dir: Path, folds: list[int], evc: pd.DataFrame, device: str, use_tta: bool,
          ckpt_name: str = "best_model.ckpt") -> np.ndarray:
    """Mean logit over the fold models -- the same aggregation the container uses."""
    acc, n = None, 0
    for fold in folds:
        fd = exp_dir / f"fold{fold}"
        ckpt = fd / ckpt_name
        if not ckpt.exists():
            logger.warning("missing %s", ckpt)
            continue
        cfg = yaml.safe_load((fd / "config.yaml").read_text())
        cfg = {**cfg, "model": {**cfg["model"], "encoder_weights": None}}
        module = LitMultiTask.load_from_checkpoint(ckpt, cfg=cfg, map_location="cpu")
        model = module.eval_model.to(device).eval()

        loader = DataLoader(
            PathDataset([REPO_ROOT / p for p in evc["path"]],
                        build_transforms(cfg["data"]["image_size"], False, cfg["data"].get("gc_prescale"))),
            batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
        )
        out = []
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            for x in loader:
                x = x.to(device, non_blocking=True)
                views = [x, torch.flip(x, [3]), torch.flip(x, [2])] if use_tta else [x]
                out.append(torch.stack([model.forward_cls(v) for v in views]).mean(0).float().cpu().numpy())
        lg = np.concatenate(out)
        acc = lg if acc is None else acc + lg
        n += 1
        logger.info("  fold %d: EVC AUROC %.4f", fold, roc_auc_score(evc["label"], lg))
        del module, model
        torch.cuda.empty_cache()
    if n == 0:
        raise RuntimeError(f"no usable checkpoints under {exp_dir}")
    return acc / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--index", type=Path,
                    default=Path(__file__).parent / "external_index.csv")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--ckpt-name", default="best_model.ckpt",
                    help="which checkpoint per fold: best_model.ckpt (val-selected) or last.ckpt (no selection)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evc = load_evc(args.index)
    y = evc["label"].to_numpy()
    logger.info("EVC external test set: %d images (%d ACHD / %d NDBT)", len(y), int(y.sum()), int((y == 0).sum()))

    results = {}
    for exp_dir in args.exp_dir:
        logger.info("=== %s ===", exp_dir.name)
        # Guard: this evaluation is only honest if EVC was not in the training set.
        cfg0 = yaml.safe_load((exp_dir / f"fold{args.folds[0]}" / "config.yaml").read_text())
        if cfg0["data"].get("use_evc"):
            logger.warning("%s TRAINED ON EVC -- this number is not a held-out result", exp_dir.name)
        logits = score(exp_dir, args.folds, evc, device, use_tta=not args.no_tta, ckpt_name=args.ckpt_name)
        # Keep per-image scores so the difference between variants can be tested.
        suffix = "" if args.ckpt_name == "best_model.ckpt" else "_" + args.ckpt_name.replace(".ckpt", "")
        pred_out = REPO_ROOT / "results" / f"evc_preds_{exp_dir.name}{suffix}.csv"
        evc.assign(logit=logits).to_csv(pred_out, index=False)
        results[exp_dir.name] = {
            "auroc": float(roc_auc_score(y, logits)),
            "auprc": float(average_precision_score(y, logits)),
            "trained_on_evc": bool(cfg0["data"].get("use_evc")),
        }
        logger.info("  ENSEMBLE  EVC AUROC=%.4f  AUPRC=%.4f%s", results[exp_dir.name]["auroc"],
                    results[exp_dir.name]["auprc"], "  (NOT held out!)" if results[exp_dir.name]["trained_on_evc"] else "")

    out = args.out or REPO_ROOT / "results" / "evc_external.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(results)
    out.write_text(json.dumps(prev, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
