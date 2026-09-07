"""Leave-one-center-out domain-shift probe.

Not part of the CV definition: this trains on one center and validates on the
other, in both directions, to measure how much of the CV number survives a
change of site. RARE26's test set spans centers we have never seen, so a large
LOCO drop is the honest early warning that CV is optimistic.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))

from datasets import ExternalSegDataset, RareDataset, build_sampler_weights  # noqa: E402
from lit import LitMultiTask  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from train import NOISY_LOGGERS, setup_logging  # noqa: E402
from transforms import build_transforms  # noqa: E402

logger = logging.getLogger("loco")
REPO_ROOT = Path(__file__).resolve().parents[3]
CENTERS = ("center_1", "center_2")


def build_loaders(cfg: dict, train_center: str, val_center: str):
    folds = pd.read_csv(REPO_ROOT / cfg["data"]["folds_csv"])
    train_df = folds[folds.center == train_center]
    val_df = folds[folds.center == val_center]

    size = cfg["data"]["image_size"]
    pre = cfg["data"].get("gc_prescale")
    train_tf = build_transforms(size, train=True, gc_prescale=pre, preset=cfg["data"].get("aug_preset", "weak"))
    val_tf = build_transforms(size, train=False, gc_prescale=pre)

    sets: list = [RareDataset(train_df, train_tf)]
    labels = [train_df["label"].to_numpy(dtype=np.float32)]
    external = [np.zeros(len(train_df), dtype=bool)]
    keep = [d for d, on in (("evc", cfg["data"]["use_evc"]), ("edd2020", cfg["data"]["use_edd2020"])) if on]
    if keep:
        ext = pd.read_csv(REPO_ROOT / cfg["data"]["external_index"])
        ext = ext[ext.dataset.isin(keep)]
        if len(ext):
            sets.append(ExternalSegDataset(ext, train_tf))
            labels.append(ext["label"].to_numpy(dtype=np.float32))
            external.append(np.ones(len(ext), dtype=bool))

    all_labels = np.concatenate(labels)
    weights = build_sampler_weights(
        all_labels, cfg["data"]["target_pos_ratio"],
        is_external=np.concatenate(external),
        external_fraction=cfg["data"].get("external_fraction", 0.15),
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(all_labels), replacement=True,
    )
    nw = cfg["data"]["num_workers"]
    train_loader = DataLoader(
        ConcatDataset(sets) if len(sets) > 1 else sets[0],
        batch_size=cfg["data"]["batch_size"], sampler=sampler, num_workers=nw,
        pin_memory=True, drop_last=True, persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        RareDataset(val_df, val_tf), batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
    )
    return train_loader, val_loader, val_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    for ov in args.override:
        key, _, raw = ov.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(raw)

    out_root = REPO_ROOT / cfg["experiment"]["results_root"] / cfg["experiment"]["name"]
    setup_logging(out_root)
    for n in NOISY_LOGGERS:
        logging.getLogger(n).setLevel(logging.WARNING)

    results = {}
    for train_center, val_center in ((CENTERS[0], CENTERS[1]), (CENTERS[1], CENTERS[0])):
        tag = f"train_{train_center}__val_{val_center}"
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        logger.info("=== %s ===", tag)

        pl.seed_everything(cfg["experiment"]["seed"], workers=True)
        train_loader, val_loader, val_df = build_loaders(cfg, train_center, val_center)
        module = LitMultiTask(cfg)
        ckpt_cb = ModelCheckpoint(
            dirpath=out_dir, filename="best_model",
            monitor=cfg["eval"]["monitor"], mode=cfg["eval"]["monitor_mode"], save_top_k=1, save_last=True,
        )
        trainer = pl.Trainer(
            default_root_dir=out_dir, max_epochs=cfg["trainer"]["max_epochs"],
            precision=cfg["trainer"]["precision"], gradient_clip_val=cfg["trainer"]["gradient_clip_val"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
            callbacks=[ckpt_cb], logger=False, enable_progress_bar=False,
        )
        trainer.fit(module, train_loader, val_loader)

        best = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")
        results[tag] = {"best_val_auroc": best, "n_val": len(val_df), "n_val_pos": int(val_df["label"].sum())}
        logger.info("%s -> best val AUROC %.4f", tag, best)

    (out_root / "loco_results.json").write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out_root / "loco_results.json")


if __name__ == "__main__":
    main()
