"""Train on every labelled image, stopping at a fixed epoch.

There is no validation split here, so the endpoint cannot be chosen by early
stopping -- it has to be transplanted from the cross-validated runs. That only
works if the fold runs have a *stable* best epoch, which is why EMA matters:
without it the best epoch scattered over 4..25 across folds and no single
endpoint was defensible.

One run snapshots every candidate endpoint (``--snapshot-epochs``) so different
stopping points can be exported and compared without retraining.

Note the epoch budget must be rescaled: with all five folds in the training set
an epoch contains ~25% more optimizer steps than a CV epoch, so matching *steps*
rather than epochs is the honest transplant. ``--match-cv-steps`` does that.
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
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))

from datasets import ExternalSegDataset, RareDataset, build_sampler_weights  # noqa: E402
from lit import LitMultiTask  # noqa: E402
from train import NOISY_LOGGERS, setup_logging  # noqa: E402
from transforms import build_transforms  # noqa: E402

logger = logging.getLogger("train_all")
REPO_ROOT = Path(__file__).resolve().parents[3]
N_FOLDS = 5


class SnapshotAtEpochs(Callback):
    """Save a checkpoint at each requested epoch (1-indexed, end of that epoch)."""

    def __init__(self, epochs: set[int], out_dir: Path):
        self.epochs = epochs
        self.out_dir = out_dir

    def on_train_epoch_end(self, trainer, pl_module):
        done = trainer.current_epoch + 1
        if done in self.epochs:
            path = self.out_dir / f"epoch{done:02d}.ckpt"
            trainer.save_checkpoint(path)
            logger.info("snapshot epoch %d -> %s", done, path.name)


def build_loader(cfg: dict) -> tuple[DataLoader, dict]:
    folds = pd.read_csv(REPO_ROOT / cfg["data"]["folds_csv"])
    size = cfg["data"]["image_size"]
    tf = build_transforms(size, train=True, gc_prescale=cfg["data"].get("gc_prescale"), preset=cfg["data"].get("aug_preset", "weak"))

    sets: list = [RareDataset(folds, tf)]
    labels = [folds["label"].to_numpy(dtype=np.float32)]
    external = [np.zeros(len(folds), dtype=bool)]
    stats = {"rare25": len(folds)}

    keep = [d for d, on in (("evc", cfg["data"]["use_evc"]), ("edd2020", cfg["data"]["use_edd2020"])) if on]
    if keep:
        ext = pd.read_csv(REPO_ROOT / cfg["data"]["external_index"])
        ext = ext[ext.dataset.isin(keep)]
        if len(ext):
            sets.append(ExternalSegDataset(ext, tf))
            labels.append(ext["label"].to_numpy(dtype=np.float32))
            external.append(np.ones(len(ext), dtype=bool))
            for ds, g in ext.groupby("dataset"):
                stats[ds] = len(g)

    all_labels = np.concatenate(labels)
    weights = build_sampler_weights(
        all_labels, cfg["data"]["target_pos_ratio"],
        is_external=np.concatenate(external),
        external_fraction=cfg["data"].get("external_fraction", 0.15),
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(all_labels), replacement=True
    )
    stats["total"] = len(all_labels)
    stats["pos"] = int(all_labels.sum())

    nw = cfg["data"]["num_workers"]
    loader = DataLoader(
        ConcatDataset(sets) if len(sets) > 1 else sets[0],
        batch_size=cfg["data"]["batch_size"], sampler=sampler, num_workers=nw,
        pin_memory=True, drop_last=True, persistent_workers=nw > 0,
    )
    return loader, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--epochs", type=int, required=True, help="stopping epoch (from the CV best epoch)")
    ap.add_argument("--snapshot-epochs", type=int, nargs="*", default=[],
                    help="extra epochs to checkpoint, so endpoints can be compared without retraining")
    ap.add_argument("--match-cv-steps", action="store_true",
                    help="shrink the epoch count so total optimizer steps match a CV run's")
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

    epochs = args.epochs
    if args.match_cv_steps:
        # A CV run saw (N_FOLDS-1)/N_FOLDS of the challenge data per epoch.
        scaled = max(1, round(epochs * (N_FOLDS - 1) / N_FOLDS))
        logger.info("match-cv-steps: %d -> %d epochs", epochs, scaled)
        epochs = scaled

    out_dir = REPO_ROOT / cfg["experiment"]["results_root"] / cfg["experiment"]["name"] / "all"
    setup_logging(out_dir)
    for n in NOISY_LOGGERS:
        logging.getLogger(n).setLevel(logging.WARNING)
    logger.info("output dir: %s", out_dir)
    logger.info("training on ALL data for %d epochs (no validation split)", epochs)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    pl.seed_everything(cfg["experiment"]["seed"], workers=True)
    loader, stats = build_loader(cfg)
    logger.info("data: %s", stats)

    module = LitMultiTask(cfg)
    if module.ema is None:
        logger.warning("EMA is disabled -- the transplanted endpoint was derived with EMA on")

    snaps = {e for e in [*args.snapshot_epochs, epochs] if 0 < e <= epochs}
    trainer = pl.Trainer(
        default_root_dir=out_dir,
        max_epochs=epochs,
        precision=cfg["trainer"]["precision"],
        accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
        gradient_clip_val=cfg["trainer"]["gradient_clip_val"],
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[SnapshotAtEpochs(snaps, out_dir), LearningRateMonitor(logging_interval="epoch")],
        logger=CSVLogger(save_dir=out_dir, name="", version=""),
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.fit(module, loader)

    (out_dir / "training_log.json").write_text(json.dumps(
        {"epochs": epochs, "requested_epochs": args.epochs, "match_cv_steps": args.match_cv_steps,
         "snapshots": sorted(snaps), "data": stats,
         "ema_decay": cfg.get("train", {}).get("ema_decay", 0.0)}, indent=2))
    logger.info("done; snapshots: %s", sorted(snaps))


if __name__ == "__main__":
    main()
