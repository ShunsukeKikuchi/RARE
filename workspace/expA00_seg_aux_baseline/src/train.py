"""Train the seg-aux multi-task classifier for one fold.

All outputs land in ``results/{experiment.name}/fold{N}/``: checkpoints, the
timestamped log, ``training_log.json`` and a copy of the resolved config. An
existing directory is never overwritten -- a ``_001`` suffix is added instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))

from datasets import ExternalSegDataset, RareDataset, build_sampler_weights  # noqa: E402
from lit import LitMultiTask  # noqa: E402
from transforms import build_transforms  # noqa: E402

logger = logging.getLogger("train")
REPO_ROOT = Path(__file__).resolve().parents[3]

# PIL logs every TIFF tag at DEBUG; with EDD2020's .tif masks in the loader that
# buries the log file under hundreds of MB of noise.
NOISY_LOGGERS = (
    "PIL", "matplotlib", "fsspec", "urllib3",
    # timm's pretrained-weight download dumps full HTTP headers at DEBUG
    "httpx", "httpcore", "huggingface_hub", "filelock", "requests",
)


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fileh = logging.FileHandler(out_dir / f"train_{stamp}.log")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(fmt)
    root.addHandler(fileh)


def _has_run(d: Path) -> bool:
    """True only if a real run lives here.

    A directory left behind by a crashed launch holds just config.yaml and a stub
    log; treating that as occupied pushed reruns into ``_001`` siblings and split
    one sweep's folds across two directories.
    """
    return d.is_dir() and (
        (d / "training_log.json").exists() or any(d.glob("*.ckpt"))
    )


class SnapshotAtEpochs(Callback):
    """Save a checkpoint at fixed epochs.

    Needed because ``save_last=True`` does NOT give the final epoch -- Lightning
    writes last.ckpt as a copy of the most recently *saved* checkpoint, which with
    ``save_top_k=1`` is the best one. Without this, there is no way to ask how much
    the epoch choice actually costs.
    """

    def __init__(self, epochs: set[int], out_dir: Path):
        self.epochs, self.out_dir = epochs, out_dir

    def on_train_epoch_end(self, trainer, pl_module):
        done = trainer.current_epoch + 1
        if done in self.epochs:
            trainer.save_checkpoint(self.out_dir / f"epoch{done:02d}.ckpt")


def resolve_out_dir(root: Path, name: str, fold: int, resume: bool) -> Path:
    base = root / name / f"fold{fold}"
    if resume or not _has_run(base):
        return base
    i = 1
    while _has_run(cand := root / f"{name}_{i:03d}" / f"fold{fold}"):
        i += 1
    return cand


def build_dataloaders(cfg: dict, fold: int) -> tuple[DataLoader, DataLoader, dict]:
    folds = pd.read_csv(REPO_ROOT / cfg["data"]["folds_csv"])
    train_df = folds[folds.fold != fold]
    val_df = folds[folds.fold == fold]

    size = cfg["data"]["image_size"]
    pre = cfg["data"].get("gc_prescale")
    train_tf = build_transforms(size, train=True, gc_prescale=pre, preset=cfg["data"].get("aug_preset", "weak"))
    val_tf = build_transforms(size, train=False, gc_prescale=pre)

    train_sets: list = [RareDataset(train_df, train_tf)]
    labels = [train_df["label"].to_numpy(dtype=np.float32)]
    external = [np.zeros(len(train_df), dtype=bool)]
    stats = {"rare25_train": len(train_df), "rare25_val": len(val_df)}

    # External pixel-supervised data is train-only and shared by every fold: it has
    # no fold assignment and must never leak into validation.
    keep = [ds for ds, on in (("evc", cfg["data"]["use_evc"]), ("edd2020", cfg["data"]["use_edd2020"])) if on]
    if keep:
        ext = pd.read_csv(REPO_ROOT / cfg["data"]["external_index"])
        ext = ext[ext.dataset.isin(keep)]
        if len(ext):
            train_sets.append(ExternalSegDataset(ext, train_tf))
            labels.append(ext["label"].to_numpy(dtype=np.float32))
            external.append(np.ones(len(ext), dtype=bool))
            for ds, g in ext.groupby("dataset"):
                stats[f"{ds}_train"] = len(g)

    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    all_labels = np.concatenate(labels)
    is_external = np.concatenate(external)
    weights = build_sampler_weights(
        all_labels,
        cfg["data"]["target_pos_ratio"],
        is_external=is_external,
        external_fraction=cfg["data"].get("external_fraction", 0.15),
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(all_labels), replacement=True
    )
    stats["train_total"] = len(all_labels)
    stats["train_pos"] = int(all_labels.sum())
    stats["expected_batch_mix"] = {
        "rare_pos": round(float(weights[(~is_external) & (all_labels == 1)].sum()), 3),
        "rare_neg": round(float(weights[(~is_external) & (all_labels == 0)].sum()), 3),
        "ext_pos": round(float(weights[is_external & (all_labels == 1)].sum()), 3),
        "ext_neg": round(float(weights[is_external & (all_labels == 0)].sum()), 3),
    }

    nw = cfg["data"]["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        sampler=sampler,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        RareDataset(val_df, val_tf),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )
    return train_loader, val_loader, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--resume", action="store_true", help="reuse the existing fold dir and continue from last.ckpt")
    ap.add_argument("--override", nargs="*", default=[], help="dotted.key=value overrides, e.g. loss.seg_weight=0.0")
    ap.add_argument("--snapshot-epochs", type=int, nargs="*", default=[],
                    help="also checkpoint at these epochs (1-indexed), for endpoint studies")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    for ov in args.override:
        key, _, raw = ov.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(raw)

    out_dir = resolve_out_dir(REPO_ROOT / cfg["experiment"]["results_root"], cfg["experiment"]["name"], args.fold, args.resume)
    setup_logging(out_dir)
    logger.info("output dir: %s", out_dir)
    logger.info("config:\n%s", yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    pl.seed_everything(cfg["experiment"]["seed"], workers=True)

    train_loader, val_loader, stats = build_dataloaders(cfg, args.fold)
    logger.info("data: %s", stats)

    module = LitMultiTask(cfg)

    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename="best_model",
        monitor=cfg["eval"]["monitor"],
        mode=cfg["eval"]["monitor_mode"],
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        default_root_dir=out_dir,
        max_epochs=cfg["trainer"]["max_epochs"],
        precision=cfg["trainer"]["precision"],
        accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
        gradient_clip_val=cfg["trainer"]["gradient_clip_val"],
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="epoch")]
        + ([SnapshotAtEpochs(set(args.snapshot_epochs), out_dir)] if args.snapshot_epochs else []),
        logger=CSVLogger(save_dir=out_dir, name="", version=""),
        num_sanity_val_steps=2,
    )

    last_ckpt = out_dir / "last.ckpt"
    resume_from = str(last_ckpt) if (args.resume and last_ckpt.exists()) else None
    if resume_from:
        logger.info("resuming from %s", resume_from)

    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_from)

    summary = {
        "fold": args.fold,
        "data": stats,
        "best_model_path": str(ckpt_cb.best_model_path),
        "best_score": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None,
        "monitor": cfg["eval"]["monitor"],
        "metrics": {k: float(v) for k, v in trainer.callback_metrics.items()},
    }
    (out_dir / "training_log.json").write_text(json.dumps(summary, indent=2))
    logger.info("best %s = %s", cfg["eval"]["monitor"], summary["best_score"])
    logger.info("wrote %s", out_dir / "training_log.json")


if __name__ == "__main__":
    main()
