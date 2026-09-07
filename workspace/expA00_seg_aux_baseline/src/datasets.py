"""Datasets for the multi-task (classification + auxiliary segmentation) model.

Every sample yields the same tuple shape regardless of source:

    image        float tensor [3, H, W]
    label        float scalar (0 = ndbe, 1 = neoplasia)
    mask         float tensor [2, H, W]  -- ch0 = BE extent, ch1 = neoplasia
    mask_weight  float tensor [2]        -- per-channel supervision flag

``mask_weight`` is how RARE25 (image-level labels only) coexists with the
external pixel-supervised sets in one batch: its weight is [0, 0], so the
segmentation loss simply skips those samples. EVC has no BE-extent annotation,
so it contributes [0, 1]. EDD2020 contributes [1, 1].

An all-zero mask with weight 1 is a *real* negative target, not missing data --
that is exactly what EVC's NDBT images and EDD2020's BE-free images provide.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
BE_CHANNEL, NEO_CHANNEL = 0, 1
N_SEG_CHANNELS = 2

cv2.setNumThreads(0)  # avoid thread thrash inside DataLoader workers


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask_mean(paths: list[Path], shape: tuple[int, int]) -> np.ndarray:
    """Mean of several binary annotations -> soft target in [0, 1].

    With a single path this is a plain binarisation; with EVC's five expert
    delineations it becomes a soft consensus that preserves disagreement.
    """
    if not paths:
        return np.zeros(shape, dtype=np.float32)
    acc = np.zeros(shape, dtype=np.float32)
    for p in paths:
        m = np.array(Image.open(p).convert("L"), dtype=np.float32)
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        acc += (m > 127).astype(np.float32)
    return acc / len(paths)


def _read_mask_union(paths: list[Path], shape: tuple[int, int]) -> np.ndarray:
    """Union of several class masks -> binary target."""
    if not paths:
        return np.zeros(shape, dtype=np.float32)
    acc = np.zeros(shape, dtype=np.float32)
    for p in paths:
        m = np.array(Image.open(p).convert("L"), dtype=np.float32)
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        acc = np.maximum(acc, (m > 127).astype(np.float32))
    return acc


class RareDataset(Dataset):
    """RARE25 challenge images: image-level label, no pixel supervision."""

    def __init__(self, df: pd.DataFrame, transform, repo_root: Path = REPO_ROOT):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.repo_root = repo_root
        self.labels = self.df["label"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _read_rgb(self.repo_root / row["path"])
        mask = np.zeros((*img.shape[:2], N_SEG_CHANNELS), dtype=np.float32)
        out = self.transform(image=img, mask=mask)
        return {
            "image": out["image"],
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "mask": out["mask"].permute(2, 0, 1).float(),
            "mask_weight": torch.zeros(N_SEG_CHANNELS, dtype=torch.float32),
        }


class ExternalSegDataset(Dataset):
    """EVC / EDD2020: image-level label plus per-channel pixel supervision."""

    def __init__(self, df: pd.DataFrame, transform, repo_root: Path = REPO_ROOT):
        self.df = df.reset_index(drop=True).fillna({"be_masks": "", "neo_masks": ""})
        self.transform = transform
        self.repo_root = repo_root
        self.labels = self.df["label"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _paths(self, cell: str) -> list[Path]:
        return [self.repo_root / p for p in str(cell).split(";") if p]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _read_rgb(self.repo_root / row["path"])
        shape = img.shape[:2]

        mask = np.zeros((*shape, N_SEG_CHANNELS), dtype=np.float32)
        mask[..., BE_CHANNEL] = _read_mask_union(self._paths(row["be_masks"]), shape)
        # EVC ships five expert delineations -> soft consensus; EDD ships one per
        # class -> the mean of a single mask is that mask.
        neo_paths = self._paths(row["neo_masks"])
        if row["dataset"] == "evc":
            mask[..., NEO_CHANNEL] = _read_mask_mean(neo_paths, shape)
        else:
            mask[..., NEO_CHANNEL] = _read_mask_union(neo_paths, shape)

        out = self.transform(image=img, mask=mask)
        return {
            "image": out["image"],
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "mask": out["mask"].permute(2, 0, 1).float(),
            "mask_weight": torch.tensor(
                [float(row["has_be"]), float(row["has_neo"])], dtype=torch.float32
            ),
        }


def _balance(labels: np.ndarray, mask: np.ndarray, share: float, pos_ratio: float | None) -> np.ndarray:
    """Distribute ``share`` of the sampling mass over ``mask``.

    ``pos_ratio`` splits that mass between classes; None keeps the natural ratio.
    """
    w = np.zeros(len(labels), dtype=np.float64)
    n = int(mask.sum())
    if n == 0 or share <= 0:
        return w
    if pos_ratio is None:
        w[mask] = share / n
        return w
    pos, neg = mask & (labels == 1), mask & (labels == 0)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        w[mask] = share / n
        return w
    w[pos] = share * pos_ratio / n_pos
    w[neg] = share * (1.0 - pos_ratio) / n_neg
    return w


def build_sampler_weights(
    labels: np.ndarray,
    target_pos_ratio: float,
    is_external: np.ndarray | None = None,
    external_fraction: float = 0.15,
) -> np.ndarray:
    """Per-sample weights for a WeightedRandomSampler.

    RARE25 is 5.1% positive, so positives have to be upsampled or the classifier
    barely sees a lesion. But upsampling positives *globally* backfires here: EVC
    and EDD2020 are 68% positive between them, so a single global class balance
    would hand external images **66% of every positive the classifier ever sees**
    (measured: RARE pos 0.086 vs external pos 0.164 of each batch). The
    classification head would then be learning "positive" largely from a different
    imaging domain than the one it is scored on.

    So the two sources are budgeted separately: RARE25 keeps
    ``1 - external_fraction`` of the batch with its positives boosted to
    ``target_pos_ratio`` within that share, and the external sets fill the rest at
    their natural class ratio -- enough to put pixel supervision in nearly every
    batch without letting them dominate the classification signal.
    """
    labels = np.asarray(labels)
    if is_external is None:
        return _balance(labels, np.ones(len(labels), dtype=bool), 1.0, target_pos_ratio)

    is_external = np.asarray(is_external, dtype=bool)
    if not is_external.any():
        return _balance(labels, ~is_external, 1.0, target_pos_ratio)

    w = _balance(labels, ~is_external, 1.0 - external_fraction, target_pos_ratio)
    w += _balance(labels, is_external, external_fraction, None)
    return w
