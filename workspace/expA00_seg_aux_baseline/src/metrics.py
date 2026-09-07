"""RARE26 evaluation metrics.

Primary challenge metric is PPV@90Recall under a simulated ~1% prevalence
(bootstrap, 1000 iterations, median). Locally we have only 158 positives in
total, so a single bootstrap iteration carries ~5-6 positives -- the statistic
is heavily quantised and variance-dominated.

**Model selection and experiment comparison MUST use AUROC.**
PPV@90Recall is computed and reported as a reference number only.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

TARGET_RECALL = 0.90
N_BOOTSTRAP = 1000
POS_RATIO = 0.01


def ppv_at_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> float:
    """Best precision achievable at recall >= ``target_recall``.

    Returns 0.0 when the target recall is unreachable.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    candidates = np.where(recall >= target_recall)[0]
    if len(candidates) == 0:
        return 0.0
    return float(np.max(precision[candidates]))


def ppv_at_recall_interp(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> float:
    """Precision linearly interpolated onto recall == ``target_recall``.

    Last year's implementation (``/data4/src/shunsuke/RARE/mim/metrics.py``) used
    this form, while ``ppv_at_recall`` above uses the max-precision reading. They
    disagree slightly and we do not know which one the organisers run, so both are
    reported. Neither drives any decision -- AUROC does.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(np.interp(target_recall, recall[::-1], precision[::-1]))


def evaluate_rare(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_iter: int = N_BOOTSTRAP,
    pos_ratio: float = POS_RATIO,
    target_recall: float = TARGET_RECALL,
    seed: int | None = 42,
) -> tuple[float, np.ndarray, float]:
    """Median PPV@90Recall over ``n_iter`` prevalence-matched bootstrap draws.

    Reproduces the official protocol: keep every negative, then resample
    positives *with replacement* until they make up ``pos_ratio`` of the
    negatives, and take the median PPV@90Recall across iterations.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)

    neg_idx = np.where(y_true == 0)[0]
    pos_idx = np.where(y_true == 1)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return float("nan"), np.array([]), float("nan")

    n_pos_sample = max(1, int(len(neg_idx) * pos_ratio))
    scores = np.empty(n_iter, dtype=np.float64)
    scores_interp = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        sampled_pos = rng.choice(pos_idx, size=n_pos_sample, replace=True)
        idx = np.concatenate([neg_idx, sampled_pos])
        yt, ys = y_true[idx], y_score[idx]
        scores[i] = ppv_at_recall(yt, ys, target_recall)
        scores_interp[i] = ppv_at_recall_interp(yt, ys, target_recall)
    return float(np.median(scores)), scores, float(np.median(scores_interp))


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_iter: int = N_BOOTSTRAP,
    seed: int | None = 42,
) -> dict[str, float]:
    """AUROC (the decision metric) plus AUPRC and the reference PPV@90Recall."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }
    ppv_median, ppv_all, ppv_interp = evaluate_rare(y_true, y_score, n_iter=n_iter, seed=seed)
    out["ppv_90recall"] = ppv_median
    out["ppv_90recall_interp"] = ppv_interp
    if len(ppv_all):
        out["ppv_90recall_p25"] = float(np.percentile(ppv_all, 25))
        out["ppv_90recall_p75"] = float(np.percentile(ppv_all, 75))
    return out


if __name__ == "__main__":
    # Sanity check: perfect ranking -> PPV ~1.0, random ranking -> PPV ~prevalence.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(2937), np.ones(158)])

    perfect = y.astype(float)
    logger.info("perfect  : %s", compute_metrics(y, perfect, n_iter=200))

    random = rng.random(len(y))
    logger.info("random   : %s", compute_metrics(y, random, n_iter=200))

    noisy = y * 0.5 + rng.random(len(y)) * 0.5
    logger.info("separable: %s", compute_metrics(y, noisy, n_iter=200))
