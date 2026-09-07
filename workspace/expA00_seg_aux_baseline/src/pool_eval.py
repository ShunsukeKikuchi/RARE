"""Compare ensemble pooling strategies on OOF predictions.

Motivated by the RARE25 winning solution (IMSY), which did NOT average its 40
models: it affine-recalibrated each model to the challenge's 1% prevalence and
then pooled with **noisy-OR**, ``1 - prod(1 - p_m)``. That is a high-recall
aggregation -- if any member fires, the ensemble score is high -- which targets
exactly our failure mode: ~27% of positives sit at p < 0.01, dragging the
90%-recall threshold to ~2e-5 and flooding the result with false positives.

Two things matter methodologically:

* **Noisy-OR is not invariant to monotone rescaling**, unlike a single model's
  PPV@90Recall. So calibration genuinely changes the pooled output -- it is not
  cosmetic here.
* **Calibration must be cross-fitted.** The winners fit on OOF and applied to a
  separate test set. Fitting and evaluating on the same OOF would be optimistic,
  so each fold's parameters are fitted on the *other* folds only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from metrics import evaluate_rare, ppv_at_recall  # noqa: E402

logger = logging.getLogger("pool_eval")
REPO_ROOT = Path(__file__).resolve().parents[3]
EPS = 1e-6
TARGET_PRIORS = [100 / 101, 1 / 101]  # the challenge's simulated 1% prevalence


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_affine(scores: np.ndarray, labels: np.ndarray, priors=TARGET_PRIORS):
    """Affine calibration (temperature + bias) toward ``priors`` -- psrcal's
    AffineCalLogLoss, matching the winning solution."""
    from psrcal.calibration import AffineCalLogLoss, calibrate

    two_col = np.stack([-scores / 2, scores / 2], axis=1)  # logits for [neg, pos]
    _, (t, b) = calibrate(
        trnscores=torch.tensor(two_col, dtype=torch.float64),
        trnlabels=torch.tensor(labels, dtype=torch.long),
        tstscores=torch.tensor(two_col, dtype=torch.float64),
        calclass=AffineCalLogLoss, bias=True, priors=priors, quiet=True,
    )
    return float(t.item()), b.detach().numpy().astype(np.float64)


def apply_affine(scores: np.ndarray, t: float, b: np.ndarray, priors=TARGET_PRIORS) -> np.ndarray:
    """Match the winning solution's application exactly:

        recal = (t * logits + b) + log(prior);  recal -= logsumexp(recal)

    The log-prior term is what actually moves the operating point to 1%
    prevalence -- passing ``priors`` to the fit alone does nothing at apply time
    (fitting yields prior-free log-likelihood ratios). Omitting it inverts the
    intended effect: measured on our OOF, leaving it out *raised* the mean
    negative probability from 0.0014 to 0.156 instead of lowering it.
    """
    two_col = np.stack([-scores / 2, scores / 2], axis=1) * t + b[None, :]
    two_col = two_col + np.log(np.asarray(priors))[None, :]
    two_col = two_col - np.log(np.exp(two_col).sum(axis=1, keepdims=True))
    return two_col[:, 1] - two_col[:, 0]


def crossfit_calibrate(logits: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Calibrate each fold's scores using parameters fitted on the other folds."""
    out = np.empty_like(logits)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        try:
            t, b = fit_affine(logits[tr], labels[tr])
            out[te] = apply_affine(logits[te], t, b)
        except Exception as exc:  # calibration is a nicety; never let it kill the run
            logger.warning("fold %s calibration failed (%s) -- passing through", f, exc)
            out[te] = logits[te]
    return out


def noisy_or(probs: np.ndarray) -> np.ndarray:
    """probs: [n_models, n_samples] -> pooled positive probability."""
    return 1.0 - np.prod(1.0 - np.clip(probs, 0, 1 - 1e-12), axis=0)


def report(name: str, y: np.ndarray, s: np.ndarray, n_iter: int) -> dict:
    ppv, _, _ = evaluate_rare(y, s, n_iter=n_iter)
    m = dict(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s),
             ppv90=ppv, ppv90_full=ppv_at_recall(y, s))
    logger.info("%-42s AUROC=%.4f  AUPRC=%.4f  PPV@90R(1%%)=%.4f  PPV@90R(full)=%.4f",
                name, m["auroc"], m["auprc"], m["ppv90"], m["ppv90_full"])
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path, nargs="+", required=True, help="oof.csv files, one per member")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "pool_eval.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    names = args.names or [p.parent.name.replace("expA00_", "") for p in args.oof]

    dfs = [pd.read_csv(p).sort_values("path").reset_index(drop=True) for p in args.oof]
    base = dfs[0]
    for d in dfs[1:]:
        assert (d["path"].values == base["path"].values).all(), "OOF files cover different images"
    y = base["label"].to_numpy()
    folds = base["fold"].to_numpy()
    logits = np.stack([to_logit(d["prob"].to_numpy()) for d in dfs])       # [M, N]
    cal = np.stack([crossfit_calibrate(l, y, folds) for l in logits])       # [M, N]

    logger.info("members: %s  (n=%d, pos=%d)", names, len(y), int(y.sum()))
    res = {}
    for n, l in zip(names, logits):
        res[f"single/{n}"] = report(f"single  {n}", y, l, args.n_iter)
    logger.info("-" * 110)
    for tag, arr in [("raw", logits), ("calibrated", cal)]:
        res[f"mean-logit/{tag}"] = report(f"mean-logit  ({tag})", y, arr.mean(0), args.n_iter)
        res[f"noisy-or/{tag}"] = report(f"noisy-OR    ({tag})", y, noisy_or(1 / (1 + np.exp(-arr))), args.n_iter)
        res[f"mean-prob/{tag}"] = report(f"mean-prob   ({tag})", y, (1 / (1 + np.exp(-arr))).mean(0), args.n_iter)
        res[f"max-prob/{tag}"] = report(f"max-prob    ({tag})", y, (1 / (1 + np.exp(-arr))).max(0), args.n_iter)

    args.out.write_text(json.dumps(res, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
