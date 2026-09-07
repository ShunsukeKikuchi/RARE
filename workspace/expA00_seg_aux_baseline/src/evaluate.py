"""Score OOF predictions.

AUROC is the decision metric. PPV@90Recall (1000-iteration bootstrap at 1%
prevalence, median) is printed alongside as the official-metric reference, in
both the max-precision and the linear-interpolation reading -- with 158 positives
in total it is far too noisy to choose a model with.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from metrics import N_BOOTSTRAP, compute_metrics  # noqa: E402

logger = logging.getLogger("evaluate")

COLS = ("auroc", "auprc", "ppv_90recall", "ppv_90recall_interp")


def report(name: str, df: pd.DataFrame, n_iter: int) -> dict | None:
    if df["label"].nunique() < 2:
        logger.warning("%-22s skipped (single class)", name)
        return None
    m = compute_metrics(df["label"].to_numpy(), df["prob"].to_numpy(), n_iter=n_iter)
    logger.info(
        "%-22s n=%-5d pos=%-4d AUROC=%.4f  AUPRC=%.4f  PPV@90R=%.4f (interp %.4f)",
        name, len(df), int(df["label"].sum()), m["auroc"], m["auprc"], m["ppv_90recall"], m["ppv_90recall_interp"],
    )
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--n-iter", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    oof = pd.read_csv(args.oof)

    results = {"overall": report("OVERALL", oof, args.n_iter)}
    logger.info("-" * 100)
    for center, g in oof.groupby("center"):
        results[center] = report(center, g, args.n_iter)
    logger.info("-" * 100)
    for fold, g in oof.groupby("fold"):
        results[f"fold{fold}"] = report(f"fold{fold}", g, args.n_iter)

    out = args.out or args.oof.with_name("oof_metrics.json")
    out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out)
    logger.info("decision metric = OVERALL AUROC = %.4f", results["overall"]["auroc"])


if __name__ == "__main__":
    main()
