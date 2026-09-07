"""Generate RARE26 CV fold assignments (version v1).

Design notes -- see ../README.md for the rationale.

* RARE25 filenames are bare UUIDs, so there is **no patient identifier**.
  GroupKFold on patients is therefore impossible; patient-level leakage cannot
  be excluded and is accepted as a known limitation.
* There are only two centers, so center cannot be a CV group either (it would
  cap us at 2 folds). Instead we stratify on the composite ``center|label`` key
  so that every fold keeps both centers at their natural ratio.
* Leave-one-center-out is evaluated separately as a domain-shift probe; it is
  not part of the fold definition.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "RARE25"
CENTERS = ("center_1", "center_2")
CLASS_TO_LABEL = {"ndbe": 0, "neo": 1}


def build_index(data_root: Path) -> pd.DataFrame:
    rows = []
    for center in CENTERS:
        for cls, label in CLASS_TO_LABEL.items():
            d = data_root / center / cls
            if not d.is_dir():
                raise FileNotFoundError(f"missing directory: {d}")
            for p in sorted(d.glob("*.png")):
                rows.append(
                    {
                        "path": str(p.relative_to(REPO_ROOT)),
                        "center": center,
                        "label": label,
                    }
                )
    df = pd.DataFrame(rows)
    df["strat"] = df["center"] + "|" + df["label"].astype(str)
    return df


def assign_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(df, df["strat"])):
        df.loc[df.index[val_idx], "fold"] = fold
    return df


def validate(df: pd.DataFrame, n_splits: int) -> None:
    assert df["fold"].between(0, n_splits - 1).all(), "unassigned rows remain"
    assert df["path"].is_unique, "duplicate paths"
    assert len(df) == 3095, f"expected 3095 images, got {len(df)}"
    assert (df["label"] == 1).sum() == 158, "expected 158 neo images"
    counts = df.groupby("center")["label"].agg(["size", "sum"]).to_dict("index")
    assert counts["center_1"] == {"size": 2279, "sum": 61}, counts["center_1"]
    assert counts["center_2"] == {"size": 816, "sum": 97}, counts["center_2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "folds.csv")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    df = build_index(args.data_root)
    df = assign_folds(df, args.n_splits, args.seed)
    validate(df, args.n_splits)

    logger.info("total: %d images (%d neo / %d ndbe)", len(df), df["label"].sum(), (df["label"] == 0).sum())
    dist = df.pivot_table(index="fold", columns="strat", values="path", aggfunc="count", fill_value=0)
    dist["neo_total"] = df[df["label"] == 1].groupby("fold").size()
    dist["n"] = df.groupby("fold").size()
    for line in dist.to_string().splitlines():
        logger.info("%s", line)

    df[["path", "center", "label", "fold"]].to_csv(args.out, index=False)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
