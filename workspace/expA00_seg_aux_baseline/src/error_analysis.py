"""Contact sheets of the worst OOF errors, for eyeballing before tuning anything.

Writes three grids next to the OOF csv:
  fp_top.png   highest-scoring negatives  -- what the model hallucinates lesions on
  fn_top.png   lowest-scoring positives   -- lesions it misses outright
  borderline.png  positives and negatives nearest the 90%-recall threshold
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from metrics import TARGET_RECALL  # noqa: E402

logger = logging.getLogger("error_analysis")
REPO_ROOT = Path(__file__).resolve().parents[3]

TILE = 256
N_COLS = 5


def threshold_at_recall(df: pd.DataFrame, target: float = TARGET_RECALL) -> float:
    """Score threshold that yields >= ``target`` recall -- the operating point the
    challenge metric is read off."""
    pos = np.sort(df.loc[df.label == 1, "prob"].to_numpy())
    if len(pos) == 0:
        return 0.5
    return float(pos[max(0, int(np.floor((1 - target) * len(pos))) - 1)]) if len(pos) > 1 else float(pos[0])


def contact_sheet(rows: pd.DataFrame, out: Path, title: str) -> None:
    if rows.empty:
        logger.warning("%s: nothing to draw", title)
        return
    tiles = []
    for _, r in rows.iterrows():
        img = cv2.imread(str(REPO_ROOT / r["path"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = TILE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        canvas = np.zeros((TILE, TILE, 3), dtype=np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        cv2.putText(canvas, f"y={int(r['label'])} p={r['prob']:.3f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(canvas, f"{r['center']} f{int(r['fold'])}", (4, TILE - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(canvas)

    while len(tiles) % N_COLS:
        tiles.append(np.zeros((TILE, TILE, 3), dtype=np.uint8))
    grid = np.concatenate(
        [np.concatenate(tiles[i : i + N_COLS], axis=1) for i in range(0, len(tiles), N_COLS)], axis=0
    )
    cv2.imwrite(str(out), grid)
    logger.info("%-12s -> %s (%d tiles)", title, out, len(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--n", type=int, default=25, help="tiles per sheet (>= 20 per the repo rule)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    oof = pd.read_csv(args.oof)
    out_dir = args.oof.parent

    thr = threshold_at_recall(oof)
    neg, pos = oof[oof.label == 0], oof[oof.label == 1]
    n_fp = int((neg["prob"] >= thr).sum())
    logger.info(
        "threshold @%.0f%% recall = %.4f -> %d false positives among %d negatives (PPV %.3f)",
        TARGET_RECALL * 100, thr, n_fp, len(neg), len(pos) * TARGET_RECALL / max(1, n_fp + len(pos) * TARGET_RECALL),
    )

    contact_sheet(neg.nlargest(args.n, "prob"), out_dir / "fp_top.png", "top FP")
    contact_sheet(pos.nsmallest(args.n, "prob"), out_dir / "fn_top.png", "top FN")
    border = pd.concat([
        neg.assign(d=(neg["prob"] - thr).abs()).nsmallest(args.n // 2, "d"),
        pos.assign(d=(pos["prob"] - thr).abs()).nsmallest(args.n // 2, "d"),
    ]).sort_values("prob", ascending=False)
    contact_sheet(border, out_dir / "borderline.png", "borderline")

    logger.info("per-center FP rate at that threshold:")
    for c, g in neg.groupby("center"):
        logger.info("  %-9s %4d / %4d  (%.1f%%)", c, int((g["prob"] >= thr).sum()), len(g),
                    100 * (g["prob"] >= thr).mean())


if __name__ == "__main__":
    main()
