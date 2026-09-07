"""Build the index of external segmentation-supervised data.

Two sources carry Barrett's-relevant pixel supervision. Neither has a patient
overlap guarantee with RARE, so both are train-only (never in a val fold).

EVC_Barretts_FullSet
    100 images, filename ``patXX_imY_{ACHD,NDBT}.png``.
    ACHD = adenocarcinoma / high-grade dysplasia (neoplasia), NDBT = non-dysplastic.
    Five independent expert delineations per image (``*_exp{1..5}.bmp``). We keep
    them as a *soft* consensus (mean over experts) rather than majority-voting, so
    inter-observer disagreement is carried into the target.
    Supervises the neoplasia channel only -- there is no BE-extent annotation.

EDD2020
    386 images, per-class masks named ``<image>_<class>.tif`` over
    {BE, suspicious, HGD, cancer, polyp}. We keep only images carrying at least one
    of BE/suspicious/HGD/cancer AND no polyp mask -- a polyp annotation marks the
    frame as colonoscopy, and 5 such frames (e.g. dye-sprayed colon labelled
    "suspicious") otherwise slip through the upper-GI filter. Both channels are
    supervised: BE extent, and neoplasia = HGD u cancer u suspicious.

    Residual risk: EDD2020 carries no organ field, so a colon frame labelled only
    cancer/HGD cannot be excluded this way. Institution NAF in particular is mostly
    colonoscopy.

Output columns:
    path, dataset, label, be_masks, neo_masks, has_be, has_neo
``be_masks``/``neo_masks`` are ';'-separated relative paths (empty = all-zero mask,
which is still a valid negative target as long as the corresponding has_* is 1).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "data" / "BarrettsEsophagus"

EVC_DIR = "EVC_Barretts_FullSet"
EDD_DIR = "EDD2020"
N_EVC_EXPERTS = 5
EDD_NEO_CLASSES = ("HGD", "cancer", "suspicious")
EDD_COLON_CLASS = "polyp"


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def build_evc(root: Path) -> list[dict]:
    img_dir = root / EVC_DIR / "images"
    ann_dir = root / EVC_DIR / "annotations_bmp"
    rows = []
    for img in sorted(img_dir.glob("*.png")):
        pathology = img.stem.rsplit("_", 1)[-1]
        if pathology not in ("ACHD", "NDBT"):
            raise ValueError(f"unexpected pathology token in {img.name}")
        experts = sorted(ann_dir.glob(f"{img.stem}_exp*.bmp"))
        if len(experts) != N_EVC_EXPERTS:
            raise FileNotFoundError(f"{img.name}: found {len(experts)} expert masks")
        rows.append(
            {
                "path": _rel(img),
                "dataset": "evc",
                "label": int(pathology == "ACHD"),
                "be_masks": "",
                "neo_masks": ";".join(_rel(e) for e in experts),
                "has_be": 0,   # EVC has no BE-extent annotation
                "has_neo": 1,
            }
        )
    return rows


def build_edd(root: Path) -> list[dict]:
    img_dir = root / EDD_DIR / "originalImages"
    mask_dir = root / EDD_DIR / "masks"
    rows = []
    for img in sorted(img_dir.glob("*.jpg")):
        masks = {}
        for m in mask_dir.glob(f"{img.stem}_*.tif"):
            masks[m.name[len(img.stem) + 1 : -len(".tif")]] = m
        if EDD_COLON_CLASS in masks:
            continue  # a polyp annotation marks this frame as colonoscopy
        be = [masks[c] for c in ("BE",) if c in masks]
        neo = [masks[c] for c in EDD_NEO_CLASSES if c in masks]
        if not be and not neo:
            continue  # unannotated -> no supervision to offer
        rows.append(
            {
                "path": _rel(img),
                "dataset": "edd2020",
                "label": int(bool(neo)),
                "be_masks": ";".join(_rel(p) for p in be),
                "neo_masks": ";".join(_rel(p) for p in neo),
                "has_be": 1,
                "has_neo": 1,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=EXTERNAL_ROOT)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "external_index.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    rows = build_evc(args.root) + build_edd(args.root)
    df = pd.DataFrame(rows)

    for ds, g in df.groupby("dataset"):
        logger.info(
            "%-8s n=%-4d pos=%-4d be_sup=%-4d neo_sup=%-4d (nonempty be=%d, neo=%d)",
            ds,
            len(g),
            int(g["label"].sum()),
            int(g["has_be"].sum()),
            int(g["has_neo"].sum()),
            int((g["be_masks"] != "").sum()),
            int((g["neo_masks"] != "").sum()),
        )

    # Measured invariants -- these guard against a silently rearranged data dir.
    evc = df[df.dataset == "evc"]
    assert len(evc) == 100 and evc["label"].sum() == 50, "EVC: expected 50 ACHD / 50 NDBT"
    assert (evc["neo_masks"].str.count(";") == N_EVC_EXPERTS - 1).all(), "EVC: expected 5 experts each"

    df.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(df))


if __name__ == "__main__":
    main()
