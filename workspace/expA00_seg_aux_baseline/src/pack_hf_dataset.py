"""Pack everything a remote trainer needs into ONE HuggingFace dataset repo.

Why private, not public: the RARE25 training data is distributed under CC-BY-NC-SA
behind a click-through licence on Theta Vision Cortex. Re-publishing it openly
would bypass that gate; a *private* repo transfers just as fast and keeps the
data behind our own credentials. EDD2020 and EVC carry their own terms and are
included on the same basis.

Third-party model weights (SurgeNetXL, DINOv3) are deliberately NOT bundled --
they are re-downloaded from their origin on the remote machine so we are not
redistributing them.

Contents (~3.5 GB):
  rare25.tar          3095 challenge PNGs, original layout
  external.tar        EDD2020 + EVC images and masks (EVC = held-out eval only)
  meta/               folds.csv, external_index.csv
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import tarfile
from pathlib import Path

logger = logging.getLogger("pack_hf")
REPO_ROOT = Path(__file__).resolve().parents[3]


def add_tar(tar: tarfile.TarFile, src: Path, arc: str) -> int:
    n = 0
    for p in sorted(src.rglob("*")):
        if p.is_file():
            tar.add(p, arcname=f"{arc}/{p.relative_to(src)}")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. negichi/rare26-work")
    ap.add_argument("--out", type=Path, default=Path("/tmp/rare26_hf"))
    ap.add_argument("--push", action="store_true", help="upload (otherwise just build locally)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    out = args.out
    (out / "meta").mkdir(parents=True, exist_ok=True)

    # 1) challenge data
    rare_tar = out / "rare25.tar"
    if not rare_tar.exists():
        with tarfile.open(rare_tar, "w") as t:
            n = add_tar(t, REPO_ROOT / "data/RARE25/center_1", "center_1")
            n += add_tar(t, REPO_ROOT / "data/RARE25/center_2", "center_2")
        logger.info("rare25.tar: %d files, %.2f GB", n, rare_tar.stat().st_size / 1e9)

    # 2) external data actually referenced by the index (not the whole 68 GB tree)
    ext_tar = out / "external.tar"
    if not ext_tar.exists():
        import pandas as pd
        idx = pd.read_csv(REPO_ROOT / "workspace/expA00_seg_aux_baseline/src/external_index.csv").fillna("")
        wanted: set[str] = set()
        for _, r in idx.iterrows():
            wanted.add(r["path"])
            for cell in (r["be_masks"], r["neo_masks"]):
                wanted.update(x for x in str(cell).split(";") if x)
        with tarfile.open(ext_tar, "w") as t:
            for rel in sorted(wanted):
                src = REPO_ROOT / rel
                if src.exists():
                    t.add(src, arcname=rel)
        logger.info("external.tar: %d files, %.2f GB", len(wanted), ext_tar.stat().st_size / 1e9)

    # 3) metadata
    for rel in ["workspace/fold/v1/folds.csv",
                "workspace/expA00_seg_aux_baseline/src/external_index.csv"]:
        dst = out / "meta" / Path(rel).name
        dst.write_bytes((REPO_ROOT / rel).read_bytes())
    logger.info("meta/: %s", [p.name for p in (out / "meta").iterdir()])

    (out / "README.md").write_text(
        "# RARE26 working set (PRIVATE)\n\n"
        "Not for redistribution. RARE25 data is CC-BY-NC-SA from the challenge organisers;\n"
        "EDD2020 and EVC_Barretts_FullSet carry their own terms. Model weights are NOT\n"
        "included -- fetch them from their origin.\n\n"
        "```\nrare25.tar     -> data/RARE25/{center_1,center_2}/\n"
        "external.tar   -> data/BarrettsEsophagus/... (paths match external_index.csv)\n"
        "meta/          -> folds.csv, external_index.csv\n```\n")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    logger.info("total payload: %.2f GB at %s", total / 1e9, out)

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
        logger.info("uploading to %s (private)", args.repo_id)
        api.upload_folder(folder_path=str(out), repo_id=args.repo_id, repo_type="dataset")
        logger.info("done: https://huggingface.co/datasets/%s", args.repo_id)
    else:
        logger.info("built locally; re-run with --push to upload")


if __name__ == "__main__":
    main()
