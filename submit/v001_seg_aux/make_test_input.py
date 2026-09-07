"""Build a Grand-Challenge-shaped test input from local RARE25 images.

The shipped example stack has only 16 frames. This makes a larger one so the
streaming reader, batching and preprocessing can be exercised against images
whose local OOF score we already know -- see the parity check in SESSION_NOTES.

Frames are squashed to a square with cv2.INTER_LINEAR, which is how Grand
Challenge itself builds the stack (measured: octagon w/h 1.25 native -> 1.00 in
the real example stack, identical fill ratio).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

logger = logging.getLogger("make_test_input")
REPO_ROOT = Path(__file__).resolve().parents[2]

GC_FRAME_SIZE = 512
INPUT_RELATIVE_PATH = "images/stacked-barretts-esophagus-endoscopy"
INPUTS_JSON = [
    {
        "file": None,
        "image": {"name": "the_original_filename_of_the_file_that_was_uploaded.suffix"},
        "value": None,
        "interface": {
            "slug": "stacked-barretts-esophagus-endoscopy-images",
            "kind": "Image",
            "super_kind": "Image",
            "relative_path": INPUT_RELATIVE_PATH,
            "example_value": None,
        },
    }
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-csv", type=Path, default=REPO_ROOT / "workspace/fold/v1/folds.csv")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-pos", type=int, default=None,
                    help="force this many positives into the stack (default: keep natural ratio)")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "test/input/interface_0")
    ap.add_argument("--manifest", type=Path, default=None, help="csv listing the frames in stack order")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    df = pd.read_csv(args.folds_csv)
    df = df[df.fold == args.fold]
    if args.n_pos:
        # A stack of pure negatives cannot distinguish a working model from one that
        # always returns 0, so the regression stack must carry positives too.
        pos = df[df.label == 1].head(args.n_pos)
        neg = df[df.label == 0].head(max(0, args.n - len(pos)))
        df = pd.concat([pos, neg]).sample(frac=1.0, random_state=0)
    else:
        df = df.head(args.n)
    df = df.reset_index(drop=True)

    frames = np.empty((len(df), GC_FRAME_SIZE, GC_FRAME_SIZE, 3), dtype=np.uint8)
    for i, path in enumerate(df["path"]):
        img = cv2.cvtColor(cv2.imread(str(REPO_ROOT / path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        frames[i] = cv2.resize(img, (GC_FRAME_SIZE, GC_FRAME_SIZE), interpolation=cv2.INTER_LINEAR)

    img_dir = args.out / INPUT_RELATIVE_PATH
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*"):
        old.unlink()

    stack_path = img_dir / f"local_fold{args.fold}_{len(df)}.mha"
    sitk.WriteImage(sitk.GetImageFromArray(frames, isVector=True), str(stack_path), useCompression=False)
    (args.out / "inputs.json").write_text(json.dumps(INPUTS_JSON, indent=4))

    manifest = args.manifest or args.out.parent.parent / "test_manifest.csv"
    df.assign(stack_index=range(len(df))).to_csv(manifest, index=False)

    logger.info("wrote %s (%d frames, %.1f MB)", stack_path, len(df), stack_path.stat().st_size / 1e6)
    logger.info("wrote %s and %s", args.out / "inputs.json", manifest)


if __name__ == "__main__":
    main()
