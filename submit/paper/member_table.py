"""Render the shipped ensemble as a markdown table for the method PDF.

Reads the manifest that build_submission.py actually wrote, so the table can never
drift from what is in the tarball -- it is generated from the same file the
container loads, not from notes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FAMILY_LABEL = {
    "unet_multitask": "seg-aux CNN/hybrid",
    "vit_lora": "DINOv3 LoRA",
    "surgenet_lora": "SurgeNetXL LoRA",
    "timm_vit_lora": "foundation-ViT LoRA",
}


def main() -> None:
    man = Path(sys.argv[1] if len(sys.argv) > 1 else "submit/v001_seg_aux/resources/manifest.json")
    members = json.loads(man.read_text())["members"]

    print(f"Ensemble: {len(members)} members\n")
    print("| # | recipe | family | px | OOF AUROC | val n |")
    print("|---|--------|--------|----|-----------|-------|")
    for i, m in enumerate(members):
        fam = FAMILY_LABEL.get(m["family"], m["family"])
        # n < 3000 means the recipe was validated on fold0 only, which is the easy fold
        # -- flagged so the two columns of numbers are not read as comparable.
        star = "" if m.get("oof_n", 0) >= 3000 else "*"
        print(f"| {i} | `{m['config']}` | {fam} | {m['image_size']} | "
              f"{m.get('oof_auroc', 0):.4f}{star} | {m.get('oof_n', 0)} |")
    single = sum(1 for m in members if m.get("oof_n", 0) < 3000)
    if single:
        print(f"\n*{single} recipes are validated on fold0 only (619 images); fold0 is the "
              "easiest of the five, so those numbers are optimistic and are not comparable "
              "with the 3095-image entries.")
    by_fam: dict[str, int] = {}
    for m in members:
        by_fam[FAMILY_LABEL.get(m["family"], m["family"])] = by_fam.get(FAMILY_LABEL.get(m["family"], m["family"]), 0) + 1
    print("\nBy family: " + ", ".join(f"{k} {v}" for k, v in sorted(by_fam.items())))


if __name__ == "__main__":
    main()
