"""Pick the largest learning rate that trains *stably* for a given config.

Why this exists: five LoRA recipes (DINOv2 B/L, CLIP, SAM, SurgeNet-DINOv2) were
discarded as "the weights are weak" when they were in fact diverging at lr 5e-4.
The tell is in the loss trajectory, not the level -- a diverging run oscillates
(0.67, 0.37, 2.57, 0.41, 1.25, ...) while a healthy one descends monotonically.
Frozen-feature linear probes showed DINOv2 was the *best* backbone we had, so the
cost of reading "unstable" as "weak" was high.

The test deliberately overfits a handful of images: any usable configuration can
drive 16 images to near-zero loss, so a run that cannot is misconfigured rather
than uninformative. Runs on CPU in a couple of minutes, so it is cheap enough to
gate every new backbone before it takes a GPU slot.

    python src/probe_lr.py --config config/dinov2_vitb.yaml
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from datasets import _read_rgb  # noqa: E402
from model import build_model  # noqa: E402
from transforms import build_transforms  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LRS = (5e-4, 2e-4, 1e-4, 5e-5)


def trajectory(cfg: dict, x: torch.Tensor, y: torch.Tensor, lr: float, steps: int) -> list[float]:
    torch.manual_seed(0)
    model = build_model(cfg).to(x.device)
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        logit = model.forward_cls(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


def score(losses: list[float]) -> tuple[int, float]:
    """(number of upward spikes, final loss). Fewer spikes first, then lower loss."""
    spikes = sum(1 for i in range(1, len(losses)) if losses[i] > losses[i - 1] * 1.5)
    return spikes, losses[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--lrs", type=float, nargs="*", default=list(DEFAULT_LRS))
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    size = int(cfg["data"]["image_size"])

    folds = pd.read_csv(REPO_ROOT / cfg["data"]["folds_csv"])
    half = args.n_images // 2
    df = pd.concat([folds[folds.label == 1].head(half), folds[folds.label == 0].head(half)])
    tf = build_transforms(size, train=False, gc_prescale=cfg["data"].get("gc_prescale"))
    x = torch.stack([tf(image=_read_rgb(REPO_ROOT / p))["image"] for p in df.path]).to(args.device)
    y = torch.tensor(df.label.to_numpy(), dtype=torch.float32, device=args.device)

    print(f"{args.config.stem}: overfitting {len(df)} images for {args.steps} steps")
    results = {}
    for lr in args.lrs:
        losses = trajectory(cfg, x, y, lr, args.steps)
        spikes, final = score(losses)
        results[lr] = (spikes, final)
        print(f"  lr={lr:.0e}  " + " ".join(f"{v:5.3f}" for v in losses) + f"   spikes={spikes} final={final:.3f}")

    # Largest lr that neither oscillates nor stalls -- speed matters, stability decides.
    ok = [lr for lr, (sp, fin) in results.items() if sp == 0 and fin < 0.3]
    best = max(ok) if ok else min(results, key=lambda lr: results[lr])
    print(f"RECOMMENDED_LR {best:.0e}" + ("" if ok else "  (no fully stable lr; picked fewest spikes)"))


if __name__ == "__main__":
    main()
