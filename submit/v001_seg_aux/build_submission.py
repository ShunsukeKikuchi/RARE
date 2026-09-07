"""Assemble the algorithm model tarball: weights + manifest + calibration.

The container is architecture-agnostic; everything it needs to rebuild a member
lives in manifest.json. This script is where the ensemble is actually decided.

Calibration: each config's out-of-fold predictions are a held-out prediction set
for "a model of this config", so one affine (t, b) is fitted there per config and
attached to every member of it -- including the all-data model, which has no
held-out data of its own. Fitting on the pooled OOF (3095 predictions) rather than
averaging five per-fold fits gives the same parameters far more stably.

The transfer 80% -> 100% data shifts a member's scores slightly, but that shift is
far smaller than the between-config scale differences calibration exists to remove
(a LoRA ViT and a UNet-CNN sit on completely different logit scales), so the
approximation is not the binding source of error.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger("build_submission")
REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_PRIORS = [100 / 101, 1 / 101]      # the challenge's simulated 1% prevalence
EPS = 1e-6

# dropout is included on purpose: it decides whether the head is a bare Linear or
# Sequential(Dropout, Linear), which changes the state_dict key names.
FAMILY_KEYS = {
    "unet_multitask": ("encoder_name", "seg_channels", "dropout"),
    "vit_lora": ("encoder_name", "lora_rank", "lora_alpha", "dropout"),
    "timm_vit_lora": ("encoder_name", "lora_rank", "lora_alpha", "dropout"),
    "surgenet_lora": ("arch", "lora_rank", "lora_alpha", "dropout"),
}


def fit_calibration(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Affine calibration toward the 1% operating prior (psrcal AffineCalLogLoss)."""
    from psrcal.calibration import AffineCalLogLoss, calibrate

    s = np.log(np.clip(probs, EPS, 1 - EPS) / (1 - np.clip(probs, EPS, 1 - EPS)))
    two = np.stack([-s / 2, s / 2], axis=1)
    _, (t, b) = calibrate(
        trnscores=torch.tensor(two, dtype=torch.float64),
        trnlabels=torch.tensor(labels, dtype=torch.long),
        tstscores=torch.tensor(two, dtype=torch.float64),
        calclass=AffineCalLogLoss, bias=True, priors=TARGET_PRIORS, quiet=True,
    )
    b = b.detach().numpy()
    # Collapse the 2-column affine + log-prior into the single positive-logit form
    # the container applies: logit' = t*logit + delta_b + delta_logprior.
    delta_b = float(b[1] - b[0]) + float(np.log(TARGET_PRIORS[1]) - np.log(TARGET_PRIORS[0]))
    return float(t.item()), delta_b


EMA_PREFIX, RAW_PREFIX = "ema.module.", "model."


def _write_weights(src: Path, dst: Path) -> None:
    """Copy a member's weights, extracting them if the source is a Lightning ckpt.

    Remote runs already export a plain state_dict; local runs leave the raw
    checkpoint (optimizer state, hparams and all). Take the EMA weights when
    present -- those are what validation selected the checkpoint on, so shipping
    the raw ones would deploy a different model than the CV score describes.
    """
    if src.suffix == ".pth":
        state = torch.load(src, map_location="cpu", weights_only=True)
    else:
        ck = torch.load(src, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        prefix = EMA_PREFIX if any(k.startswith(EMA_PREFIX) for k in sd) else RAW_PREFIX
        state = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not state:
            raise RuntimeError(f"{src}: no '{prefix}*' tensors")
        logger.info("  extracted %s weights from %s", "EMA" if prefix == EMA_PREFIX else "raw", src.name)
    # Weights ship in the dtype they were trained in. Casting them to fp16 saved 38% of
    # the tarball and cost two silent all-NaN incidents -- a BatchNorm running_var above
    # fp16's 65504 ceiling, and DINOv3 ViT-L whose activations leave fp16's range
    # entirely. The size was never actually a constraint, so the saving bought nothing.
    # Inference precision is decided per member at load time instead, where a probe can
    # verify it rather than a build-time assumption standing in for one.
    nonfinite = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if nonfinite:
        raise RuntimeError(f"{dst.name}: non-finite tensors {nonfinite[:5]}")
    torch.save(state, dst)


def _load_config(name: str, results_dir: Path) -> dict:
    """Find the config for a results dir.

    Remote runs push weights and metrics but not config.yaml, and the experiment
    name (``expA02_surgenet_vitb``) need not match the config filename
    (``surgenet_vitb.yaml``) -- so fall back to matching on experiment.name.
    """
    local = next(iter(sorted(results_dir.glob("fold*/config.yaml"))), None)
    if local is not None:
        return yaml.safe_load(local.read_text())
    cfg_dir = REPO_ROOT / "workspace/expA00_seg_aux_baseline/config"
    direct = cfg_dir / f"{name}.yaml"
    if direct.exists():
        return yaml.safe_load(direct.read_text())
    for f in sorted(cfg_dir.glob("*.yaml")):
        c = yaml.safe_load(f.read_text())
        if c.get("experiment", {}).get("name") == name:
            return c
    raise FileNotFoundError(f"no config for results dir '{name}'")


def _fold_val_curves(name: str, d: Path) -> "pd.DataFrame | None":
    """Per-epoch validation AUROC for every fold of this config, epochs x folds.

    Looks inside the results dir first (the compact export copies metrics_fold*.csv
    there), then falls back to the standalone metrics collections, which are keyed
    by the vast.ai label rather than by experiment.name.
    """
    cands = sorted(d.glob("metrics_fold*.csv")) or sorted(d.glob("fold*/metrics.csv"))
    if not cands:
        for root in (REPO_ROOT / "results/remote_metrics2", REPO_ROOT / "results/remote_metrics"):
            for sub in sorted(root.glob("*/")) if root.is_dir() else []:
                lab = sub.name
                if lab in name or name in lab or name.replace("expA02_", "") == lab:
                    cands = sorted(sub.glob("fold*.csv"))
                    break
            if cands:
                break
    cols = []
    for f in cands:
        try:
            df = pd.read_csv(f)
            c = [x for x in df.columns if "auroc" in x.lower()]
            if not c:
                continue
            cols.append(df[["epoch", c[0]]].dropna().groupby("epoch").last()[c[0]])
        except Exception:
            continue
    return pd.concat(cols, axis=1) if cols else None


def _pick_all_snapshot(d: Path, name: str) -> "Path | None":
    """Choose the all-data checkpoint closest to this config's own best epoch.

    An all-data run has no validation split, so its endpoint cannot be chosen from
    its own curve -- it has to be transplanted from the 5-fold runs. That transplant
    is worth doing: shipping the final epoch instead costs 0.008-0.016 val AUROC on
    most configs here, more than the gap between pooling rules.
    """
    # Two layouts reach this point: vast-exported members arrive as compact
    # all.pth / all_epochNN.pth, while locally trained ones keep Lightning's raw
    # all/epochNN.ckpt. Looking only for the former silently dropped every local
    # member from the bundle -- the builder reported success with a third of the
    # ensemble missing, so both layouts are matched here.
    snaps: dict[int | None, Path] = {}
    for w in d.glob("all*.pth"):
        m = re.match(r"all_epoch(\d+)$", w.stem)
        snaps[int(m.group(1)) if m else None] = w      # None = the final-epoch model
    for w in (d / "all").glob("epoch*.ckpt"):
        m = re.match(r"epoch(\d+)$", w.stem)
        if m:
            snaps.setdefault(int(m.group(1)), w)
    if not snaps:
        logger.warning("%-34s no all-data checkpoint (looked for all*.pth and all/epoch*.ckpt)", name)
        return None

    curves = _fold_val_curves(name, d)
    if curves is None:
        pick = snaps.get(None) or snaps[max(k for k in snaps if k is not None)]
        logger.warning("%-34s no fold curves -> endpoint not transplanted (%s)", name, pick.name)
        return pick

    mean = curves.mean(axis=1)
    best_ep = int(mean.idxmax()) + 1
    # The final-epoch model corresponds to the last epoch actually recorded.
    known = {(len(mean) if k is None else k): w for k, w in snaps.items()}
    chosen_ep = min(known, key=lambda e: (abs(e - best_ep), e))
    cost = mean.iloc[min(chosen_ep, len(mean)) - 1] - mean.max()
    logger.info("%-34s best epoch %d -> ship %s (val AUROC %+.4f vs best)",
                name, best_ep, known[chosen_ep].name, cost)
    return known[chosen_ep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                    help="results dir names to include (local results/ or results/remote/)")
    # Default to the all-data models: they see 100% of the data where a fold model
    # sees 80%, and the fold runs are still needed -- but as instrumentation, to
    # supply the calibration parameters and the endpoint, not as shipped members.
    ap.add_argument("--members", choices=["fold", "all", "both"], default="all")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "resources")
    ap.add_argument("--no-calibration", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    members: list[dict] = []
    bases: dict[str, Path] = {}
    for name in args.configs:
        # Both roots can hold a directory of the same name -- results/ sometimes has a
        # leftover stub (weights only, no oof.csv) beside the real vast-collected copy
        # in results/remote/. Taking the first that merely *exists* silently shipped
        # those members uncalibrated, so prefer whichever actually has the OOF.
        cands = [q for q in (REPO_ROOT / "results" / name, REPO_ROOT / "results/remote" / name) if q.is_dir()]
        d = next((q for q in cands if (q / "oof.csv").is_file()), cands[0] if cands else None)
        if d is None:
            logger.warning("skip %s (not found)", name)
            continue
        cfg = _load_config(name, d)
        m, data = cfg["model"], cfg["data"]
        fam = m.get("family", "unet_multitask")

        t, b = 1.0, 0.0
        oof = d / "oof.csv"
        if oof.exists() and not args.no_calibration:
            o = pd.read_csv(oof)
            t, b = fit_calibration(o["prob"].to_numpy(), o["label"].to_numpy())
            from sklearn.metrics import roc_auc_score
            member_auroc = float(roc_auc_score(o["label"], o["prob"]))
            # Recipes are validated on fold0 only, full 5-fold configs on all 3095
            # images. fold0 is the easy fold (0.9953 vs 0.9556 across folds for one
            # config), so the two numbers are not on the same scale and ranking them
            # together would push single-fold recipes above stronger members -- and the
            # container drops members from the END of that ranking when time runs out.
            # Rank within coverage tier instead: fully-validated members first.
            member_cov = len(o)
            logger.info("%-34s calibration t=%.3f b=%+.3f (from %d OOF preds)", name, t, b, len(o))
        else:
            logger.warning("%-34s NO OOF -> uncalibrated (t=1,b=0)", name)
            member_auroc, member_cov = 0.0, 0

        spec_base = {"config": name, "family": fam, "image_size": int(data["image_size"]),
                     "cal_t": t, "cal_b": b, "oof_auroc": round(member_auroc, 4),
                     "oof_n": member_cov}
        for k in FAMILY_KEYS[fam]:
            if k in m:
                spec_base[k] = m[k]
        if fam == "surgenet_lora":
            src = REPO_ROOT / m["weights_path"]
            if src.exists() and src.name not in bases:
                bases[src.name] = src
            spec_base["base_weights"] = src.name
            spec_base["partial"] = True          # LoRA members ship adapters+head only
        if fam == "vit_lora":
            # The export strips LoRA members down to adapters+head, so these files do
            # NOT contain the DINOv3 backbone. Materialise it once from timm here --
            # the container has no network and cannot fetch it itself.
            base_name = f"base__{m['encoder_name'].replace('/', '_')}.pth"
            if base_name not in bases:
                import timm
                net = timm.create_model(m["encoder_name"], pretrained=True, num_classes=1,
                                        img_size=int(data["image_size"]))
                tmp = out / base_name
                torch.save(net.state_dict(), tmp)
                bases[base_name] = tmp
                logger.info("materialised shared DINOv3 base %s (%.0f MB)",
                            base_name, tmp.stat().st_size / 1e6)
            spec_base["base_weights"] = base_name
            spec_base["partial"] = True
        if fam == "timm_vit_lora":
            spec_base["partial"] = False

        want = []
        if args.members in ("fold", "both"):
            want += sorted(d.glob("fold?.pth")) or sorted(d.glob("fold*/best_model.ckpt"))
        if args.members in ("all", "both"):
            picked = _pick_all_snapshot(d, name)
            if picked is not None:
                want.append(picked)
        for w in want:
            tag = w.stem if w.suffix == ".pth" else w.parent.name
            dst = f"{name}__{tag}.pth"
            _write_weights(w, out / dst)
            members.append({**spec_base, "weights": dst})

    for n, p in bases.items():
        if p.resolve() != (out / n).resolve():
            shutil.copy(p, out / n)
        logger.info("shared frozen base: %s (%.0f MB)", n, p.stat().st_size / 1e6)

    # Load every shipped file before declaring the bundle built. A weights file
    # copied while rsync was still writing its source is the right size and the
    # wrong bytes; without this check the first symptom is the container dying in
    # Grand Challenge with "failed finding central directory".
    broken = []
    for w in sorted(out.glob("*.pth")):
        try:
            sd = torch.load(w, map_location="cpu", weights_only=True)
            if not isinstance(sd, dict) or not sd:
                broken.append(f"{w.name}: empty or not a state_dict")
            else:
                nf = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
                if nf:
                    broken.append(f"{w.name}: non-finite tensors {nf[:3]}")
        except Exception as exc:
            broken.append(f"{w.name}: {type(exc).__name__}")
    if broken:
        raise RuntimeError("unloadable weights in bundle:\n  " + "\n  ".join(broken))
    logger.info("verified %d weight file(s) load cleanly", len(list(out.glob("*.pth"))))

    # Strongest member first. The container scores members in this order and, if the
    # 600 s job limit looms, drops members from the end -- so the order decides which
    # members survive a slow job. OOF AUROC is the ranking we trust for that.
    # Full-coverage members outrank single-fold ones regardless of score; within a
    # tier, higher OOF AUROC first.
    members.sort(key=lambda m: (-(m.get("oof_n", 0) >= 3000), -m.get("oof_auroc", 0.0)))
    for r, m in enumerate(members):
        logger.info("  rank %2d  %-34s oof_auroc=%.4f (n=%d)", r, m["config"],
                    m.get("oof_auroc", 0.0), m.get("oof_n", 0))
    # Rank-linear weights: the best member by OOF AUROC gets K, the worst 1, normalised.
    # Measured against uniform by fitting the weights on one random half of the cohort and
    # scoring the other (60 splits): FPR@90%TPR 0.0144 vs 0.0154, better in 40 of 60
    # (binomial P=0.007). The effect is small and its CI spans zero, but the direction is
    # consistent, and unlike discrete subset selection -- which lost 19 splits out of 20 --
    # a continuous weighting degrades gracefully when the rank estimate is wrong. The top
    # weight is only ~2x uniform, so no single member can carry the pool.
    k = len(members)
    for rank, m in enumerate(members):          # members are already sorted best-first
        m["weight"] = round((k - rank) / (k * (k + 1) / 2), 6)
    logger.info("rank-linear weights: top %.3f (%s), bottom %.3f (%s)",
                members[0]["weight"], members[0]["config"],
                members[-1]["weight"], members[-1]["config"])
    (out / "manifest.json").write_text(json.dumps({"members": members}, indent=2))
    total = sum(p.stat().st_size for p in out.iterdir())
    logger.info("%d member(s), %d config(s), %.0f MB -> %s", len(members),
                len({m['config'] for m in members}), total / 1e6, out)
    for fam in sorted({m["family"] for m in members}):
        logger.info("  %-16s %d member(s)", fam, sum(1 for m in members if m["family"] == fam))


if __name__ == "__main__":
    main()
