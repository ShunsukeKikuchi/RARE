"""Heterogeneous ensemble predictor for the RARE26 algorithm container.

The ensemble mixes architectures and pretraining corpora on purpose -- EfficientNet /
ConvNeXt / ResNeXt / MaxViT UNets alongside DINOv2 / DINOv3 / SurgeNetXL LoRA ViTs --
because the test set spans 12 unseen centres and decorrelated errors are what an
ensemble actually buys. A ``manifest.json`` shipped with the weights tells the
container how to rebuild each member, so adding a member never requires a new image.

Three things here are deliberate:

* **Streaming input.** Stacks are read in slabs via SimpleITK's extract API rather
  than loaded whole; a 25k-frame stack would otherwise be ~20 GB in RAM and the
  runtime only has 16 GB.
* **Per-resolution grouping.** Members run at different input sizes (336 for the
  ViTs, 384 for the CNNs), so each slab is preprocessed once per distinct size and
  shared by every member using it, instead of once per member.
* **Batched inference.** One image at a time would turn a ~30 s job into minutes.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from .net import DinoV3LoRANet, MultiTaskNet, SurgeNetLoRANet, TimmViTLoRANet

logger = logging.getLogger(__name__)

CPU_BATCH_SIZE = 8
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GC_FRAME_SIZE = 512  # Grand Challenge already stacks at 512; see the training transforms
DINOV3_REPO_PATH = "/opt/app/dinov3_repo"


def build_member(spec: dict, model_dir: Path) -> torch.nn.Module:
    fam = spec.get("family", "unet_multitask")
    size = int(spec["image_size"])
    drop = float(spec.get("dropout", 0.0))
    if fam == "unet_multitask":
        return MultiTaskNet(encoder_name=spec["encoder_name"], encoder_weights=None,
                            seg_channels=spec.get("seg_channels", 2), dropout=drop)
    if fam == "surgenet_lora":
        # LoRA members ship adapters only; the frozen base is shared and travels once.
        return SurgeNetLoRANet(arch=spec["arch"], weights_path=str(model_dir / spec["base_weights"]),
                               img_size=size, rank=spec.get("lora_rank", 32),
                               alpha=spec.get("lora_alpha", 64), dinov3_repo=DINOV3_REPO_PATH,
                               dropout=drop)
    if fam == "vit_lora":
        return DinoV3LoRANet(model_name=spec["encoder_name"], img_size=size,
                             rank=spec.get("lora_rank", 32), alpha=spec.get("lora_alpha", 64),
                             dropout=drop, pretrained=False,
                             base_weights_path=str(model_dir / spec["base_weights"]))
    if fam == "timm_vit_lora":
        return TimmViTLoRANet(model_name=spec["encoder_name"], weights_path=None, img_size=size,
                              rank=spec.get("lora_rank", 32), alpha=spec.get("lora_alpha", 64),
                              dropout=drop, pretrained=False)
    raise ValueError(f"unknown family in manifest: {fam}")


# Grand Challenge kills a job at 600 s (one job = one .tiff of 384 frames), and a
# killed job produces NO output -- far worse than a slightly smaller ensemble.
# Members are scored in manifest order (most valuable first); once the elapsed
# time plus the slowest member seen so far would cross the budget, the rest are
# skipped and the pool is taken over the members that finished. The first member
# always runs, so there is always an output.
TIME_LIMIT_S = 600.0
TIME_BUDGET_S = 480.0

class Ensemble:
    def __init__(self, models_root: Path, batch_size: int, use_tta: bool):
        self.use_tta = use_tta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type == "cuda"
        # The CPU path exists only for local verification on boxes without GPU
        # passthrough, where a large batch exhausts RAM.
        self.batch_size = batch_size if self.half else min(batch_size, CPU_BATCH_SIZE)

        self.model_dir = self._find_model_dir(models_root)
        manifest = json.loads((self.model_dir / "manifest.json").read_text())
        self.members: list[dict] = manifest["members"]
        logger.info("manifest: %d member(s) from %s", len(self.members), self.model_dir)

        self.by_size: dict[int, list[dict]] = defaultdict(list)
        for m in self.members:
            self.by_size[int(m["image_size"])].append(m)
        logger.info("input sizes: %s", {k: len(v) for k, v in self.by_size.items()})

    @staticmethod
    def _find_model_dir(models_root: Path) -> Path:
        """Prefer the separately-uploaded model tarball.

        Grand Challenge extracts it to /opt/ml/model. That is checked first on
        purpose: with an image-first order, a stale copy baked into the image would
        keep being served after a tarball update and nothing in the output would say so.
        """
        for root in (Path("/opt/ml/model"), Path("/opt/ml/models"), models_root, Path("/opt/app/resources")):
            if root.is_dir() and (root / "manifest.json").is_file():
                return root
        raise RuntimeError("no manifest.json found under /opt/ml/model or resources/")

    def _load(self, spec: dict, state: "dict | None" = None) -> torch.nn.Module:
        model = build_member(spec, self.model_dir)
        if state is None:
            state = torch.load(self.model_dir / spec["weights"], map_location="cpu", weights_only=True)
        # LoRA members carry only adapters+head; the rest comes from the frozen base.
        missing, unexpected = model.load_state_dict(state, strict=not spec.get("partial", False))
        if unexpected:
            raise RuntimeError(f"{spec['weights']}: unexpected keys {list(unexpected)[:5]}")
        model.to(self.device).eval()
        if not self.half:
            return model
        # fp16 is not safe for every backbone: DINOv3 ViT-L carries base weights up to
        # |w| = 51 and its activations leave fp16's range entirely -- the forward returns
        # NaN while nothing raises, and one such member turns the whole pooled output into
        # NaN. Probe each model once and step up to bfloat16 (same exponent range as fp32,
        # supported on the platform's A10G) and then fp32 only where it is actually needed.
        size = int(spec["image_size"])
        probe = torch.zeros(2, 3, size, size, device=self.device)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            cand = model.to(dtype)
            with torch.no_grad():
                if torch.isfinite(cand.forward_cls(probe.to(dtype))).all():
                    if dtype is not torch.float16:
                        logger.warning("%s: fp16 forward is non-finite, running in %s",
                                       spec["config"], str(dtype).split(".")[-1])
                    return cand
        raise RuntimeError(f"{spec['config']}: non-finite forward in every dtype")

    def _preprocess(self, frames: np.ndarray, size: int) -> torch.Tensor:
        import cv2

        out = np.empty((len(frames), size, size, 3), dtype=np.float32)
        for i, f in enumerate(frames):
            if f.ndim == 2:
                f = np.stack([f] * 3, axis=-1)
            elif f.shape[-1] == 4:
                f = f[..., :3]
            r = cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
            out[i] = (r.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def _member_logits(self, model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
        out = []
        for b in range(0, len(tensor), self.batch_size):
            x = tensor[b : b + self.batch_size].to(self.device, non_blocking=True)
            # The cache is stored fp16 to halve host memory. Take the dtype from the
            # model itself rather than assuming fp16: _load steps a member up to
            # bfloat16 or fp32 when its forward overflows, and the input has to follow.
            x = x.to(next(model.parameters()).dtype)
            views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2])] if self.use_tta else [x]
            lg = torch.stack([model.forward_cls(v).float() for v in views]).mean(0)
            out.append(lg.cpu().numpy())
        return np.concatenate(out)

    def predict_file(self, path: Path, chunk: int = 256, block: int = 2048) -> list[float]:
        """Score a stacked file.

        Frames are decoded in slabs and cached as fp16 CPU tensors, then each member
        is loaded ONCE and run over the whole block. The obvious alternative --
        looping members inside the chunk loop -- reloads every model for every chunk,
        which for a 40-member ensemble dominates the runtime.

        ``block`` bounds the cache: a job here is ~384 frames (~0.3 GB per input
        size), but a very large stack is split so the cache cannot exhaust the
        16 GB runtime.
        """
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        size = list(reader.GetSize())
        n = size[2] if reader.GetDimension() >= 3 else 1
        logger.info("input %s: size=%s -> %d image(s)", path.name, size, n)

        acc = np.zeros((len(self.members), n), dtype=np.float64)
        t0 = time.monotonic()
        active = list(range(len(self.members)))     # members still in the ensemble
        slowest = 0.0
        for b0 in range(0, n, block):
            bn = min(block, n - b0)
            cache: dict[int, list[torch.Tensor]] = {s: [] for s in self.by_size}
            for start in range(b0, b0 + bn, chunk):
                take = min(chunk, b0 + bn - start)
                frames = self._read_slab(reader, size, start, take)
                for s in self.by_size:
                    cache[s].append(self._preprocess(frames, s).half())
            tensors = {s: torch.cat(v) for s, v in cache.items()}
            del cache

            # The try-out on the platform spent 81 s loading 30 members at 0% GPU and
            # 0.26% CPU -- the card idles while each state_dict comes off disk. Read the
            # next member's file on a worker thread so that wait overlaps with the
            # current member's compute. Only the read is prefetched; construction and
            # weight loading stay on the main thread, so nothing about the model changes.
            import concurrent.futures as _cf
            pool = _cf.ThreadPoolExecutor(max_workers=1)
            def _read(j):
                return torch.load(self.model_dir / self.members[j]["weights"],
                                  map_location="cpu", weights_only=True)
            ahead = pool.submit(_read, active[0]) if active else None
            kept = []
            for pos, i in enumerate(active):
                spec = self.members[i]
                elapsed = time.monotonic() - t0
                if kept and elapsed + slowest > TIME_BUDGET_S:
                    logger.warning("time budget: %.0fs elapsed, slowest member %.0fs -> skipping %d member(s) from %s",
                                   elapsed, slowest, len(active) - len(kept), spec["config"])
                    break
                tm = time.monotonic()
                state = ahead.result() if ahead is not None else None
                ahead = pool.submit(_read, active[pos + 1]) if pos + 1 < len(active) else None
                model = self._load(spec, state)
                acc[i, b0 : b0 + bn] = self._member_logits(model, tensors[int(spec["image_size"])])
                del model
                if self.half:
                    torch.cuda.empty_cache()
                dt = time.monotonic() - tm
                slowest = max(slowest, dt)
                kept.append(i)
                logger.info("member %2d %-34s %5.1fs (elapsed %5.1fs)", i, spec["config"], dt, time.monotonic() - t0)
            pool.shutdown(wait=False)
            active = kept                            # later blocks use the same subset
            del tensors
            logger.info("scored %d / %d", b0 + bn, n)
        logger.info("pooled %d / %d member(s) in %.0fs", len(active), len(self.members), time.monotonic() - t0)
        return self._pool(acc, active)

    def _pool(self, logits: np.ndarray, active: list[int] | None = None) -> list[float]:
        """Calibrate per member, then average. Calibration parameters were fitted on
        out-of-fold predictions; without them the members sit on different scales.
        ``active`` restricts the pool to the members that actually ran."""
        idx = list(range(len(self.members))) if active is None else list(active)
        cal = np.empty((len(idx), logits.shape[1]), dtype=np.float64)
        w = np.empty(len(idx), dtype=np.float64)
        for r, i in enumerate(idx):
            spec = self.members[i]
            t = float(spec.get("cal_t", 1.0))
            b = float(spec.get("cal_b", 0.0))
            cal[r] = logits[i] * t + b
            # Absent weights mean a uniform pool, so an older manifest still works.
            w[r] = float(spec.get("weight", 1.0))
        # Renormalise over the members that actually ran: when the time-budget guard
        # drops the tail, the survivors must still form a proper weighted mean rather
        # than a pool that silently sums to less than one.
        w = w / w.sum() if w.sum() > 0 else np.full(len(idx), 1.0 / len(idx))
        mean = (cal * w[:, None]).sum(axis=0)
        probs = 1.0 / (1.0 + np.exp(-mean))
        # A single inf in one member's weights once turned every prediction into NaN
        # while all 30 members "ran" and the output looked structurally valid. Fail
        # loudly here instead: a crashed job is diagnosable, an all-NaN one scores zero
        # and looks like a modelling problem.
        if not np.isfinite(probs).all():
            culprits = [self.members[i]["config"] for r, i in enumerate(idx)
                        if not np.isfinite(cal[r]).all()]
            raise RuntimeError(f"non-finite predictions from member(s): {culprits[:5]}")
        return [float(p) for p in probs]

    @staticmethod
    def _read_slab(reader: sitk.ImageFileReader, size: list[int], start: int, take: int) -> np.ndarray:
        if reader.GetDimension() < 3:
            arr = sitk.GetArrayFromImage(reader.Execute())
            return arr[None] if arr.ndim in (2, 3) and start == 0 else arr
        reader.SetExtractIndex([0, 0, start])
        reader.SetExtractSize([size[0], size[1], take])
        arr = sitk.GetArrayFromImage(reader.Execute())
        if arr.ndim == 2:
            arr = arr[None, ..., None]
        elif arr.ndim == 3 and take == 1:
            arr = arr[None]
        return arr
