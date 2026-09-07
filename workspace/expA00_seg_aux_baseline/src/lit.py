"""LightningModule for the seg-aux multi-task classifier."""

from __future__ import annotations

import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from timm.utils import ModelEmaV3
from torch import nn

from metrics import compute_metrics
from model import build_model

logger = logging.getLogger(__name__)

DICE_EPS = 1.0


def masked_seg_loss(
    seg_logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Soft-dice + BCE, averaged over supervised (sample, channel) pairs only.

    ``weight`` is [B, C] and is 0 wherever a channel has no annotation for that
    sample -- which is how RARE25's mask-free images ride along in the same batch.
    Targets may be soft (EVC's five-expert consensus); both terms handle that.
    """
    bce = F.binary_cross_entropy_with_logits(seg_logits, target, reduction="none").mean(dim=(2, 3))

    prob = torch.sigmoid(seg_logits)
    inter = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * inter + DICE_EPS) / (denom + DICE_EPS)

    per_channel = bce + dice
    total = weight.sum()
    if total <= 0:
        return seg_logits.sum() * 0.0  # keeps the graph alive with zero gradient
    return (per_channel * weight).sum() / total


class LitMultiTask(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = build_model(cfg)

        # Raw val/auroc swings ~0.010 over the last 10 epochs and the best epoch lands
        # anywhere from 4 to 25, so `save_top_k=1` was largely selecting noise: the
        # best-minus-final gap reached 0.021, an order of magnitude above the effects
        # being compared. EMA smooths the weight trajectory so the epoch curve --
        # and hence the best epoch -- is meaningful.
        decay = float(cfg["train"].get("ema_decay", 0.0)) if "train" in cfg else 0.0
        self.ema = None
        if decay > 0:
            self.ema = ModelEmaV3(self.model, decay=decay, use_warmup=True)
            # ModelEmaV3 does not freeze its copy; without this the EMA weights would
            # be handed to the optimizer and clipped as if they were trainable.
            self.ema.module.requires_grad_(False)

        self.seg_weight = float(cfg["loss"]["seg_weight"])
        pos_weight = cfg["loss"].get("pos_weight")
        self.cls_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(pos_weight)) if pos_weight else None
        )
        self._val_logits: list[np.ndarray] = []
        self._val_labels: list[np.ndarray] = []

    def forward(self, x):
        return self.model(x)

    @property
    def eval_model(self) -> nn.Module:
        """Weights used for validation and inference: the EMA copy when enabled."""
        return self.ema.module if self.ema is not None else self.model

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update(self.model, step=self.global_step)

    def _shared_step(self, batch, stage: str):
        model = self.model if stage == "train" else self.eval_model
        seg_logits, cls_logits = model(batch["image"])
        loss_cls = self.cls_loss(cls_logits, batch["label"])

        # Classification-only members (e.g. the DINOv3 LoRA ViT) return no seg map.
        if self.seg_weight > 0 and seg_logits is not None:
            loss_seg = masked_seg_loss(seg_logits, batch["mask"], batch["mask_weight"])
            loss = loss_cls + self.seg_weight * loss_seg
            self.log(f"{stage}/loss_seg", loss_seg, on_step=False, on_epoch=True, batch_size=len(cls_logits))
        else:
            loss = loss_cls

        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(cls_logits))
        self.log(f"{stage}/loss_cls", loss_cls, on_step=False, on_epoch=True, batch_size=len(cls_logits))
        return loss, cls_logits

    def training_step(self, batch, _):
        loss, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, _):
        _, cls_logits = self._shared_step(batch, "val")
        self._val_logits.append(cls_logits.detach().float().cpu().numpy())
        self._val_labels.append(batch["label"].detach().float().cpu().numpy())

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        logits = np.concatenate(self._val_logits)
        labels = np.concatenate(self._val_labels)
        self._val_logits.clear()
        self._val_labels.clear()
        if labels.sum() == 0 or labels.sum() == len(labels):
            return

        probs = 1.0 / (1.0 + np.exp(-logits))
        # Cheap bootstrap in-loop; the 1000-iteration reference number is computed
        # once over the full OOF in evaluate.py.
        m = compute_metrics(labels, probs, n_iter=self.cfg["eval"]["val_bootstrap_iter"])
        # AUROC is the decision metric -- ppv_90recall is logged for reference only.
        self.log("val/auroc", m["auroc"], prog_bar=True)
        self.log("val/auprc", m["auprc"])
        self.log("val/ppv_90recall", m["ppv_90recall"])

    def configure_optimizers(self):
        opt_cfg = self.cfg["optimizer"]
        lr = float(opt_cfg["lr"])
        wd = float(opt_cfg["weight_decay"])

        # A randomly initialised head and a LoRA adapter want very different step
        # sizes, and putting them in one group is not a stylistic issue -- it broke
        # every timm_vit_lora recipe we tried. Measured on dinov2_vitb (fold0 AUROC,
        # 120 steps): LoRA frozen 0.877, lora_lr 1e-5 0.909, 5e-5 0.937, 1e-4 0.901,
        # 5e-4 (one group, the old behaviour) 0.62-0.71 -- i.e. training the adapter
        # at the head's rate destroyed the pretrained representation, scoring worse
        # than not adapting it at all. ``lora_lr_scale`` defaults to 1.0 so configs
        # that already trained stay bit-for-bit reproducible; new LoRA recipes set 0.1.
        scale = float(opt_cfg.get("lora_lr_scale", 1.0))
        # Scope to the trained weights: self.parameters() would also hand the frozen
        # EMA copy to the optimizer.
        named = [(n, q) for n, q in self.model.named_parameters() if q.requires_grad]
        adapters = [q for n, q in named if n.endswith((".A", ".B"))]
        rest = [q for n, q in named if not n.endswith((".A", ".B"))]
        if scale != 1.0 and adapters:
            groups = [{"params": adapters, "lr": lr * scale}, {"params": rest, "lr": lr}]
            logger.info("optimizer: %d adapter tensors at lr %.1e, %d others at lr %.1e",
                        len(adapters), lr * scale, len(rest), lr)
        else:
            groups = [{"params": [q for _, q in named], "lr": lr}]
        optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.cfg["trainer"]["max_epochs"]),
            eta_min=float(opt_cfg.get("eta_min", 1e-6)),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
