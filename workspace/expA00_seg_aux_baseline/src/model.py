"""Multi-task network: one encoder, a classification head and an auxiliary
segmentation decoder.

``segmentation_models_pytorch`` already supports exactly this shape via
``aux_params``: ``Unet.forward`` returns ``(mask_logits, cls_logits)`` and the
classification head hangs off the deepest encoder feature map.

Only the classification head is used at inference, so ``forward_cls`` skips the
decoder entirely -- that is a real speedup on the 25k-image test stack, not just
tidiness.
"""

from __future__ import annotations

import logging

import re
import sys

import segmentation_models_pytorch as smp
import timm
import torch
from torch import nn

logger = logging.getLogger(__name__)


def _remap_convnextv2(sd: dict) -> dict:
    """SurgeNet ConvNeXtV2 (original facebookresearch layout) -> timm names.

    Unlike PVTv2 this does NOT align positionally: the checkpoint emits every
    ``downsample_layers.*`` first, while timm interleaves each stage's downsample
    with its blocks. So the mapping has to be by name.
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("downsample_layers.0."):
            k = "stem." + k.split(".", 2)[2]
        elif (m := re.match(r"downsample_layers\.(\d+)\.(.+)", k)):
            k = f"stages.{m.group(1)}.downsample.{m.group(2)}"
        elif (m := re.match(r"stages\.(\d+)\.(\d+)\.(.+)", k)):
            i, j, rest = m.groups()
            rest = (rest.replace("dwconv", "conv_dw")
                        .replace("pwconv1", "mlp.fc1").replace("pwconv2", "mlp.fc2")
                        .replace("grn.gamma", "mlp.grn.weight").replace("grn.beta", "mlp.grn.bias"))
            k = f"stages.{i}.blocks.{j}.{rest}"
        elif k.startswith("norm."):
            k = "head." + k  # final pre-classifier norm
        # timm stores GRN as (C,), the reference implementation as (1, 1, 1, C).
        out[k] = v.reshape(-1) if k.endswith(("grn.weight", "grn.bias")) else v
    return out


def _remap_positional(sd: dict, ref: dict) -> dict:
    """Pair checkpoint tensors with ``ref`` by position, asserting every shape.

    Valid only for checkpoints that enumerate parameters in timm's own order --
    verified for SurgeNet PVTv2 (332 vs 332, all shapes agree). A single shape
    mismatch means the architecture guess is wrong, so this fails loudly rather
    than silently training a partly-random backbone.
    """
    ck = [k for k in sd if not k.startswith(("head", "norm."))]
    rk = list(ref)
    if len(ck) != len(rk):
        raise RuntimeError(f"positional remap: {len(ck)} checkpoint vs {len(rk)} model tensors")
    out = {}
    for r, c in zip(rk, ck):
        if ref[r].shape != sd[c].shape:
            raise RuntimeError(f"positional remap: {r} {tuple(ref[r].shape)} != {c} {tuple(sd[c].shape)}")
        out[r] = sd[c]
    return out


def _to_featurelist_keys(sd: dict, target: dict) -> dict:
    """timm's canonical names -> the flattened names smp's ``tu-`` encoder uses.

    ``FeatureListNet`` splices the top-level container index into the name with an
    underscore (``stem.0`` -> ``stem_0``), leaving everything deeper untouched.
    """
    out = {re.sub(r"^([a-z_]+)\.(\d+)\.", r"\1_\2.", k): v for k, v in sd.items()}
    if not set(out) <= set(target):
        raise RuntimeError(f"flatten produced unknown keys: {sorted(set(out) - set(target))[:5]}")
    return out


def load_surgenet_encoder(encoder, weights_path: str, arch: str) -> None:
    """Load SurgeNet self-supervised weights into an smp ``tu-`` timm encoder.

    Strict: any missing or unexpected tensor raises. That strictness is what
    caught the LayerScale rename on SurgeNetXL and the head-dropout mismatch in
    the container, so it stays.
    """
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = sd.get("teacher", sd.get("model", sd.get("state_dict", sd)))
    sd = {k.replace("backbone.", "", 1): v for k, v in sd.items() if not k.startswith("head.")}

    target = encoder.model.state_dict()
    if arch == "convnextv2":
        sd = _remap_convnextv2(sd)
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
        sd = _to_featurelist_keys(sd, target)
    elif arch == "pvt_v2":
        sd = _remap_positional(sd, target)
    else:
        raise ValueError(f"unknown surgenet encoder arch: {arch}")

    missing, unexpected = encoder.model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{arch} <- {weights_path}: missing={missing[:5]} unexpected={unexpected[:5]}")
    logger.info("loaded SurgeNet %s encoder from %s (%d tensors, clean)", arch, weights_path, len(sd))


class MultiTaskNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "tu-tf_efficientnetv2_s",
        encoder_weights: str | None = "imagenet",
        seg_channels: int = 2,
        dropout: float = 0.3,
        pooling: str = "avg",
        encoder_weights_path: str | None = None,
        encoder_ckpt_arch: str | None = None,
    ):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=seg_channels,
            aux_params={"classes": 1, "dropout": dropout, "pooling": pooling},
        )
        if encoder_weights_path:
            load_surgenet_encoder(self.net.encoder, encoder_weights_path, encoder_ckpt_arch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (seg_logits [B, C, H, W], cls_logits [B])."""
        seg_logits, cls_logits = self.net(x)
        return seg_logits, cls_logits.squeeze(-1)

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        """Classification only -- encoder + head, decoder skipped."""
        features = self.net.encoder(x)
        return self.net.classification_head(features[-1]).squeeze(-1)


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    family = m.get("family", "unet_multitask")
    if family == "vit_lora":
        return DinoV3LoRANet(
            model_name=m["encoder_name"],
            img_size=cfg["data"]["image_size"],
            rank=m.get("lora_rank", 32),
            alpha=m.get("lora_alpha", 64),
            dropout=m.get("dropout", 0.0),
        )
    if family == "timm_vit_lora":
        return TimmViTLoRANet(
            model_name=m["encoder_name"],
            weights_path=m.get("weights_path"),
            img_size=cfg["data"]["image_size"],
            rank=m.get("lora_rank", 32),
            alpha=m.get("lora_alpha", 64),
            dropout=m.get("dropout", 0.0),
        )
    if family == "surgenet_lora":
        return SurgeNetLoRANet(
            arch=m.get("arch", "vit_large"),
            weights_path=m["weights_path"],
            img_size=cfg["data"]["image_size"],
            rank=m.get("lora_rank", 32),
            alpha=m.get("lora_alpha", 64),
            dinov3_repo=m.get("dinov3_repo", DINOV3_REPO),
            dropout=m.get("dropout", 0.0),
        )
    if family != "unet_multitask":
        raise ValueError(f"unknown model.family: {family}")
    return MultiTaskNet(
        encoder_name=m["encoder_name"],
        encoder_weights=m.get("encoder_weights", "imagenet"),
        seg_channels=m.get("seg_channels", 2),
        dropout=m.get("dropout", 0.3),
        pooling=m.get("pooling", "avg"),
        encoder_weights_path=m.get("encoder_weights_path"),
        encoder_ckpt_arch=m.get("encoder_ckpt_arch"),
    )


class LoRALinear(nn.Module):
    """``base(x) + B @ A @ x * (alpha/rank)`` with ``base`` frozen.

    Matches the RARE25 winner's adapter (rank 32, alpha 64, applied to every
    nn.Linear in the backbone).
    """

    def __init__(self, base: nn.Linear, rank: int = 32, alpha: int = 64):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)  # B stays zero -> identity at init
        self.scale = alpha / rank

    # The DINOv3 attention module introspects its Linear layers (in_features etc.),
    # so the wrapper has to look like the layer it replaces.
    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + nn.functional.linear(nn.functional.linear(x, self.A), self.B) * self.scale


def _add_lora(module: nn.Module, rank: int, alpha: int) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            n += 1
        else:
            n += _add_lora(child, rank, alpha)
    return n


class DinoV3LoRANet(nn.Module):
    """DINOv3 ViT with LoRA adapters -- classification only.

    Deliberately has no segmentation decoder: ViT patch features are single-scale
    and bolting a decoder on adds risk for little gain, while the point of this
    member is *diversity* for the noisy-OR ensemble (different architecture,
    different pretraining corpus) rather than reproducing the seg-aux recipe.
    Its ``forward`` therefore returns ``None`` for the segmentation logits.

    Only ~4% of parameters train, and the frozen base is shared across folds, so
    N members cost one base plus N small adapters instead of N full checkpoints.
    """

    def __init__(
        self,
        model_name: str = "vit_large_patch16_dinov3.lvd1689m",
        img_size: int = 384,
        rank: int = 32,
        alpha: int = 64,
        dropout: float = 0.0,
        pretrained: bool = True,
        base_weights_path: str | None = None,
    ):
        super().__init__()
        # Grand Challenge evaluates with no network. Members ship a complete
        # state_dict, so the container must construct the backbone WITHOUT asking
        # the hub for weights -- otherwise the first thing it does at inference is
        # try to download and die.
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=1, img_size=img_size)
        if base_weights_path:
            # The frozen DINOv3 backbone travels as ONE shared file; each member ships
            # only its LoRA adapters and head. It has to be loaded here, before
            # _add_lora rewrites every nn.Linear into LoRALinear -- afterwards the
            # keys are "...qkv.base.weight" and a plain timm state_dict no longer fits.
            base = torch.load(base_weights_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.net.load_state_dict(base, strict=False)
            missing = [k for k in missing if not k.startswith("head")]
            if missing or unexpected:
                raise RuntimeError(f"DINOv3 base {base_weights_path}: missing={missing[:5]} unexpected={unexpected[:5]}")
        for p in self.net.parameters():
            p.requires_grad_(False)
        n = _add_lora(self.net.blocks, rank, alpha)
        for p in self.net.get_classifier().parameters():
            p.requires_grad_(True)
        if dropout > 0:
            self.net.head_drop = nn.Dropout(dropout)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info("DinoV3LoRANet %s: %d LoRA layers, %.1fM/%.1fM trainable (%.2f%%)",
                    model_name, n, trainable / 1e6, total / 1e6, 100 * trainable / total)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        return None, self.net(x).squeeze(-1)

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def lora_state_dict(model: nn.Module) -> dict:
    """Only the adapter + head tensors -- what has to ship per ensemble member."""
    return {k: v for k, v in model.state_dict().items()
            if (".A" in k or ".B" in k or "head" in k) and "base" not in k}


# SurgeNetXL ships DINOv3 backbones trained on surgical/endoscopic video. They are
# NOT loadable through timm: timm's dinov3 uses 4 register tokens and split
# q_bias/v_bias, while SurgeNetXL has no register tokens, a fused qkv.bias, and a
# different RoPE period table (verified: timm's rope.periods does not match).
# The official implementation loads them strictly, so we build through that.
DINOV3_REPO = "/data4/src/shunsuke/RARE/dinov3"


def _remap_surgenet(sd: dict) -> dict:
    """SurgeNetXL uses DINOv2-era LayerScale names; the repo expects ls1/ls2."""
    out = {}
    for k, v in sd.items():
        k = re.sub(r"^blocks\.(\d+)\.gamma_1$", r"blocks.\1.ls1.gamma", k)
        k = re.sub(r"^blocks\.(\d+)\.gamma_2$", r"blocks.\1.ls2.gamma", k)
        out[k] = v
    return out


class SurgeNetLoRANet(nn.Module):
    """DINOv3 ViT pretrained on SurgeNetXL, adapted with LoRA. Classification only.

    LayerScale must be enabled (``layerscale_init``) or the gamma tensors are
    dropped and the residual branches run unscaled -- a silent, severe corruption
    of the pretrained weights. The load is asserted strict for exactly that reason.
    """

    def __init__(
        self,
        arch: str = "vit_large",
        weights_path: str = "DINOv3_ViTl16_size336_SurgeNetXL.pth",
        img_size: int = 336,
        rank: int = 32,
        alpha: int = 64,
        dinov3_repo: str = DINOV3_REPO,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dinov3_repo not in sys.path:
            sys.path.insert(0, dinov3_repo)
        from dinov3.models.vision_transformer import vit_base, vit_large  # noqa: E402

        ctor = {"vit_base": vit_base, "vit_large": vit_large}[arch]
        self.backbone = ctor(patch_size=16, img_size=img_size, layerscale_init=1.0e-5)
        sd = _remap_surgenet(torch.load(weights_path, map_location="cpu", weights_only=False))
        self.backbone.load_state_dict(sd, strict=True)   # strict: catch any silent mismatch

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        n = _add_lora(self.backbone.blocks, rank, alpha)

        dim = self.backbone.embed_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1)) if dropout > 0 else nn.Linear(dim, 1)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info("SurgeNetLoRANet %s @%d: %d LoRA layers, %.1fM/%.1fM trainable (%.2f%%)",
                    arch, img_size, n, trainable / 1e6, total / 1e6, 100 * trainable / total)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        return None, self.head(self.backbone(x)).squeeze(-1)

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)



def _fit_sam_state(sd: dict, target: dict) -> dict:
    """Resize a SAM ViT checkpoint (native 1024 px) to the target resolution.

    Two things are tied to the training grid in SAM's image encoder and must be
    resampled when we run at 384: the NHWC absolute ``pos_embed`` (64x64 -> 24x24)
    and the decomposed relative-position tables ``rel_pos_h/w`` of the four
    global-attention blocks (length 2*64-1 -> 2*24-1). The windowed blocks use a
    14-px window in both cases and already match. Everything else loads as-is.
    """
    import torch.nn.functional as F
    from timm.layers import resample_abs_pos_embed_nhwc
    out = {}
    for k, v in sd.items():
        t = target.get(k)
        if t is None or t.shape == v.shape:
            out[k] = v
        elif k == "pos_embed" and v.ndim == 4:
            out[k] = resample_abs_pos_embed_nhwc(v, new_size=list(t.shape[1:3]))
        elif "rel_pos" in k and v.ndim == 2:
            out[k] = F.interpolate(v.T[None], size=t.shape[0], mode="linear",
                                   align_corners=False)[0].T
        else:
            out[k] = v          # let the strict check report it
    return out


class TimmViTLoRANet(nn.Module):
    """Any timm ViT + LoRA, optionally initialised from a local checkpoint.

    Used for the DINOv2/SurgeNetXL weights, which -- unlike the DINOv3 ones --
    match timm's layout directly. ``strict_load`` stays True by default so an
    architecture mismatch fails loudly instead of silently training from a
    partially-random backbone; ``allow_unexpected`` lists SSL-only tensors
    (mask_token, dino_head.*) that legitimately have no place in a classifier.
    """

    def __init__(
        self,
        model_name: str,
        weights_path: str | None = None,
        img_size: int = 336,
        rank: int = 32,
        alpha: int = 64,
        dropout: float = 0.0,
        allow_unexpected: tuple[str, ...] = ("mask_token", "dino_head", "ibot_head", "register_tokens", "head.mlp", "head.last_layer"),
        pretrained: bool | None = None,
    ):
        super().__init__()
        # Same offline constraint as DinoV3LoRANet: default to "fetch weights only
        # when no local checkpoint was given", but let the container force it off.
        if pretrained is None:
            pretrained = weights_path is None
        kw = {} if model_name.startswith("sam2_hiera") else {"img_size": img_size}
        if model_name == "sam3_pe_trunk":
            # SAM3 / Medical-SAM3 image trunk: a Perception-Encoder ViT that timm has no
            # registered name for (32 layers, MLP 4736, RoPE, pre-norm, no cls token).
            # timm's Eva class implements exactly that family, so build it by hand.
            from timm.models.eva import Eva
            self.net = Eva(img_size=img_size, patch_size=14, embed_dim=1024, depth=32, num_heads=16,
                           mlp_ratio=4736 / 1024, qkv_bias=True, class_token=False, use_rot_pos_emb=True,
                           use_abs_pos_emb=True, use_pre_transformer_norm=True,
                           use_post_transformer_norm=False, use_fc_norm=False,
                           ref_feat_shape=(24, 24), num_classes=1, global_pool="avg")
        else:
            self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=1, **kw)
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu", weights_only=False)
            sd = sd.get("teacher", sd.get("model", sd.get("state_dict", sd)))
            # Checkpoints trained at a different resolution (e.g. SurgeNetXL DINOv1 at
            # 224) carry a pos_embed of the wrong length. timm's standard resampler
            # handles it; without this the load would fail on shape rather than
            # silently, but we would lose the weights entirely.
            if any(k.startswith("image_encoder.trunk.") for k in sd):
                # SAM2 / MedSAM2 / surgical-SAM2 checkpoints carry the whole video model;
                # the image trunk is a Hiera that timm knows as sam2_hiera_*. Keep the
                # trunk, drop the prefix, and rename its MLP (layers.0/1 -> fc1/fc2).
                import re as _re
                from timm.models.hiera import checkpoint_filter_fn as _hiera_filter
                P = "image_encoder.trunk."
                sd = {_re.sub(r"\.mlp\.layers\.(\d)\.", lambda m: f".mlp.fc{int(m.group(1))+1}.", k[len(P):]): v
                      for k, v in sd.items() if k.startswith(P)}
                sd = _hiera_filter(sd, self.net)
            if any("rel_pos" in k for k in sd):
                # timm's SAM filter handles the "image_encoder." prefix of original
                # SAM / MedSAM checkpoints; then fit the grid-dependent tensors.
                from timm.models.vision_transformer_sam import checkpoint_filter_fn as _sam_filter
                sd = _sam_filter(sd, self.net)
                sd = _fit_sam_state(dict(sd), self.net.state_dict())
            pe = sd.get("pos_embed")
            tgt = getattr(self.net, "pos_embed", None)
            if pe is not None and tgt is not None and pe.shape != tgt.shape and pe.ndim == 3:
                from timm.layers import resample_abs_pos_embed
                grid = img_size // self.net.patch_embed.patch_size[0]
                sd = dict(sd)
                sd["pos_embed"] = resample_abs_pos_embed(
                    pe, new_size=[grid, grid],
                    num_prefix_tokens=getattr(self.net, "num_prefix_tokens", 1), verbose=False)
                logger.info("resampled pos_embed %s -> %s", tuple(pe.shape), tuple(sd["pos_embed"].shape))
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            missing = [k for k in missing if not k.startswith("head")]
            unexpected = [k for k in unexpected if not any(k.startswith(a) for a in allow_unexpected)]
            if missing or unexpected:
                raise RuntimeError(
                    f"{model_name} <- {weights_path}: missing={missing[:5]} unexpected={unexpected[:5]}"
                )
            logger.info("loaded %s from %s (clean)", model_name, weights_path)

        for p in self.net.parameters():
            p.requires_grad_(False)
        n = _add_lora(self.net.blocks, rank, alpha)
        for p in self.net.get_classifier().parameters():
            p.requires_grad_(True)
        if dropout > 0:
            self.net.head_drop = nn.Dropout(dropout)
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        logger.info("TimmViTLoRANet %s @%d: %d LoRA layers, %.1fM/%.1fM (%.2f%%)",
                    model_name, img_size, n, tr / 1e6, tot / 1e6, 100 * tr / tot)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        return None, self.net(x).squeeze(-1)

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
