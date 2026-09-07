# Remote run matrix — one vast.ai instance per config

Deadline: 2026-09-08 09:00 JST. Prefer wall-clock over $/TFLOP.
GPU: RTX 4090 (24 GB) preferred, RTX 3090 acceptable. 24 GB is enough —
ViT-L LoRA @336 peaks at ~9 GB; A100/H100 is wasted money here.

## Per instance

```bash
HF_TOKEN=hf_xxx bash deploy/vast_bootstrap.sh
cd /workspace/RARE
CONFIG=workspace/expA00_seg_aux_baseline/config/<CFG>.yaml \
  JOBS_PER_GPU=2 ./workspace/expA00_seg_aux_baseline/run.sh sweep_cfg <NAME> 0
```
With a 2-GPU instance, run two configs concurrently (`... sweep_cfg <NAME> 0` and `... 1`).

## Configs (all EVC-free: EVC is the held-out external test set)

| # | config | arch / pretraining | seg | img | est. 5-fold on 4090 |
|---|--------|--------------------|-----|-----|---------------------|
| 1 | `surgenet_vitl`         | DINOv3 ViT-L / **SurgeNetXL** (surgical) | – | 336 | ~1.5 h |
| 2 | `surgenet_vitb`         | DINOv3 ViT-B / **SurgeNetXL**            | – | 336 | ~0.6 h |
| 3 | `dinov3_vitl`           | DINOv3 ViT-L / LVD-1689M                 | – | 336 | ~1.5 h |
| 4 | `dinov3_vitb`           | DINOv3 ViT-B / LVD-1689M                 | – | 336 | ~0.6 h |
| 5 | `convnext_dinov3_seg`   | ConvNeXt-S / **DINOv3** LVD-1689M        | ✓ | 384 | ~0.8 h |
| 6 | `convnext_dinov3_noseg` | ConvNeXt-S / DINOv3 LVD-1689M            | – | 384 | ~0.8 h |
| 7 | `maxvit_seg`            | MaxViT-S / ImageNet (RARE25 2nd place)   | ✓ | 384 | ~1.0 h |
| 8 | `effnet_noseg_evcfree`  | EfficientNetV2-S / IN21k                 | – | 384 | ~0.6 h |

Already trained locally (do NOT redo): `a_rare_only`, `b_seg_aux`(+EMA),
`a2_ext_cls`, `c_no_evc_ema`, ep15/ep20 schedule probes, LOCO.

5 and 6 together isolate the segmentation signal on a *different* architecture
than the effnet pair did — the one axis where seg was significant (LOCO +0.034).

## After each run finishes
Copy `results/<NAME>/fold*/best_model.ckpt` back (or push to the HF repo), then
locally: OOF -> EVC external eval -> noisy-OR + calibration.
