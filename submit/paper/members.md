Ensemble: 30 members

| # | recipe | family | px | OOF AUROC | val n |
|---|--------|--------|----|-----------|-------|
| 0 | `resnext101_swsl_seg` | seg-aux CNN/hybrid | 384 | 0.9813 | 3095 |
| 1 | `resnext101_swsl_geometric` | seg-aux CNN/hybrid | 384 | 0.9794 | 3095 |
| 2 | `dinov2_vitl_lora2` | foundation-ViT LoRA | 336 | 0.9767 | 3095 |
| 3 | `effnetv2m_geometric` | seg-aux CNN/hybrid | 384 | 0.9733 | 3095 |
| 4 | `effnet_seg_strongaug` | seg-aux CNN/hybrid | 384 | 0.9712 | 3095 |
| 5 | `surgenet_dinov2_vitb_lora2` | foundation-ViT LoRA | 336 | 0.9700 | 3095 |
| 6 | `resnext101_32x4d_swsl_strongaug` | seg-aux CNN/hybrid | 384 | 0.9685 | 3095 |
| 7 | `expA02_surgenet_vitb_strongaug` | SurgeNetXL LoRA | 336 | 0.9682 | 3095 |
| 8 | `dinov2_vitb_lora2` | foundation-ViT LoRA | 336 | 0.9671 | 3095 |
| 9 | `pvtv2_512_strongaug` | seg-aux CNN/hybrid | 512 | 0.9668 | 3095 |
| 10 | `expA02_surgenet_vitb` | SurgeNetXL LoRA | 336 | 0.9658 | 3095 |
| 11 | `pvtv2_strongaug` | seg-aux CNN/hybrid | 384 | 0.9647 | 3095 |
| 12 | `expA00_a_rare_only` | seg-aux CNN/hybrid | 384 | 0.9635 | 3095 |
| 13 | `clip_vitb_lora2` | foundation-ViT LoRA | 384 | 0.9634 | 3095 |
| 14 | `pvtv2_geometric` | seg-aux CNN/hybrid | 384 | 0.9628 | 3095 |
| 15 | `dinov3_vitl_lora_strongaug` | DINOv3 LoRA | 336 | 0.9619 | 3095 |
| 16 | `medical_sam3_pe_lora2` | foundation-ViT LoRA | 336 | 0.9619 | 3095 |
| 17 | `expA00_c_no_evc_ema` | seg-aux CNN/hybrid | 384 | 0.9589 | 3095 |
| 18 | `siglip2_vitb_lora2` | foundation-ViT LoRA | 384 | 0.9588 | 3095 |
| 19 | `medsam2_hiera_t_lora2` | foundation-ViT LoRA | 384 | 0.9558 | 3095 |
| 20 | `eva02_base_lora2` | foundation-ViT LoRA | 336 | 0.9553 | 3095 |
| 21 | `caformer_s36_strongaug` | seg-aux CNN/hybrid | 384 | 0.9539 | 3095 |
| 22 | `expA02_surgenet_vitl` | SurgeNetXL LoRA | 336 | 0.9426 | 3095 |
| 23 | `dinov3_vitl_lora` | DINOv3 LoRA | 336 | 0.9415 | 3095 |
| 24 | `maxvit_seg` | seg-aux CNN/hybrid | 384 | 0.9409 | 3095 |
| 25 | `sam_vitb_lora2` | foundation-ViT LoRA | 384 | 0.9360 | 3095 |
| 26 | `surgisam2_hiera_s_lora2` | foundation-ViT LoRA | 384 | 0.9333 | 3095 |
| 27 | `convnext_dinov3_seg` | seg-aux CNN/hybrid | 384 | 0.8885 | 3095 |
| 28 | `medsam_vitb_lora2` | foundation-ViT LoRA | 384 | 0.8657 | 3095 |
| 29 | `swin_b_strongaug` | seg-aux CNN/hybrid | 384 | 0.9616* | 619 |

*1 recipes are validated on fold0 only (619 images); fold0 is the easiest of the five, so those numbers are optimistic and are not comparable with the 3095-image entries.

By family: DINOv3 LoRA 2, SurgeNetXL LoRA 3, foundation-ViT LoRA 11, seg-aux CNN/hybrid 14
