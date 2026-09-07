"""Augmentation pipelines.

Deliberately weak to start with (per the repo rule: strengthen only once
overfitting is observed). Endoscopy frames have no canonical orientation, so
both flips and free rotation are label-preserving.

Geometry -- two things here are measured, not guessed, and should not be
"cleaned up":

1. **Square resize, not aspect-preserving.** Grand Challenge hands the algorithm
   a stack of 512x512 frames built by squashing the native images
   anisotropically: on last year's real example stack the endoscope octagon has
   width/height 1.00, versus 1.25 in the native PNGs, at an identical 0.93 fill
   ratio. Padding to preserve aspect would train on a frame shape that never
   occurs at inference.

2. **The 512 pre-scale.** Because GC downsamples to 512 first, the container
   necessarily sees native -> 512 -> model_size, while a naive local pipeline
   would see native -> model_size directly. Measured on 200 images, those two
   paths give Spearman 0.96 and up to 0.40 absolute difference in predicted
   probability; routing through 512 reproduces the container bit-for-bit
   (max abs diff 0.0, versus 0.009 from fp16 alone). So we replicate GC's
   downsample here and train on exactly the resampling the model will be served.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Frame size Grand Challenge stacks images at. None disables the pre-scale.
GC_FRAME_SIZE = 512


def build_transforms(
    image_size: int,
    train: bool,
    gc_prescale: int | None = GC_FRAME_SIZE,
    preset: str = "weak",
) -> A.Compose:
    """``preset`` selects the augmentation strength.

    Presets exist as an *ensemble diversity axis*, not because weak was failing.
    The test set spans 12 unseen centres with no standardised imaging protocol, so
    aggressive augmentation encodes the right prior. "weak" and "strong" differ in
    degree; "photometric" and "geometric" split that same budget into two roughly
    orthogonal halves, so two members trained on them are invariant to different
    things and disagree in different places -- which is what an ensemble is paid for.

      weak        mild everything
      strong      aggressive everything
      photometric flips only, but heavy colour/illumination/noise/JPEG
      geometric   heavy warping/rotation/occlusion, but near-neutral colour

    The base geometry (square resize + the 512 pre-scale) is identical in every
    preset because that part is dictated by how Grand Challenge builds the stack,
    not by taste.
    """
    pre = [A.Resize(gc_prescale, gc_prescale)] if gc_prescale else []

    if train and preset == "photometric":
        # The dominant shift across unseen centres is the imaging chain -- scope
        # model, light source, white balance, JPEG pipeline -- not anatomy. This
        # member specialises in being invariant to that and leaves shape alone.
        aug = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03),
                     rotate=(-10, 10), border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.45, p=0.9),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=45, val_shift_limit=30, p=0.8),
            A.RandomGamma(gamma_limit=(60, 160), p=0.6),
            A.OneOf([A.CLAHE(clip_limit=4.0, p=1.0),
                     A.RandomToneCurve(scale=0.3, p=1.0)], p=0.5),
            A.OneOf([A.GaussNoise(p=1.0), A.ISONoise(p=1.0),
                     A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0)], p=0.5),
            A.OneOf([A.MotionBlur(blur_limit=7, p=1.0), A.GaussianBlur(blur_limit=7, p=1.0),
                     A.Defocus(radius=(1, 4), p=1.0)], p=0.4),
            A.ImageCompression(quality_range=(30, 95), p=0.6),
        ]
    elif train and preset == "geometric":
        # The complement: endoscope pose, insufflation and mucosal deformation vary
        # far more than colour within a centre. Colour is left near-neutral so this
        # member's errors do not line up with the photometric one's.
        aug = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.75, 1.3), translate_percent=(-0.12, 0.12),
                     rotate=(-60, 60), shear=(-12, 12), border_mode=0, p=0.9),
            A.OneOf([A.ElasticTransform(alpha=200, sigma=8, p=1.0),
                     A.GridDistortion(num_steps=6, distort_limit=0.4, p=1.0),
                     A.OpticalDistortion(distort_limit=0.6, p=1.0)], p=0.7),
            A.Perspective(scale=(0.03, 0.09), p=0.3),
            A.CoarseDropout(num_holes_range=(1, 10), hole_height_range=(0.06, 0.20),
                            hole_width_range=(0.06, 0.20), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        ]
    elif train and preset == "strong":
        aug = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.85, 1.18), translate_percent=(-0.08, 0.08),
                     rotate=(-30, 30), shear=(-6, 6), border_mode=0, p=0.8),
            A.OneOf([A.ElasticTransform(alpha=120, sigma=6, p=1.0),
                     A.GridDistortion(num_steps=5, distort_limit=0.25, p=1.0),
                     A.OpticalDistortion(distort_limit=0.4, p=1.0)], p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.6),
            A.RandomGamma(gamma_limit=(70, 140), p=0.4),
            A.CLAHE(clip_limit=3.0, p=0.3),
            A.OneOf([A.GaussNoise(p=1.0), A.ISONoise(p=1.0)], p=0.3),
            A.OneOf([A.MotionBlur(blur_limit=5, p=1.0), A.GaussianBlur(blur_limit=5, p=1.0),
                     A.Defocus(radius=(1, 3), p=1.0)], p=0.3),
            # Different sites store images through different JPEG pipelines.
            A.ImageCompression(quality_range=(45, 100), p=0.4),
            A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(0.05, 0.15),
                            hole_width_range=(0.05, 0.15), p=0.3),
        ]
    elif train:
        aug = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-20, 20),
                border_mode=0,
                p=0.7,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        ]
    else:
        aug = [A.Resize(image_size, image_size)]

    return A.Compose(pre + aug + [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
