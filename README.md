# RARE26 — Barrett's neoplasia detection

Training and inference code for our RARE26 (Grand Challenge) submission: binary
classification of early neoplasia in Barrett's oesophagus endoscopy images, scored
by PPV at 90% recall under a simulated 1% prevalence.

Released under the [MIT licence](LICENSE).

## What is here

| Path | Contents |
|---|---|
| `workspace/expA00_seg_aux_baseline/src/` | Training: models, data, transforms, LightningModule, OOF prediction, metrics |
| `workspace/expA00_seg_aux_baseline/config/` | One YAML per recipe — every shipped member has one |
| `workspace/fold/` | Fold design and the seeded generator every experiment shares |
| `submit/v001_seg_aux/` | Inference container: `Dockerfile`, `inference.py`, `model/`, build/test/export scripts |
| `submit/v001_seg_aux/build_submission.py` | Assembles the model bundle: picks each recipe's endpoint, fits calibration, ranks and weights members, writes `manifest.json` |
| `submit/paper/` | The method description submitted to the challenge, and its build script |
| `deploy/` | Multi-machine training orchestration and the TensorRT investigation notes |

Model weights, datasets and the assembled submission bundle are **not** in this
repository — see *Weights* below. Two derived index files are also left out because
they enumerate the challenge data; both are regenerated deterministically:

```bash
python workspace/fold/v1/generate_folds.py                       # -> folds.csv
python workspace/expA00_seg_aux_baseline/src/build_external_index.py  # -> external_index.csv
```

## Reproducing a member

```bash
python workspace/expA00_seg_aux_baseline/src/train.py \
    --config workspace/expA00_seg_aux_baseline/config/<recipe>.yaml --fold 0
```

Then the all-data model that actually ships, stopped at the epoch the fold curves
picked:

```bash
python workspace/expA00_seg_aux_baseline/src/train_all.py \
    --config workspace/expA00_seg_aux_baseline/config/<recipe>.yaml \
    --epochs <best epoch> --override data.use_evc=true
```

`deploy/recipe_queue_v5.sh` runs that pair end to end for a list of recipes,
including the out-of-fold prediction the calibration is fitted on.

## Building the container

```bash
python submit/v001_seg_aux/build_submission.py --members all --configs <recipe> ...
cd submit/v001_seg_aux && bash test.sh    # regression test with --network=none
bash export.sh                            # image tarball + model tarball
```

`test.sh` runs the container exactly as the platform does — no network, `/input`
read-only — because several defects here were invisible until it did: weights that
were the right size and corrupt, backbones fetching pretrained weights from the hub
at inference, and precision changes that produced all-NaN output while every member
appeared to run.

## Data

* **RARE25** (challenge training set, 3,095 images) — from the organisers.
* **EVC_Barretts_FullSet**, **EDD2020** — public sets used for the auxiliary
  segmentation supervision; oesophageal subset only for EDD2020.

Expected under `data/` (git-ignored), laid out as
`data/RARE25/center_{1,2}/{ndbe,neo}/*.png` and `data/BarrettsEsophagus/<set>/`.

## Weights

Third-party pretrained checkpoints are **fetched or converted from their own
sources**, never redistributed here — each carries its own licence:

* DINOv3, DINOv2, CLIP, SigLIP2, EVA02, MaxViT, CAFormer, Swin, ResNeXt-SWSL,
  EfficientNet — via `timm`
* SurgeNetXL / SurgeNet — from the SurgeNet authors' release
* SAM, MedSAM, SAM2, MedSAM2, surgical-SAM2, Medical-SAM3 — from their respective
  repositories; `src/model.py` holds the key remapping each one needs

The `weights/` directory holds locally converted copies and is git-ignored.

## Method

`submit/paper/method.pdf`. Briefly: 30 members spanning architecture, pretraining
corpus and augmentation policy, each a segmentation-auxiliary CNN/hybrid or a
LoRA-adapted foundation ViT; combined as a rank-weighted mean of per-member
calibrated logits. Preprocessing reproduces the platform's own anisotropic square
resize and its 512-pixel intermediate, both measured rather than assumed.
