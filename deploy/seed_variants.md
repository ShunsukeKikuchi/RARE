# Growing the ensemble with all-data models

An all-data run costs ~1/5 of a 5-fold sweep, so once a config's fold runs exist
(they supply the calibration parameters and the endpoint), extra members are cheap:

```bash
CFG=surgenet_vitb
for SEED in 101 202 303; do
  python workspace/expA00_seg_aux_baseline/src/train_all.py \
      --config workspace/expA00_seg_aux_baseline/config/$CFG.yaml \
      --epochs <that config's measured best epoch> \
      --override experiment.name=${CFG}_seed$SEED experiment.seed=$SEED data.use_evc=true
done
```

Each variant reuses the config's calibration (t, b) — same architecture, same
recipe, so the score scale is unchanged.

Measured endpoints (mean best epoch over folds), used per config rather than one
global number, because they differ by a factor of ~4:

| config                     | endpoint |
|----------------------------|----------|
| convnext_dinov3_seg        | 24       |
| resnext101_swsl            | 12       |
| dinov3_vitl                | 11       |
| maxvit_seg                 | 10       |
| dinov3_vitl_strongaug      |  9       |
| resnext101_swsl_strongaug  |  8       |
| surgenet_vitl (lr halved)  | re-measure |
