# A Diversity-First Calibrated Ensemble under Domain Shift

**Team**: Jmees — Shunsuke Kikuchi, Atsushi Kouno (Jmees Inc.) · **Challenge**: RARE26 (Grand Challenge) · **Code**: https://github.com/ShunsukeKikuchi/RARE (MIT)

## 1. Problem framing

The metric is PPV at 90% recall, evaluated by resampling positives to a ~1% prevalence and taking the median over 1000 draws. Two properties of that metric drove every design decision.

**The score is decided by a handful of images.** On our out-of-fold predictions, 42 of 158 positives were assigned p < 0.01, which pushes the 90%-recall threshold down to 0.00002 and lets false positives flood in. Recovering the 16 hardest positives moves PPV@90R from 0.50 to 0.87. Average-case accuracy is almost irrelevant; the tail is everything.

**Local validation cannot rank models on this metric.** With 158 positives, a bootstrap draw at 1% prevalence contains 5-6 positives, so PPV@90R quantises badly: we measured a signal-to-noise ratio of 0.71 and a 95% CI width of 0.35 on the full-cohort variant. AUROC was the most stable of the three candidates (SNR 0.79) but still below 1. We therefore use **AUROC for model selection** and treat PPV@90R as a reported number only, and we lean on design arguments and external evaluation rather than small CV differences.

The dominant risk is domain shift: the test set spans 12 unseen centres, and the organisers identify the train/test gap as the main difficulty of the previous edition. Our strategy is consequently **diversity first** — many decorrelated members rather than one tuned model.

## 2. Data and validation

Training data is RARE25 (3,095 images; 2,937 non-dysplastic, 158 neoplastic) from two centres. Two external, publicly licensed sets supply pixel-level supervision: **EVC_Barretts_FullSet** (100 images, five expert masks each) and **EDD2020** (oesophageal subset).

Filenames are UUIDs with no patient identifier, so patient-level grouping is impossible and we state plainly that patient leakage cannot be excluded. Centre is a strong confounder (11.9% positive in centre 2 versus 2.7% in centre 1), so we use **5-fold stratification on the composite `centre|label` key**, which keeps both centres at their natural ratio in every fold. External data never enters validation; it is added to the training side of every fold. Leave-one-centre-out is measured separately as a domain-shift probe.

## 3. Preprocessing: matching what the platform actually delivers

Two measured facts, not assumptions, fix the preprocessing.

**The platform squashes images to square.** On last year's real input stack the endoscope octagon has width/height 1.00 versus 1.25 in the native images, at identical fill ratio. We therefore train with a plain anisotropic `Resize(S, S)`. Aspect-preserving pad-and-resize looks tidier but teaches the network a frame shape that never occurs at inference.

**The platform downsamples to 512 before stacking.** The container necessarily sees native → 512 → model size, while a naive local pipeline sees native → model size. Across 200 images those paths give Spearman 0.96 and up to 0.40 absolute difference in predicted probability; inserting the 512 stage reproduces container output exactly (max abs difference 0.000, versus 0.009 attributable to fp16). Every training, validation and OOF pass therefore routes through 512.

## 4. Members

Each member is one *recipe* = architecture × pretraining corpus × augmentation policy. Diversity is sought along all three axes.

**Segmentation-auxiliary CNN/hybrid members.** A U-Net decoder is attached to the classifier (`smp` `aux_params`), predicting two channels — Barrett's extent and neoplasia — supervised only where masks exist and masked out elsewhere. Inference uses the classification head only, so the decoder costs nothing at test time. The ablation is instructive: the auxiliary task does **not** help in-domain (−0.004, P=0.66) but helps significantly under shift, **+0.034 (P=0.010) leave-one-centre-out** and +0.031 (P=0.001) on EVC. A middle arm with the external images but no segmentation supervision gained only +0.015 (P=0.088), so the effect comes from the pixel supervision rather than from the extra images. This is the clearest evidence in our study that in-domain CV alone would have discarded a method that works in the deployment regime.

**LoRA-adapted foundation ViTs.** Rank-32 adapters on every attention and MLP projection, base frozen, roughly 5% of parameters trained. Backbones span general (DINOv3), surgical/endoscopic (SurgeNetXL), and promptable-segmentation (SAM, MedSAM, SAM2, Medical-SAM3) pretraining. Because the frozen base is shared, N members cost one base plus N small adapter files. Every checkpoint is loaded with a **strict** state-dict check; that strictness caught a LayerScale rename in SurgeNetXL, a GRN parameter-shape change in ConvNeXtV2, and a head-dropout mismatch that would otherwise have shipped a randomly initialised head.

**Augmentation as a diversity axis.** Rather than one "strong" preset, we split the same budget into two roughly orthogonal halves: `photometric` (flips only, but aggressive colour, illumination, noise and JPEG) and `geometric` (large rotation, scale, shear, elastic and grid distortion, occlusion, with colour near-neutral). Geometry common to all presets — square resize plus the 512 pre-scale — is dictated by the platform, not by taste. Across five presets on one backbone the ordering was geometric 0.9827 > strong 0.9754 > weak 0.9731 > photometric 0.9621 (mean of per-fold best validation AUROC), which argues that invariance to scope pose and mucosal deformation matters more here than invariance to the imaging chain.

## 5. Combining members

**Pooling is a mean of calibrated logits** (a log-opinion pool), weighted by each member's rank on out-of-fold AUROC — the best of K members gets weight K, the worst 1, normalised. Rank weighting was checked the way a subset search should be and usually is not: fitting the weights on one random half of the cohort and scoring the other, over 60 splits, gives FPR@90%TPR 0.0144 against 0.0154 uniform, better in 40 of 60 (binomial P = 0.007). The same protocol rejects *discrete* selection outright — greedy forward selection reaches a higher score on the data it selected on (0.9895 vs 0.9865) and then loses to the full ensemble in 19 of 20 held-out splits. A continuous weighting degrades gracefully when the rank estimate is wrong; dropping a member does not.

We compared the pool against noisy-OR at the operating point the metric is actually read at, rather than on AUROC. At 1% prevalence the score reduces almost exactly to PPV@90R = 0.009 / (0.009 + FPR@90%TPR × 0.99), so FPR at 90% sensitivity is the quantity to compare: mean-logit 0.0143 against noisy-OR 0.0157, a difference well inside the bootstrap CI (P = 0.38). What separates them is the tail. Ranking the sixteen hardest positives by the percentile of negatives each still beats, mean-logit puts the worst one at 20; noisy-OR at 4, mean-of-top-3 at 5, max-member at 3. Every rule that leans further toward "any one member fired" pushes the hardest positive *down*, because the same rule lifts every negative that some minority member got wrong. That ordering, not the aggregate statistic, is why the pool is a mean.

**Calibration** is a two-parameter affine map per member, fitted with `psrcal`'s `AffineCalLogLoss` toward the 1% operating prior, on that member's own held-out predictions. For a single model this is irrelevant (PPV@90R sweeps the threshold, and an affine map on logits is monotone); it exists solely to put members on a common scale before averaging.

**Shipped weights are trained on all data**, including EVC. The fold models remain the instrumentation: they supply the calibration parameters and the stopping epoch, which are transplanted to the all-data model of the same recipe. Endpoint transplantation is not a detail — the optimum is architecture-dependent (epoch 5 to 28 across our recipes) and shipping a fixed final epoch instead costs 0.008 to 0.016 validation AUROC on most recipes, more than the gap between pooling rules.

## 6. Inference container

One job is one multi-page TIFF of 384 frames with a 600-second limit, and an over-running job produces no output at all. The container therefore reads frames in slabs via `SimpleITK`'s streaming reader, caches preprocessed tensors once per input resolution, and loads each member exactly once rather than per chunk. Members are ordered by out-of-fold AUROC and a **time-budget guard** stops admitting members once elapsed time plus the slowest observed member would cross 480 s, pooling whatever finished; the first member always runs, so an output always exists.

The container is verified offline (`--network=none`) before every submission. That verification is what caught three defects that would each have scored zero: a weights file that was the right size but corrupt (a race between collection and packaging, now prevented by loading every file at build time), backbones that fetched pretrained weights from the hub at inference, and LoRA members shipping adapters without their frozen base. The second defect concealed the third, which is the general lesson we would pass on: constraints that only exist in production must be exercised early, because a bug can hide behind another.

## 7. What we would flag to a reader

Three of our conclusions are honest negatives. First, inference-side optimisation was a dead end: `torch.compile` gave 1.02×, and ONNX Runtime was both slower (0.27–0.50×) and numerically broken (max |Δp| = 1.00). Second, external evaluation on EVC saturated — every current member scores 0.98–1.00, so it no longer discriminates and only leave-one-centre-out remains as a shift probe. Third, we discarded several recipes as "weak weights" when a linear probe on frozen features later showed one of them to be the *best* backbone available to us; the failure was in our training configuration, not the weights. We record it because the same reasoning error is easy to make under time pressure, and because it means our final member list is a floor, not a ceiling.
