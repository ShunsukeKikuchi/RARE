# Fold definitions

Every experiment reads its split from a versioned CSV, selected by `data.folds_csv`
in the recipe config. Old versions are never deleted, so a result can always be traced
back to the split it was measured on.

## v1 — centre x label stratified 5-fold (`v1/folds.csv`)

```bash
python workspace/fold/v1/generate_folds.py
```

Covers `data/RARE25/center_{1,2}/{ndbe,neo}/*.png` = 3,095 images
(2,937 non-dysplastic, 158 neoplasia). The script is seeded and asserts the per-centre
counts, so it reproduces `folds.csv` byte for byte; the CSV itself is not in git
because it is an index of the challenge data.

### Why StratifiedKFold, and not something else

| Candidate | Used | Reason |
|---|---|---|
| GroupKFold by patient | **impossible** | RARE25 filenames are bare UUIDs — there is no patient identifier, so images of one patient cannot be kept in one fold. |
| GroupKFold by centre | no | There are only two centres, which caps the design at two folds. |
| Plain StratifiedKFold on the label | no | Centre is a strong confounder (11.9% positive in centre 2 against 2.7% in centre 1), so label-only stratification lets the centre mix drift between folds. |
| **StratifiedKFold on `centre\|label`** | **yes** | Every fold keeps both centres at their natural prevalence, which is the closest we can get to a fair split without patient IDs. |

Each validation fold holds about 619 images with 31-32 neoplasia.

### Known limitation

Patient-level leakage across folds **cannot be excluded**, and the gap between our
out-of-fold numbers and the challenge leaderboard suggests it is present. Treat
out-of-fold absolutes as optimistic; they are useful for ranking recipes against each
other, not as a prediction of test performance.

### Domain shift

Leave-one-centre-out is evaluated separately (`src/loco.py`) as a domain-shift probe.
It is deliberately not part of the fold definition — see the table above.

## External data

The 359 external images (EVC_Barretts_FullSet, EDD2020 oesophageal subset) are indexed
by `src/build_external_index.py` and joined onto the **training** side of every fold.
They never appear in validation.
