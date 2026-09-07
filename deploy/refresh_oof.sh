#!/usr/bin/env bash
# Regenerate OOF wherever more folds exist than the stored oof.csv covers.
#
# The recipe queue writes a fold0-only OOF (that is all a recipe has). When a config
# is later completed to 5 folds -- e.g. the pvt_v2 members finished on dl2 -- the
# stale single-fold file stays behind, and it is used for BOTH the affine calibration
# and the member ranking that decides who survives the container's 600 s budget.
# Refreshing costs inference only.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GPU=${GPU:-2}
for d in results/*/; do
  n=$(basename "$d"); [ "${n:0:1}" = "_" ] && continue
  folds=$(ls -d "$d"fold*/training_log.json 2>/dev/null | wc -l)
  [ "$folds" -ge 5 ] || continue
  [ -f "$d/oof.csv" ] && rows=$(( $(wc -l < "$d/oof.csv") - 1 )) || rows=0
  [ "$rows" -ge 3000 ] && continue
  echo "$(date +%H:%M) $n: folds=$folds oof_rows=$rows -> regenerating"
  [ -f "$d/oof.csv" ] && mv "$d/oof.csv" "$d/oof_partial_${rows}.csv"
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python workspace/expA00_seg_aux_baseline/src/predict_oof.py \
      --exp-dir "$d" > "logs/oof_refresh_${n}.log" 2>&1 \
    && echo "  DONE $n" || echo "  FAILED $n"
done
echo REFRESH_COMPLETE
