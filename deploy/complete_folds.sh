#!/usr/bin/env bash
# Finish folds 1-4 for recipes that only have fold0.
#
# Two payoffs, only one of which is the composition search:
#  * calibration: the affine (t,b) per member is currently fitted on 619 images for
#    these recipes and on 3095 for the rest -- unequal, and the small fits are noisy;
#  * ranking: fold0 is the easy fold, so single-fold members carry inflated scores
#    and sit in the wrong tier of the manifest the time-budget guard trims from.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GPU=${GPU:?}; FOLDS=${FOLDS:-"1 2 3 4"}; NEED_MB=${NEED_MB:-11000}
SRC=workspace/expA00_seg_aux_baseline/src; CFGD=workspace/expA00_seg_aux_baseline/config
PY=.venv/bin/python
free_mb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$GPU" | awk -F, '{gsub(/ /,""); print $2-$1}'; }
for cfg in "$@"; do
  f=$CFGD/$cfg.yaml; [ -f "$f" ] || { echo "$cfg: no config -- skip"; continue; }
  name=$($PY -c "import yaml;print(yaml.safe_load(open('$f'))['experiment']['name'])"); R=results/$name
  for fold in $FOLDS; do
    [ -f "$R/fold$fold/training_log.json" ] && continue
    pgrep -f "train.py --config $f --fold $fold\b" >/dev/null && continue
    while [ "$(free_mb)" -lt "$NEED_MB" ]; do sleep 120; done
    resume=""; [ -d "$R/fold$fold" ] && resume="--resume"
    echo "$(date +%H:%M) $cfg fold$fold on gpu$GPU $resume"
    CUDA_VISIBLE_DEVICES=$GPU $PY $SRC/train.py --config "$f" --fold "$fold" $resume \
        --snapshot-epochs 5 10 15 --override trainer.max_epochs=20 \
        > "logs/fold_${cfg}_f${fold}.log" 2>&1 \
      && echo "$(date +%H:%M) DONE $cfg fold$fold" || echo "  FAILED $cfg fold$fold"
  done
  # refresh this member's OOF as soon as all five folds exist
  if [ "$(ls -d $R/fold*/training_log.json 2>/dev/null | wc -l)" -ge 5 ]; then
    [ -f "$R/oof.csv" ] && mv "$R/oof.csv" "$R/oof_fold0only.csv"
    CUDA_VISIBLE_DEVICES=$GPU $PY $SRC/predict_oof.py --exp-dir "$R" > "logs/oof_${cfg}.log" 2>&1 \
      && echo "$(date +%H:%M) OOF-5FOLD $cfg" || echo "  OOF FAILED $cfg"
  fi
done
echo "FOLDS_COMPLETE gpu$GPU: $*"
