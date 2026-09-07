#!/usr/bin/env bash
# dl2 variant of the recipe runner: folds 1-4 only (fold0 + all-data are done on dl1), one job
# at a time on the 16 GB A4000, host-local venv, CUDA index aligned to nvidia-smi.
#
#   FOLDS="1 2 3 4" GPU=0 PY=/tmp/rare_dl2/venv/bin/python bash deploy/recipe_queue_dl2.sh cfg ...
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # torch enumerates by capability (4090 first); nvidia-smi by bus
GPU=${GPU:?}; PY=${PY:-.venv/bin/python}; FOLDS=${FOLDS:-"1 2 3 4"}; NEED_MB=${NEED_MB:-9000}
SRC=workspace/expA00_seg_aux_baseline/src; CFGD=workspace/expA00_seg_aux_baseline/config
free_mb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$GPU" | awk -F, '{gsub(/ /,""); print $2-$1}'; }
for cfg in "$@"; do
  f=$CFGD/$cfg.yaml; [ -f "$f" ] || { echo "$(date +%H:%M) $cfg: no config -- skip"; continue; }
  name=$($PY -c "import yaml;print(yaml.safe_load(open('$f'))['experiment']['name'])"); R=results/$name
  for fold in $FOLDS; do
    [ -f "$R/fold$fold/training_log.json" ] && continue
    if pgrep -f "train.py --config $f --fold $fold\b" >/dev/null; then echo "  $cfg fold$fold already running elsewhere -- skip"; continue; fi
    while [ "$(free_mb)" -lt "$NEED_MB" ]; do sleep 120; done
    resume=""; [ -d "$R/fold$fold" ] && resume="--resume"
    echo "$(date +%H:%M) $cfg: fold$fold on dl2 gpu$GPU $resume"
    CUDA_VISIBLE_DEVICES=$GPU $PY $SRC/train.py --config "$f" --fold "$fold" $resume \
        --snapshot-epochs 5 10 15 --override trainer.max_epochs=20 \
        > "logs/dl2_${cfg}_fold${fold}.log" 2>&1 || { echo "  FAILED $cfg fold$fold (logs/dl2_${cfg}_fold${fold}.log)"; continue; }
    echo "$(date +%H:%M) DONE $cfg fold$fold"
  done
done
echo "QUEUE_COMPLETE dl2: $*"
