#!/usr/bin/env bash
# Retrain the all-data models whose best-epoch snapshot was lost with its instance.
#
# Each entry is "config:epoch", the epoch being that config's own 5-fold optimum.
# These are pure recovery: the folds, the OOF and therefore the calibration all
# already exist -- only the shipped all-data checkpoint is at the wrong endpoint,
# costing 0.007-0.013 val AUROC per member.
#
# Waits for GPU room rather than assuming it: the fold sweeps are still running and
# we must not exceed the two-GPU budget on this box.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GPU=${GPU:-2}
NEED_MB=${NEED_MB:-13000}
JOBS=${JOBS:-"surgenet_vitb_strongaug:10 dinov3_vitl:14 surgenet_vitl:8"}
for job in $JOBS; do
  cfg=${job%%:*}; ep=${job##*:}
  f=workspace/expA00_seg_aux_baseline/config/$cfg.yaml
  [ -f "$f" ] || { echo "$(printf '%-28s' "$cfg") no config -- skip"; continue; }
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    [ $((tot-used)) -ge "$NEED_MB" ] && break
    sleep 300
  done
  echo "$(date +%H:%M) $(printf '%-28s' "$cfg") all-data to epoch $ep on gpu$GPU"
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python workspace/expA00_seg_aux_baseline/src/train_all.py \
      --config "$f" --epochs "$ep" --override data.use_evc=true \
    && echo "  DONE $cfg@$ep" || echo "  FAILED $cfg@$ep"
done
echo ALL_DATA_QUEUE_COMPLETE
