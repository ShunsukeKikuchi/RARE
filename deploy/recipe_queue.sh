#!/usr/bin/env bash
# One "recipe" = fold0 (endpoint + calibration only) -> all-data model at that endpoint.
#
# No 5-fold: the user's call, and it is what makes 30-40 distinct members affordable
# (~1.6 fold-equivalents per recipe instead of ~6.2). fold0 gives the epoch to stop
# the all-data run at, plus 619 held-out predictions for the two-parameter affine
# calibration. What we give up is full-coverage OOF for leave-one-out analysis.
#
#   GPU=1 bash deploy/recipe_queue.sh cfg1 cfg2 ...
#
# Launch discipline (learned the hard way -- two workers OOM'd each other):
#  * launches on one GPU are serialised through a lock, held for LAUNCH_HOLD seconds
#    so the new job has claimed its memory before the next worker measures free space;
#  * the memory requirement scales with the model (ViT-L LoRA @336 needs ~20 GB);
#  * a recipe whose job is already running (e.g. an orphan from a restarted worker)
#    is waited on, not duplicated into the same results dir.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GPU=${GPU:?}; LAUNCH_HOLD=${LAUNCH_HOLD:-150}
SRC=workspace/expA00_seg_aux_baseline/src; CFGD=workspace/expA00_seg_aux_baseline/config
PY=.venv/bin/python
LOCK=/tmp/rare_gpu${GPU}.launch.lock; exec 9>"$LOCK"

need_mb() { case "$1" in
  *vitl*|*large*|*maxvit_b*|*swin_b*|*convnext_b*|*regnety120*|*_512*) echo 20000;;
  *) echo 12000;; esac; }
free_mb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$GPU" \
            | awk -F, '{gsub(/ /,""); print $2-$1}'; }
# run "$@" on this GPU: take the launch lock, wait for room, start, hold the lock, then wait.
run_gpu() { local need=$1 log=$2; shift 2
  flock 9
  while [ "$(free_mb)" -lt "$need" ]; do sleep 120; done
  CUDA_VISIBLE_DEVICES=$GPU "$@" > "$log" 2>&1 & local pid=$!
  sleep "$LAUNCH_HOLD"; flock -u 9
  wait "$pid"; }
wait_if_running() { # $1 = pgrep pattern; block while a matching job runs
  while pgrep -f "$1" >/dev/null; do sleep 120; done; }

for cfg in "$@"; do
  f=$CFGD/$cfg.yaml; [ -f "$f" ] || { echo "$(date +%H:%M) $cfg: no config -- skip"; continue; }
  name=$($PY -c "import yaml;print(yaml.safe_load(open('$f'))['experiment']['name'])")
  R=results/$name; need=$(need_mb "$cfg")

  # --- fold0 ---
  wait_if_running "train.py --config $f --fold 0"
  if [ ! -f "$R/fold0/training_log.json" ]; then
    resume=""; [ -d "$R/fold0" ] && resume="--resume"   # reuse a partial dir, never spawn _001
    echo "$(date +%H:%M) $cfg: fold0 on gpu$GPU (need ${need}MB) $resume"
    run_gpu "$need" "logs/recipe_${cfg}_fold0.log" $PY $SRC/train.py --config "$f" --fold 0 $resume \
        --snapshot-epochs 5 10 15 --override trainer.max_epochs=20 \
      || { echo "  FAILED fold0 $cfg (see logs/recipe_${cfg}_fold0.log)"; continue; }
  fi

  # --- endpoint from fold0's own curve ---
  ep=$($PY - "$R/fold0/metrics.csv" <<'PYE'
import sys, pandas as pd
d=pd.read_csv(sys.argv[1]); c=[x for x in d.columns if 'auroc' in x.lower()]
g=d[['epoch',c[0]]].dropna().groupby('epoch').last()[c[0]]
print(max(3, min(20, int(g.idxmax())+1)))
PYE
)
  [ -n "$ep" ] || { echo "  FAILED endpoint $cfg"; continue; }

  # --- all-data at that endpoint (EVC included: the shipped model sees everything) ---
  wait_if_running "train_all.py --config $f "
  if [ ! -f "$R/all/epoch$(printf '%02d' "$ep").ckpt" ]; then
    echo "$(date +%H:%M) $cfg: all-data to epoch $ep on gpu$GPU"
    run_gpu "$need" "logs/recipe_${cfg}_all.log" $PY $SRC/train_all.py --config "$f" --epochs "$ep" \
        --override data.use_evc=true \
      || { echo "  FAILED all-data $cfg"; continue; }
  fi

  # --- fold0 OOF for calibration (small; no lock needed) ---
  [ -f "$R/oof.csv" ] || CUDA_VISIBLE_DEVICES=$GPU $PY $SRC/predict_oof.py --exp-dir "$R" --folds 0 \
        > "logs/recipe_${cfg}_oof.log" 2>&1 || echo "  WARN oof $cfg"
  echo "$(date +%H:%M) DONE $cfg (endpoint ep$ep)"
done
echo "QUEUE_COMPLETE gpu$GPU: $*"
