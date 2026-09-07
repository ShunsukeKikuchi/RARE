#!/usr/bin/env bash
# Entry point for expA00. Training and inference go through here, never by
# invoking the python files directly.
#
#   ./run.sh train   <variant> <fold> [gpu]   train one fold
#   ./run.sh sweep   <variant> [gpu...]       train all 5 folds, one per GPU
#   ./run.sh oof     <variant>                out-of-fold predictions + metrics
#   ./run.sh loco    <variant> <gpu>          leave-one-center-out domain probe
#   ./run.sh errors  <variant>                FP/FN contact sheets
#   ./run.sh evc     <variant>...             external eval on held-out EVC
#
# Variants (the ablation axis; see SESSION_NOTES.md):
#   a_rare_only   RARE25 only, no seg loss          -- pure baseline
#   a2_ext_cls    RARE25 + external, no seg loss    -- isolates "more images"
#   b_seg_aux     RARE25 + external + seg loss      -- the proposed model
#   a2_edd_cls    RARE25 + EDD2020, no seg loss     -- "more images" arm, EVC-free
#   c_no_evc      RARE25 + EDD2020 + seg loss       -- the EVC-free proposed model
#
# EVC is deliberately NOT trained on: it is held out as an external test set
# (100 balanced images from a third site). See eval_evc.py.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO_ROOT=$(cd ../.. && pwd)
# Suffix for the results dir, so a variation (e.g. EXP_SUFFIX=_ema) does not collide
# with an earlier run of the same variant.
SUFFIX="${EXP_SUFFIX:-}"
PY="$REPO_ROOT/.venv/bin/python"
# Overridable so architecture variants can supply their own config.
CONFIG="${CONFIG:-config/base.yaml}"
SRC=src

variant_overrides() {
  case "$1" in
    a_rare_only) echo "data.use_evc=false data.use_edd2020=false loss.seg_weight=0.0" ;;
    a2_ext_cls)  echo "data.use_evc=true  data.use_edd2020=true  loss.seg_weight=0.0" ;;
    a2_edd_cls)  echo "data.use_evc=false data.use_edd2020=true  loss.seg_weight=0.0" ;;
    b_seg_aux)   echo "data.use_evc=true  data.use_edd2020=true  loss.seg_weight=0.5" ;;
    c_no_evc)    echo "data.use_evc=false data.use_edd2020=true  loss.seg_weight=0.5" ;;
    *) echo "unknown variant: $1" >&2; exit 1 ;;
  esac
}

cmd=${1:?usage: run.sh <train|sweep|oof|loco|errors> ...}; shift

case "$cmd" in
  train)
    variant=${1:?variant}; fold=${2:?fold}; gpu=${3:-0}
    CUDA_VISIBLE_DEVICES=$gpu "$PY" $SRC/train.py --config $CONFIG --fold "$fold" \
      --override experiment.name="expA00_${variant}${SUFFIX}" $(variant_overrides "$variant")
    ;;

  sweep_cfg)
    # Sweep an alternative config file (architecture variants). $CONFIG selects it and
    # the experiment name comes from that config, so no variant_overrides are applied.
    name=${1:?name}; shift
    if [ $# -eq 0 ]; then gpus=(2 3); else gpus=("$@"); fi
    per_gpu="${JOBS_PER_GPU:-2}"
    for g in "${gpus[@]}"; do
      free=$(nvidia-smi --id="$g" --query-gpu=memory.free --format=csv,noheader,nounits)
      echo "GPU $g: ${free}MB free"
    done
    declare -a pids=() tags=(); i=0
    for fold in 0 1 2 3 4; do
      if [ -f "$REPO_ROOT/results/${name}/fold${fold}/training_log.json" ]; then
        echo ">>> fold $fold already done, skipping"; continue; fi
      gpu=${gpus[$((i % ${#gpus[@]}))]}
      echo ">>> $name fold=$fold gpu=$gpu"
      CUDA_VISIBLE_DEVICES=$gpu "$PY" $SRC/train.py --config "$CONFIG" --fold "$fold" &
      pids+=($!); tags+=("fold$fold"); i=$((i+1))
      if [ ${#pids[@]} -ge $((per_gpu * ${#gpus[@]})) ]; then
        for k in "${!pids[@]}"; do wait "${pids[$k]}" || echo "FAILED ${tags[$k]}" >&2; done
        pids=(); tags=()
      fi
      sleep 5
    done
    if [ ${#pids[@]} -gt 0 ]; then for k in "${!pids[@]}"; do wait "${pids[$k]}" || echo "FAILED ${tags[$k]}" >&2; done; fi
    echo "$name: $(ls -d "$REPO_ROOT/results/${name}"/fold*/training_log.json 2>/dev/null | wc -l)/5 folds done"
    ;;

  sweep)
    variant=${1:?variant}; shift
    # EXTRA_OVERRIDE lets a sweep vary one hyper-parameter (e.g. trainer.max_epochs)
    # without editing the config.
    extra="${EXTRA_OVERRIDE:-}"
    # GPUs 0 and 1 are usually taken by other jobs on this box.
    if [ $# -eq 0 ]; then gpus=(2 3); else gpus=("$@"); fi
    per_gpu="${JOBS_PER_GPU:-2}"
    need_mb="${NEED_FREE_MB:-10000}"

    # Pre-flight: a fold that lands on a full GPU dies with CUDA OOM seconds after
    # start, which previously took the whole sweep down with it. Check first.
    for g in "${gpus[@]}"; do
      free=$(nvidia-smi --id="$g" --query-gpu=memory.free --format=csv,noheader,nounits)
      want=$((need_mb * per_gpu))
      if [ "$free" -lt "$want" ]; then
        echo "GPU $g has ${free}MB free, need ~${want}MB for ${per_gpu} jobs" >&2
        exit 1
      fi
      echo "GPU $g: ${free}MB free -> ${per_gpu} job(s)"
    done

    declare -a pids=() tags=()
    i=0
    for fold in 0 1 2 3 4; do
      # Skip folds already finished, so a re-run resumes instead of redoing work.
      if [ -f "$REPO_ROOT/results/expA00_${variant}${SUFFIX}/fold${fold}/training_log.json" ]; then
        echo ">>> fold $fold already done, skipping"; continue
      fi
      gpu=${gpus[$((i % ${#gpus[@]}))]}
      echo ">>> variant=$variant fold=$fold gpu=$gpu"
      CUDA_VISIBLE_DEVICES=$gpu "$PY" $SRC/train.py --config $CONFIG --fold "$fold" \
        --override experiment.name="expA00_${variant}${SUFFIX}" $(variant_overrides "$variant") $extra &
      pids+=($!); tags+=("fold$fold(gpu$gpu)")
      i=$((i + 1))
      if [ ${#pids[@]} -ge $((per_gpu * ${#gpus[@]})) ]; then
        # Wait per-pid so one failure does not abort the sweep via `set -e`.
        failed=""
        for k in "${!pids[@]}"; do wait "${pids[$k]}" || failed="$failed ${tags[$k]}"; done
        [ -n "$failed" ] && echo "FAILED:$failed" >&2
        pids=(); tags=()
      fi
      sleep 5
    done
    rc=0
    if [ ${#pids[@]} -gt 0 ]; then
      failed=""
      for k in "${!pids[@]}"; do wait "${pids[$k]}" || failed="$failed ${tags[$k]}"; done
      if [ -n "$failed" ]; then echo "FAILED:$failed" >&2; rc=1; fi
    fi
    n_done=$(ls -d "$REPO_ROOT/results/expA00_${variant}${SUFFIX}"/fold*/training_log.json 2>/dev/null | wc -l)
    echo "sweep $variant: ${n_done}/5 folds have training_log.json"
    [ "$n_done" -eq 5 ] || rc=1
    exit $rc
    ;;

  oof)
    variant=${1:?variant}
    exp_dir="$REPO_ROOT/results/expA00_${variant}${SUFFIX}"
    "$PY" $SRC/predict_oof.py --exp-dir "$exp_dir"
    "$PY" $SRC/evaluate.py --oof "$exp_dir/oof.csv"
    ;;

  loco)
    variant=${1:?variant}; gpu=${2:-0}
    CUDA_VISIBLE_DEVICES=$gpu "$PY" $SRC/loco.py --config $CONFIG \
      --override experiment.name="expA00_${variant}${SUFFIX}_loco" $(variant_overrides "$variant")
    ;;

  evc)
    # External held-out evaluation on EVC (never trained on). Accepts several variants.
    shift_args=""
    dirs=""
    for v in "$@"; do dirs="$dirs $REPO_ROOT/results/expA00_${v}${SUFFIX}"; done
    "$PY" $SRC/eval_evc.py --exp-dir $dirs
    ;;

  errors)
    variant=${1:?variant}
    "$PY" $SRC/error_analysis.py --oof "$REPO_ROOT/results/expA00_${variant}${SUFFIX}/oof.csv"
    ;;

  *) echo "unknown command: $cmd" >&2; exit 1 ;;
esac
