#!/usr/bin/env bash
# Runs ONE config end-to-end on a vast.ai box and pushes compact results back.
#
#   HF_TOKEN=hf_xxx CFG=surgenet_vitl bash remote_run.sh
#
# Pushes back only what we need (a few hundred MB, not raw checkpoints):
#   results/<CFG>/oof.csv, evc preds, training_log.json, compact weights.
set -euo pipefail
WORK=${WORK:-/workspace/RARE}
REPO=${REPO:-negichi/rare26-work}
CFG=${CFG:?set CFG, e.g. surgenet_vitl}
: "${HF_TOKEN:?set HF_TOKEN}"
export WORK REPO CFG HF_TOKEN

cd "$WORK"
export PYTHONUNBUFFERED=1
PY=$(command -v python)
SRC=workspace/expA00_seg_aux_baseline/src
CONF=workspace/expA00_seg_aux_baseline/config/$CFG.yaml
NAME=$($PY -c "import yaml;print(yaml.safe_load(open('$CONF'))['experiment']['name'])")

echo "=== training $CFG (experiment=$NAME) ==="
for fold in 0 1 2 3 4; do
  [ -f "results/$NAME/fold$fold/training_log.json" ] && { echo "fold $fold done"; continue; }
  $PY $SRC/train.py --config "$CONF" --fold "$fold"
done

# The shipped model trains on EVERYTHING, EVC included: external data is explicitly
# permitted (with disclosure) and EVC is our best segmentation supervision -- 100
# Barrett's images with five-expert consensus masks. The 5-fold models above stay
# EVC-free so their OOF remains a clean calibration set; the affine (t, b) fitted
# there is transferred to this model (same architecture and recipe, 80% vs 100% data).
echo "=== all-data model (shipped) -- EVC included ==="
# Snapshot densely: the optimal endpoint is architecture-dependent, not universal.
# A LoRA ViT reaches ~0.96 val AUROC in 2 epochs while the EfficientNet needs ~15,
# so a single fixed endpoint would under- or over-train half the ensemble. The
# per-config choice is made later from that config's own 5-fold val curves.
$PY $SRC/train_all.py --config "$CONF" --epochs 30 --snapshot-epochs 5 10 15 20 25 \
    --override data.use_evc=true || { echo "TRAIN_ALL FAILED"; exit 1; }

echo "=== OOF + EVC ==="
$PY $SRC/predict_oof.py --exp-dir "results/$NAME" || true
$PY $SRC/evaluate.py --oof "results/$NAME/oof.csv" || true
$PY $SRC/eval_evc.py --exp-dir "results/$NAME" || true

echo "=== compact export + push ==="
$PY - <<'PY'
import os, torch, json, tarfile
from pathlib import Path
work=Path(os.environ['WORK']); name=os.environ['NAME'] if 'NAME' in os.environ else None
import yaml
cfg=yaml.safe_load(open(work/f"workspace/expA00_seg_aux_baseline/config/{os.environ['CFG']}.yaml"))
name=cfg['experiment']['name']
out=work/'export'/name; out.mkdir(parents=True, exist_ok=True)
EMA, RAW = 'ema.module.', 'model.'
for ck in sorted((work/'results'/name).rglob('*.ckpt')):
    if ck.name.startswith('last'): continue
    sd=torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
    pre = EMA if any(k.startswith(EMA) for k in sd) else RAW
    st={k[len(pre):]:v for k,v in sd.items() if k.startswith(pre)}
    # LoRA members: the frozen base is identical everywhere, ship only adapters+head.
    if any('.A' in k and '.base.' not in k for k in st):
        st={k:v for k,v in st.items() if ('.A' in k or '.B' in k or 'head' in k) and '.base.' not in k}
    # The dir name alone is NOT unique: all/ holds epoch05..epoch30 snapshots, so
    # keying on it collapsed every endpoint into one all.pth (last write won) and
    # threw away exactly the snapshots --snapshot-epochs exists to produce.
    tag = ck.parent.name if ck.stem in ('best_model', 'final') else f'{ck.parent.name}_{ck.stem}'
    torch.save(st, out/f'{tag}.pth')
    print(tag, round(sum(v.numel() for v in st.values())*4/1e6,1),'MB')
for extra in ['oof.csv','oof_metrics.json']:
    p=work/'results'/name/extra
    if p.exists(): (out/extra).write_bytes(p.read_bytes())
for p in (work/'results').glob(f'evc_preds_{name}*.csv'): (out/p.name).write_bytes(p.read_bytes())
for p in (work/'results'/name).rglob('training_log.json'):
    (out/f'training_log_{p.parent.name}.json').write_bytes(p.read_bytes())
# Per-fold val curves: needed to pick this config's own endpoint.
for p in (work/'results'/name).rglob('metrics.csv'):
    (out/f'metrics_{p.parent.name}.csv').write_bytes(p.read_bytes())
# Results are collected over scp by deploy/collect_results.sh, not pushed: the
# private HF repo hit its storage quota mid-run and every push after that failed
# with the work already complete on disk. Leaving export/ in place and pulling it
# has no quota and no single point of failure.
print('export ready at', out)
PY
echo "=== DONE $CFG ==="
# Stop billing as soon as the results are safely in the HF repo. Without this the
# box idles at ~$0.35/h after finishing, and with a dozen instances that adds up.
if [ -n "${VAST_CONTAINERLABEL:-}" ] && command -v vastai >/dev/null; then
  vastai destroy instance "${VAST_CONTAINERLABEL#C.}" || true
fi
# Fallback: signal completion so the poller can reap it.
touch /workspace/DONE
