#!/usr/bin/env bash
# Re-run the all-data model at the endpoint that config's own 5-fold curves picked.
#
# The instances launched before the snapshot list was widened ran with
# "--snapshot-epochs 20 25", so all/ holds only epoch20/25/30 -- and most configs
# peak near epoch 10. Rather than ship a 20-epoch model for a 10-epoch optimum,
# just train to the right endpoint: these are 30-60s/epoch, so ~10 epochs is minutes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
declare -A BEST=( [resnext101_swsl]=10 [maxvit_seg]=11 [surgenet_vitl_strongaug]=10
                  [dinov3_vitl_strongaug]=10 [surgenet_vitl]=8 [dinov3_vitl]=14
                  [convnext_dinov3_seg]=28 [resnext101_swsl_strongaug]=30 )
vastai show instances --raw 2>/dev/null | .venv/bin/python -c "
import json,sys
for i in json.load(sys.stdin):
    if i.get('actual_status')=='running': print(i['id'], i.get('label','').replace('rare26-',''))" > /tmp/_rb.txt
while read id label; do
  ep=${BEST[$label]:-}
  [ -z "$ep" ] && { echo "$(printf '%-32s' "$label") no measured best epoch yet -- skip"; continue; }
  url=$(timeout 30 vastai ssh-url "$id" 2>/dev/null) || continue
  h=$(echo "$url"|sed -E 's|ssh://root@([^:]+):.*|\1|'); p=$(echo "$url"|sed -E 's|.*:([0-9]+)$|\1|')
  SSH="ssh -n -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p $p root@$h"
  # already have it?
  if timeout 40 $SSH "ls /workspace/RARE/results/*/all/epoch$(printf '%02d' "$ep").ckpt" >/dev/null 2>&1; then
    echo "$(printf '%-32s' "$label") epoch$ep already present"; continue; fi
  echo "$(printf '%-32s' "$label") training all-data to epoch $ep ..."
  timeout 5400 $SSH "cd /workspace/RARE && PYTHONUNBUFFERED=1 python \
      workspace/expA00_seg_aux_baseline/src/train_all.py \
      --config workspace/expA00_seg_aux_baseline/config/$label.yaml \
      --epochs $ep --override data.use_evc=true" 2>&1 | tail -2
done < /tmp/_rb.txt
