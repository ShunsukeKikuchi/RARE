#!/usr/bin/env bash
# Pull per-fold metrics.csv (the val curves) off every live instance.
#
# Needed because the endpoint for all-data training is architecture-dependent -- a
# LoRA ViT plateaus in a couple of epochs where the EfficientNet needs ~15 -- so
# each config's endpoint is chosen from its own curve. The instances launched
# before remote_run.sh started exporting metrics.csv still have the files locally.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=results/remote_metrics; mkdir -p "$OUT"
vastai show instances --raw 2>/dev/null | .venv/bin/python -c "
import json,sys
for i in json.load(sys.stdin): print(i['id'], i.get('label','').replace('rare26-',''))" > /tmp/_inst.txt
while read id label; do
  url=$(timeout 30 vastai ssh-url "$id" 2>/dev/null) || continue
  host=$(echo "$url" | sed -E 's|ssh://root@([^:]+):.*|\1|'); port=$(echo "$url" | sed -E 's|.*:([0-9]+)$|\1|')
  mkdir -p "$OUT/$label"
  timeout 90 scp -q -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=12 -P "$port" \
    "root@$host:/workspace/RARE/results/*/fold*/metrics.csv" "$OUT/$label/" 2>/dev/null
  n=$(ls "$OUT/$label"/*.csv 2>/dev/null | wc -l)
  # scp flattens same-named files; pull them one fold at a time instead
  if [ "$n" -le 1 ]; then
    for f in 0 1 2 3 4; do
      timeout 60 scp -q -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=12 -P "$port" \
        "root@$host:/workspace/RARE/results/*/fold$f/metrics.csv" "$OUT/$label/fold$f.csv" 2>/dev/null
    done
  fi
  echo "$(printf '%-38s' "$label") $(ls "$OUT/$label"/*.csv 2>/dev/null | wc -l) curve(s)"
done < /tmp/_inst.txt
