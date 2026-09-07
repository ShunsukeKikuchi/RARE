#!/usr/bin/env bash
# One snapshot of every vast.ai instance: fold progress, OOF, all-data checkpoints.
#
# Prints one line per instance, stable and sorted, so a caller can diff successive
# runs and surface only what changed. Deliberately does NOT destroy anything --
# an auto-reaper keyed on a stale "finished" test previously destroyed three
# instances whose snapshots had not been re-exported yet.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
vastai show instances --raw 2>/dev/null | .venv/bin/python -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
for i in d:
    if i.get('actual_status')=='running' and i.get('ssh_host'):
        print(i['id'], i.get('label','').replace('rare26-','') or '?', i['ssh_host'], i['ssh_port'], i.get('actual_status'))" > /tmp/_poll.txt
[ -s /tmp/_poll.txt ] || { echo "VAST: no instances (or API unreachable)"; exit 0; }
while read id label h p status; do
  if [ "$status" != "running" ]; then
    printf "%-32s STATUS=%s\n" "$label" "$status"; continue
  fi
  r=$(timeout 45 ssh -n -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=12 \
        -p "$p" root@"$h" '
    d=$(ls -d /workspace/RARE/results/*/ 2>/dev/null | grep -v remote | head -1)
    [ -z "$d" ] && { echo "not started"; exit 0; }
    folds=$(ls -d $d/fold*/training_log.json 2>/dev/null | wc -l)
    alld=$(ls $d/all/*.ckpt 2>/dev/null | wc -l)
    oof=$([ -f "$d/oof.csv" ] && echo Y || echo n)
    err=$(grep -lE "Traceback|CUDA out of memory" $d/*/*.log 2>/dev/null | head -1)
    echo "folds=$folds/5 all_ckpt=$alld oof=$oof${err:+ ERROR:$(basename $(dirname $err))}"' 2>/dev/null)
  printf "%-32s %s\n" "$label" "${r:-ssh timeout}"
done < /tmp/_poll.txt | sort
