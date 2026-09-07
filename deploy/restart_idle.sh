#!/usr/bin/env bash
# Restart training on instances that are reachable but idle.
#
# Two boxes sat at 0% GPU for ~10 hours with folds=0/5: the training process died
# without the wrapper noticing, so the instance kept billing and produced nothing.
# Polling GPU utilisation is what surfaces this -- "actual_status=running" does not.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${HF_TOKEN:=none}"
vastai show instances --raw 2>/dev/null | .venv/bin/python -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
for i in d:
    if i.get('actual_status')=='running' and i.get('ssh_host'):
        print(i['id'], i.get('label','').replace('rare26-','') or '?', i['ssh_host'], i['ssh_port'])" > /tmp/_idle.txt
while read id label host port; do
  SSH="ssh -n -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=12 -p $port root@$host"
  info=$(timeout 40 $SSH 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; \
     pgrep -f "train(_all)?\.py" | wc -l' 2>/dev/null) || { echo "$(printf '%-32s' "$label") unreachable"; continue; }
  mem=$(echo "$info" | head -1 | tr -d ' '); procs=$(echo "$info" | tail -1 | tr -d ' ')
  if [ "${mem:-0}" -gt 500 ]; then
    echo "$(printf '%-32s' "$label") busy (${mem} MiB, $procs procs)"; continue
  fi
  echo "$(printf '%-32s' "$label") IDLE (${mem} MiB) -- restarting remote_run.sh"
  timeout 40 scp -o StrictHostKeyChecking=no -o BatchMode=yes -P "$port" \
      deploy/remote_run.sh root@"$host":/workspace/RARE/deploy/remote_run.sh >/dev/null 2>&1
  # detached: the ssh session must be able to close without taking training with it
  timeout 60 $SSH "cd /workspace/RARE && HF_TOKEN=$HF_TOKEN CFG=$label \
      setsid nohup bash deploy/remote_run.sh > /workspace/rerun_${label}.log 2>&1 < /dev/null & echo started" 2>&1 | tail -1
done < /tmp/_idle.txt
