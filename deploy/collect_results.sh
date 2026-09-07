#!/usr/bin/env bash
# Pull finished results straight off the instances.
#
# The HF relay hit the private-storage quota, so every instance that finished
# after that point failed at the push step with its work intact on disk. Weights
# are small (LoRA members ~19 MB), so scp is a fine substitute for the relay.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=results/remote; mkdir -p "$OUT"
vastai show instances --raw 2>/dev/null | .venv/bin/python -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
for i in d:
    if i.get('actual_status')=='running' and i.get('ssh_host'):
        print(i['id'], i.get('label','').replace('rare26-','') or '?', i['ssh_host'], i['ssh_port'], i.get('actual_status'))" > /tmp/_inst.txt
while read id label host port status; do
  # export/ is written by remote_run.sh before the push attempt
  names=$(timeout 40 ssh -n -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=12 \
      -p "$port" root@"$host" 'ls /workspace/RARE/export 2>/dev/null' 2>/dev/null)
  for n in $names; do
    mkdir -p "$OUT/$n"
    # rsync, not scp: it resumes, skips files already pulled (instances are polled
    # repeatedly), and does not need the fiddly trailing-dot source syntax.
    # Bounded hard: one unreachable host held the whole sweep for 15 min, which with
    # 8 instances outran the 10-minute poll so collections stacked up and fought
    # each other over the same links.
    timeout 240 rsync -a --partial --timeout=45 \
      -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 -p $port" \
      "root@$host:/workspace/RARE/export/$n/" "$OUT/$n/" 2>/dev/null
    echo "$(printf '%-34s' "$label") -> $n ($(ls "$OUT/$n" 2>/dev/null | wc -l) files, $(du -sh "$OUT/$n" 2>/dev/null | cut -f1))"
  done
  [ -z "$names" ] && echo "$(printf '%-34s' "$label") (no export yet)"
done < /tmp/_inst.txt
