#!/usr/bin/env bash
# Re-run only the export step on every instance whose training already finished.
#
# The original export keyed the output filename on the checkpoint's DIRECTORY, so
# all/epoch05..epoch30 all landed in one all.pth and only the last survived. The
# raw .ckpt files are still on the instances, so the snapshots are recoverable --
# we just have to write them out with distinct names.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
vastai show instances --raw 2>/dev/null | .venv/bin/python -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
for i in d:
    if i.get('actual_status')=='running' and i.get('ssh_host'):
        print(i['id'], i.get('label','').replace('rare26-','') or '?', i['ssh_host'], i['ssh_port'])" > /tmp/_reinst.txt
while read id label host port; do
  SSH="ssh -n -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p $port root@$host"
  # Only worth doing once the all-data run has produced its snapshots.
  n=$(timeout 40 $SSH 'ls /workspace/RARE/results/*/all/epoch*.ckpt 2>/dev/null | wc -l' 2>/dev/null)
  [ -z "${n:-}" ] && { echo "$(printf '%-34s' "$label") unreachable"; continue; }
  [ "$n" -eq 0 ] && { echo "$(printf '%-34s' "$label") no snapshots yet"; continue; }
  timeout 40 scp -o StrictHostKeyChecking=no -o BatchMode=yes -P "$port" \
      deploy/remote_run.sh root@"$host":/workspace/RARE/deploy/remote_run.sh >/dev/null 2>&1
  # remote_run.sh sets PY/WORK/CFG above the export block; slicing the block out
  # leaves them unset, which silently ran '- <<PY' as a command instead of python.
  out=$(timeout 600 $SSH "cd /workspace/RARE && export CFG=$label WORK=/workspace/RARE \
      PY=\$(command -v python) && sed -n '/=== compact export/,\$p' deploy/remote_run.sh | bash" 2>&1 | tail -4)
  echo "$(printf '%-34s' "$label") snapshots=$n :: $(echo "$out" | tr '\n' ' ')"
done < /tmp/_reinst.txt
