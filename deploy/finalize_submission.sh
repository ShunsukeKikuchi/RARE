#!/usr/bin/env bash
# Everything between "training is done" and "the tarballs are ready to upload".
#
# Written as one script because the order matters and a missed step is silent:
# a stale fold0-only OOF mis-ranks a member, and the container drops members from
# the end of that ranking when the 600 s job limit bites.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python

echo "=== 1/5 refresh OOF where more folds exist than the stored file covers ==="
bash deploy/refresh_oof.sh

echo "=== 2/5 collect every member that has both an all-data checkpoint and an OOF ==="
mapfile -t MEMBERS < <($PY - <<'PYE'
import glob, os
from pathlib import Path
out=[]
for d in sorted(glob.glob("results/*/")) + sorted(glob.glob("results/remote/*/")):
    n=Path(d).name
    if n.startswith(("_","remote")) or "discard" in n or "held" in n: continue
    has_all = len(glob.glob(d+"all/epoch*.ckpt")) + len(glob.glob(d+"all*.pth"))
    if has_all and os.path.exists(d+"oof.csv") and n not in out:
        out.append(n)
print("\n".join(out))
PYE
)
echo "  ${#MEMBERS[@]} members: ${MEMBERS[*]}"
[ "${#MEMBERS[@]}" -gt 0 ] || { echo "no members -- aborting"; exit 1; }

echo "=== 3/5 build the model bundle (ranks members, verifies every file loads) ==="
$PY submit/v001_seg_aux/build_submission.py --configs "${MEMBERS[@]}" --members all || exit 1

echo "=== 4/5 container regression test with no network ==="
( cd submit/v001_seg_aux && bash test.sh ) || { echo "CONTAINER TEST FAILED"; exit 1; }

echo "=== 5/5 export image + model tarballs ==="
( cd submit/v001_seg_aux && bash export.sh ) || exit 1

echo "=== member table for the method PDF ==="
$PY submit/paper/member_table.py | tee submit/paper/members.md
$PY submit/paper/build_pdf.py
echo "FINALIZE_COMPLETE"
