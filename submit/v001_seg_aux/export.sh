#!/usr/bin/env bash
# Produce the tar.gz to upload to Grand Challenge.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TAG="${DOCKER_IMAGE_TAG:-rare26-seg-aux}"
source ./build.sh
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# Grand Challenge takes TWO uploads: the algorithm container, and a separate
# "Algorithm Models" tarball that it extracts to /opt/ml/model at run time. The
# weights are NOT baked into the image -- keeping them out is what lets us swap
# the ensemble without rebuilding and re-uploading a multi-GB image.
OUT="${TAG}_${STAMP}.tar.gz"
echo "=+= saving image $TAG -> $OUT"
docker save "$TAG" | gzip -c > "$OUT"

MODELS="${TAG}_models_${STAMP}.tar.gz"
echo "=+= packing model tarball -> $MODELS"
[ -f resources/manifest.json ] || { echo "resources/manifest.json missing -- run build_submission.py first"; exit 1; }
# Contents sit at the archive root: GC extracts straight into /opt/ml/model.
tar -czf "$MODELS" -C resources .

echo "=+= manifest summary"
python3 - <<'PY'
import json, collections
m = json.load(open("resources/manifest.json"))["members"]
print(f"  {len(m)} members, {len({x['config'] for x in m})} configs")
for fam, n in sorted(collections.Counter(x["family"] for x in m).items()):
    print(f"    {fam:16s} {n}")
PY
ls -lh "$OUT" "$MODELS"
