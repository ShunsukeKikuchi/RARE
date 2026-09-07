#!/usr/bin/env bash
# Local regression test under the real Grand Challenge runtime constraints:
# no network, the same /input and /output mount points, weights baked in.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TAG="${DOCKER_IMAGE_TAG:-rare26-seg-aux}"
IN="$PWD/test/input/interface_0"
OUT="$PWD/test/output/interface_0"
VOLUME="${TAG}-tmp"

source ./build.sh

mkdir -p "$OUT"
chmod -R -f o+rX "$IN" "$PWD/resources" || true
chmod -f o+rwX "$OUT" || true

docker volume create "$VOLUME" > /dev/null
cleanup() {
  docker run --rm --platform=linux/amd64 --quiet ${GPU_FLAG/--gpus all/} --volume "$OUT":/output \
    --entrypoint /bin/sh "$TAG" -c "chmod -R -f o+rwX /output/* || true"
  docker volume rm "$VOLUME" > /dev/null
}
trap cleanup EXIT

# Grand Challenge runs with a GPU. This host may not: its docker default runtime is
# "nvidia" but nvidia-container-runtime is not installed, so we fall back to runc and
# run on CPU. That still verifies the whole I/O contract -- it is just slower.
GPU_FLAG="--gpus all"
if ! docker run --rm --gpus all --entrypoint /bin/true "$TAG" 2>/dev/null; then
  echo "=+= WARNING: no GPU passthrough on this host -- running on CPU via runc"
  GPU_FLAG="--runtime=runc"
fi

echo "=+= running with --network none ${GPU_FLAG:-(cpu)}"
docker run --rm --platform=linux/amd64 \
  --network none \
  $GPU_FLAG \
  --volume "$IN":/input:ro \
  --volume "$OUT":/output \
  --volume "$VOLUME":/tmp \
  --volume "$PWD/resources":/opt/ml/model:ro \
  "$TAG"

echo "=+= validating output"
# The repo venv has SimpleITK; the system python usually does not.
VALIDATE_PY="$PWD/../../.venv/bin/python"
[ -x "$VALIDATE_PY" ] || VALIDATE_PY=python3
"$VALIDATE_PY" - "$IN" "$OUT" <<'PYEOF'
import json, sys
from glob import glob
from pathlib import Path

in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_file = out_dir / "stacked-neoplastic-lesion-likelihoods.json"
assert out_file.is_file(), f"missing {out_file}"
probs = json.loads(out_file.read_text())

stack = sorted(glob(str(in_dir / "images/stacked-barretts-esophagus-endoscopy/*")))[0]
import SimpleITK as sitk
r = sitk.ImageFileReader(); r.SetFileName(stack); r.ReadImageInformation()
n = r.GetSize()[2] if r.GetDimension() >= 3 else 1

assert isinstance(probs, list), f"expected a JSON list, got {type(probs)}"
assert len(probs) == n, f"expected {n} likelihoods, got {len(probs)}"
assert all(isinstance(p, (int, float)) for p in probs), "non-numeric entry"
assert all(0.0 <= p <= 1.0 for p in probs), f"out of range: min {min(probs)} max {max(probs)}"
print(f"OK: {len(probs)} likelihoods, min {min(probs):.6f}, max {max(probs):.6f}")
print("first 8:", [round(p, 6) for p in probs[:8]])
PYEOF
