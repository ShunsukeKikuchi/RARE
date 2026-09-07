#!/usr/bin/env bash
# Build the algorithm model tarball that Grand Challenge extracts to /opt/ml/model.
#
# Keeping weights OUT of the image means the container is built and validated once,
# then the ensemble can be updated up to the deadline by re-uploading only this
# tarball (~hundreds of MB instead of a ~10 GB image).
#
#   ./export_model_tarball.sh            # packs resources/ as-is
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SRC=${1:-resources}
OUT="algorithmmodel_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
[ -d "$SRC" ] || { echo "no such dir: $SRC" >&2; exit 1; }
n=$(ls "$SRC"/fold*.pth 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || { echo "no fold*.pth in $SRC" >&2; exit 1; }
# The trailing '.' matters: GC extracts the archive contents directly into /opt/ml/model.
tar -czvf "$OUT" -C "$SRC" . >/dev/null
echo "packed $n member(s) -> $OUT ($(du -h "$OUT" | cut -f1))"
tar -tzf "$OUT" | head -5
