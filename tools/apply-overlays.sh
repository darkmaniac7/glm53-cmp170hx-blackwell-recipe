#!/bin/bash
# apply-overlays.sh <recipe-checkout> [ppfix] [g4] [sq] [g5] [jor] [bs]: copy this repository's overlay files into a
# checkout of cebeuq/glm53-cmp170hx-recipe. Without names all six are copied (g5 needs g4's files, so copy both). Only adds
# files; nothing of the recipe is overwritten. Afterwards extend COMPOSE_FILE in engine/prod.vars (overlays/README.md).
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
R=${1:?usage: apply-overlays.sh <recipe-checkout> [ppfix] [g4] [sq] [g5] [jor] [bs]}; shift
[ -f "$R/engine/compose.yaml" ] && [ -d "$R/patches/vllm" ] || { echo "$R is not a checkout of the recipe"; exit 2; }
[ $# -gt 0 ] || set -- ppfix g4 sq g5 jor bs
for o in "$@"; do
  [ -d "$HERE/overlays/$o" ] || { echo "unknown overlay $o"; exit 2; }
  for d in engine patches bench pipespec docs; do
    [ -d "$HERE/overlays/$o/$d" ] || continue
    (cd "$HERE/overlays/$o" && find "$d" -type f) | while read -r f; do
      [ -e "$R/$f" ] && { echo "refusing to overwrite $R/$f"; exit 3; }
      mkdir -p "$R/$(dirname "$f")"; cp "$HERE/overlays/$o/$f" "$R/$f"; echo "  + $f"
    done
  done
  echo "applied $o"
done
