#!/usr/bin/env python3
"""view-build.py <dst-dir> <src-dir> <first-layer> <last-layer-inclusive> [embed] [head] [mtp]

Builds <dst-dir>: symlinks to the shards that hold one node's tensors plus all the small files, and a filtered
model.safetensors.index.json. The fork's loader globs *.safetensors, so a node reads only its own shards, and it
refuses an index that names absent files, hence the filtered copy.
  embed  the node holds the first stage (token embedding)
  head   the node holds the last stage (final norm and lm_head)
  mtp    the last stage also runs the MTP drafter (layer 78 and a copy of the embedding)
The links are absolute: mount <src-dir> into the container at the same path so they resolve.
"""
import json
import os
import re
import sys

dst, SRC, a, b = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"), int(sys.argv[3]), int(sys.argv[4])
flags = set(sys.argv[5:])
idx = json.load(open(SRC + "/model.safetensors.index.json"))["weight_map"]
layers = set(range(a, b + 1)) | ({78} if "mtp" in flags else set())
need = set()
for k, f in idx.items():
    m = re.match(r"model\.layers\.(\d+)\.", k)
    if m:
        if int(m.group(1)) in layers:
            need.add(f)
    elif k == "model.embed_tokens.weight" and ("embed" in flags or "mtp" in flags):
        need.add(f)
    elif k in ("lm_head.weight", "model.norm.weight") and ("head" in flags or "mtp" in flags):
        need.add(f)
    elif not k.startswith("model.layers.") and k not in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        need.add(f)  # unknown top-level tensor: always include
os.makedirs(dst, exist_ok=True)
for e in os.listdir(dst):
    os.unlink(os.path.join(dst, e))
n = 0
gb = 0.0
full = json.load(open(SRC + "/model.safetensors.index.json"))
full["weight_map"] = {k: f for k, f in idx.items() if f in need}
json.dump(full, open(dst + "/model.safetensors.index.json", "w"))
for e in sorted(os.listdir(SRC)):
    if e.endswith(".safetensors") and e not in need:
        continue
    if e == "model.safetensors.index.json":
        continue
    os.symlink(os.path.join(SRC, e), os.path.join(dst, e))
    if e.endswith(".safetensors"):
        n += 1
        gb += os.path.getsize(os.path.join(SRC, e)) / 1e9
print("VIEW %s layers %d-%d flags=%s shards=%d GB=%.1f" % (dst, a, b, ",".join(sorted(flags)) or "-", n, gb), flush=True)
