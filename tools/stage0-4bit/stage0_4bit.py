#!/usr/bin/env python3
"""stage0_4bit.py: re-quantize the stage-0 INT8 residue of the hybrid GLM-5.3 checkpoint to NVFP4 (W4A16).

  --model  the hybrid checkpoint of cebeuq's recipe (NVFP4 routed experts plus an INT8-g128 residue), as his requant
           tool writes it. Read only.
  --hub    the BF16 parents of that residue: model.safetensors.index.json and the shards of
           dealignai/GLM-5.3-UNCENSORED-NVFP4 that hold layers 0-29 (README.md lists them). Read only.
  --out    the new checkpoint directory.

For every shard of --model that holds a target module, --out gets a new safetensors file with the same name: every
other tensor byte for byte, and each target module as NVFP4 (weight_packed U8 [N, K/2], weight_scale F8_E4M3
[N, K/16], weight_global_scale F32 [], one global scale for a module pair that vLLM fuses), quantized from its BF16
parent with the compressed-tensors 0.17.0 functions the engine's loader uses. Every other file of --model is linked
into --out (--link). Then a new model.safetensors.index.json and a new config.json: group_1 narrowed to layers 30-77
plus the modules of 0-29 kept at INT8, group_2 = nvfp4-pack-quantized for the targets.

Targets (layers 0-29): the attention projections q_a, kv_a, q_b, kv_b and o, the shared experts of layers 3-29 and
the dense MLPs of layers 0-2, minus --keep-int8. The default keep list gives the served form: q_b_proj, o_proj and the
shared experts at NVFP4; q_a and kv_a (fused_qkv_a_proj), kv_b_proj and the dense MLPs of 0-2 stay INT8.

One shard at a time; the quantizer works in blocks of rows. --dry-run only plans. --work holds the amax cache and the
report (not the output directory: the engine and view-build.py take every file of a checkpoint directory).
"""
import argparse
import copy
import errno
import json
import os
import re
import shutil
import sys
import time

import torch
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams, generate_gparam
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
ATTN = ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.q_b_proj", "self_attn.kv_b_proj", "self_attn.o_proj"]
SHARED = ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj", "mlp.shared_experts.down_proj"]
DENSE = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
FUSED = {"self_attn.q_a_proj": "self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_proj_with_mqa": "self_attn.q_a_proj",
         "mlp.shared_experts.gate_proj": "mlp.shared_experts.up_proj", "mlp.shared_experts.up_proj": "mlp.shared_experts.gate_proj",
         "mlp.gate_proj": "mlp.up_proj", "mlp.up_proj": "mlp.gate_proj"}
ARGS_NVFP4 = QuantizationArgs(num_bits=4, type="float", strategy="tensor_group", group_size=16, symmetric=True, dynamic=False, scale_dtype=torch.float8_e4m3fn)
ROWS = 2048
SERVED_KEEP = "self_attn.q_a_proj,self_attn.kv_a_proj_with_mqa,self_attn.kv_b_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj"


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


_open = {}


def opn(p):
    if p not in _open: _open[p] = safe_open(p, framework="pt", device="cpu")
    return _open[p]


def hdr(p):
    import struct
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; return json.loads(f.read(n))


# ---------------------------------------------------------------- quantizer (row blocks; identical results to a whole-tensor call)
def q_nvfp4(W32, gscale):
    packs, scales = [], []
    for r in range(0, W32.shape[0], ROWS):
        Wb = W32[r:r + ROWS]; Wg = Wb.reshape(Wb.shape[0], -1, 16)
        scale, _ = calculate_qparams(Wg.amin(2), Wg.amax(2), ARGS_NVFP4, global_scale=gscale)
        q = quantize(x=Wb, scale=scale, zero_point=None, args=ARGS_NVFP4, global_scale=gscale)
        packs.append(pack_fp4_to_uint8(q)); scales.append(scale.to(torch.float8_e4m3fn))
    return torch.cat(packs, 0).contiguous(), torch.cat(scales, 0).contiguous()


def place(src, dst, how):
    """put one untouched file of --model into --out"""
    if how == "hardlink":
        os.link(src, dst)
    elif how == "symlink":
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="Re-quantize the stage-0 INT8 residue of the hybrid GLM-5.3 checkpoint to NVFP4.")
    ap.add_argument("--model", required=True, help="the hybrid checkpoint (read only)")
    ap.add_argument("--hub", required=True, help="BF16 parents: model.safetensors.index.json + the shards that hold layers 0-29")
    ap.add_argument("--out", required=True, help="new checkpoint directory")
    ap.add_argument("--work", default=".", help="directory for the amax cache and the report (default: current directory)")
    ap.add_argument("--layers", default="0-29"); ap.add_argument("--keep-int8", default=SERVED_KEEP)
    ap.add_argument("--link", choices=["hardlink", "symlink", "copy"], default="hardlink",
                    help="how untouched files get into --out: hardlink (same filesystem, default), symlink (the container "
                         "must then also see --model at the same path), copy")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--only-shards", default="")
    ap.add_argument("--amax-json", default="", help="amax cache (default: <work>/stage0_amax.json)")
    ap.add_argument("--skip-shards", action="store_true", help="reuse the rewritten shards already in --out; only links, index, config")
    a = ap.parse_args()
    MODEL, HUB = a.model, a.hub
    amax_json = a.amax_json or os.path.join(a.work, "stage0_amax.json"); report_json = os.path.join(a.work, "stage0_report.json")
    lo, hi = [int(x) for x in a.layers.split("-")]; layers = list(range(lo, hi + 1)); keep = set(x for x in a.keep_int8.split(",") if x)
    idx_path = os.path.join(MODEL, "model.safetensors.index.json"); idx_full = json.load(open(idx_path)); idx = idx_full["weight_map"]
    hidx = json.load(open(os.path.join(HUB, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    # targets
    targets = {}  # module name -> dict(layer, mod, shard)
    for L in layers:
        for m in ATTN + (DENSE if L < 3 else SHARED):
            if m in keep: continue
            n = "model.layers.%d.%s" % (L, m)
            assert n + ".weight_packed" in idx, n
            shard = idx[n + ".weight_packed"]
            assert idx[n + ".weight_scale"] == shard and idx[n + ".weight_shape"] == shard, ("module straddles shards", n)
            assert n + ".weight" in hidx, ("no hub bf16 for", n)
            targets[n] = {"layer": L, "mod": m, "shard": shard}
    shards = sorted(set(t["shard"] for t in targets.values()))
    old_names = set(); new_names = {}
    for n, t in targets.items():
        for s in ("weight_packed", "weight_scale", "weight_shape"): old_names.add(n + "." + s)
        for s in ("weight_packed", "weight_scale", "weight_global_scale"): new_names[n + "." + s] = t["shard"]
    log("format nvfp4", "layers", a.layers, "keep int8", sorted(keep), "target modules", len(targets), "affected shards", len(shards))
    if a.dry_run:
        for s in shards: print(s, sum(1 for t in targets.values() if t["shard"] == s), "modules")
        print("hub shards read:", " ".join(sorted(set(hidx[n + ".weight"] for n in targets))))
        print("new tensor names:", len(new_names), "old names removed:", len(old_names)); return
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.work, exist_ok=True)
    # pre-pass: amax per target tensor from the BF16 parents (for the fused global scales)
    amax = {}
    if os.path.exists(amax_json):
        amax = json.load(open(amax_json)); log("amax cache", len(amax))
    if any(n not in amax for n in targets):
        t0 = time.time()
        for n in targets:
            if n in amax: continue
            W = opn(os.path.join(HUB, hidx[n + ".weight"])).get_tensor(n + ".weight")
            amax[n] = {"amax": W.abs().max().item(), "rms": W.float().pow(2).mean().sqrt().item(), "shape": list(W.shape)}; del W
        json.dump(amax, open(amax_json, "w"), indent=1); log("amax pre-pass done", round(time.time() - t0), "s")

    def gscale_for(n, t):
        partner = FUSED.get(t["mod"])
        am = amax[n]["amax"]
        if partner:
            pn = "model.layers.%d.%s" % (t["layer"], partner)
            assert pn in targets, ("fused partner not a target", n, pn)
            am = max(am, amax[pn]["amax"])
        g = generate_gparam(torch.tensor([-am]), torch.tensor([am]))   # 448*6/amax, f32 [1]
        return g
    # shards
    sel = set(a.only_shards.split(",")) if a.only_shards else None
    report = {"format": "nvfp4", "layers": a.layers, "keep_int8": sorted(keep), "shards": {}, "started": time.strftime("%FT%T")}
    for s in shards:
        if sel and s not in sel: continue
        if a.skip_shards:
            assert os.path.exists(os.path.join(a.out, s)) and not os.path.samefile(os.path.join(a.out, s), os.path.join(MODEL, s)), ("missing rewritten shard", s)
            continue
        t0 = time.time(); src = os.path.join(MODEL, s); dst = os.path.join(a.out, s)
        h = hdr(src); names = [k for k in h if k != "__metadata__"]
        f = opn(src); new = {}
        kept = 0
        for k in names:
            if k in old_names: continue
            new[k] = f.get_tensor(k); kept += 1
        mods_here = [n for n, t in targets.items() if t["shard"] == s]
        for n in sorted(mods_here):
            t = targets[n]
            W = opn(os.path.join(HUB, hidx[n + ".weight"])).get_tensor(n + ".weight").float()
            g = gscale_for(n, t); p, sc = q_nvfp4(W, g)
            new[n + ".weight_packed"] = p; new[n + ".weight_scale"] = sc; new[n + ".weight_global_scale"] = g.reshape(()).clone()
            del W
        if os.path.exists(dst): os.unlink(dst)
        save_file(new, dst, metadata={"format": "pt"})
        rep = {"old_bytes": os.path.getsize(src), "new_bytes": os.path.getsize(dst), "tensors_kept": kept, "modules_requantized": len(mods_here), "sec": round(time.time() - t0, 1)}
        report["shards"][s] = rep; log(s, rep)
        del new; _open.pop(src, None)
        json.dump(report, open(report_json, "w"), indent=1)
    if sel: log("partial run (--only-shards), no links, index or config"); return
    # every other file of --model: linked or copied
    linked = 0
    for e in sorted(os.listdir(MODEL)):
        if e in ("config.json", "model.safetensors.index.json") or e in shards: continue
        d = os.path.join(a.out, e)
        if os.path.lexists(d) and os.path.exists(d) and os.path.samefile(d, os.path.join(MODEL, e)): linked += 1; continue
        if os.path.lexists(d): os.unlink(d)
        try:
            place(os.path.join(MODEL, e), d, a.link); linked += 1
        except OSError as exc:
            if exc.errno != errno.EXDEV: raise
            sys.exit("--out is on another filesystem than --model: run with --link copy, or --link symlink")
    log("files linked or copied", linked, "with", a.link)
    # index (total_size over all shards: the rewritten ones from --out, the untouched ones from --model)
    wm = {k: v for k, v in idx.items() if k not in old_names}; wm.update(new_names)
    total = sum(os.path.getsize(os.path.join(a.out, e)) if e in shards else os.path.getsize(os.path.join(MODEL, e)) for e in os.listdir(MODEL) if e.endswith(".safetensors"))
    new_idx = {"metadata": dict(idx_full.get("metadata", {})), "weight_map": dict(sorted(wm.items()))}; new_idx["metadata"]["total_size"] = total
    json.dump(new_idx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    # config: group_1 narrowed to layers 30-77 (+ kept modules of 0-29), group_2 = the 4-bit targets of layers 0-29
    qc = cfg["quantization_config"]; g1 = qc["config_groups"]["group_1"]
    rng = "(?:3[0-9]|[4-6][0-9]|7[0-7])"; rng0 = "(?:[0-9]|1[0-9]|2[0-9])"; rng3 = "(?:[3-9]|1[0-9]|2[0-9])"
    assert lo == 0 and hi == 29, "regexes below assume layers 0-29"
    attn_all = "q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|fused_qkv_a_proj"
    g1_targets = ["re:.*\\.layers\\.%s\\.self_attn\\.(%s)$" % (rng, attn_all),
                  "re:.*\\.layers\\.%s\\.mlp\\.shared_experts\\.(gate_proj|up_proj|down_proj|gate_up_proj)$" % rng]
    kept_attn = sorted(m.split(".")[-1] for m in keep if m.startswith("self_attn."))
    if "q_a_proj" in kept_attn and "kv_a_proj_with_mqa" in kept_attn: kept_attn.append("fused_qkv_a_proj")
    if kept_attn: g1_targets.append("re:.*\\.layers\\.%s\\.self_attn\\.(%s)$" % (rng0, "|".join(kept_attn)))
    kept_shared = sorted(m.split(".")[-1] for m in keep if m.startswith("mlp.shared_experts."))
    if "gate_proj" in kept_shared and "up_proj" in kept_shared: kept_shared.append("gate_up_proj")
    if kept_shared: g1_targets.append("re:.*\\.layers\\.%s\\.mlp\\.shared_experts\\.(%s)$" % (rng3, "|".join(kept_shared)))
    kept_dense = sorted(m.split(".")[-1] for m in keep if re.match(r"mlp\.(gate|up|down)_proj$", m))
    if "gate_proj" in kept_dense and "up_proj" in kept_dense: kept_dense.append("gate_up_proj")
    if kept_dense: g1_targets.append("re:.*\\.layers\\.[0-2]\\.mlp\\.(%s)$" % "|".join(kept_dense))
    attn4 = [x for x in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"] if "self_attn." + x not in keep]
    if "q_a_proj" in attn4 and "kv_a_proj_with_mqa" in attn4: attn4.append("fused_qkv_a_proj")
    shared4 = [x for x in ["gate_proj", "up_proj", "down_proj"] if "mlp.shared_experts." + x not in keep]
    if "gate_proj" in shared4 and "up_proj" in shared4: shared4.append("gate_up_proj")
    dense4 = [x for x in ["gate_proj", "up_proj", "down_proj"] if "mlp." + x not in keep]
    if "gate_proj" in dense4 and "up_proj" in dense4: dense4.append("gate_up_proj")
    g2_targets = []
    if attn4: g2_targets.append("re:.*\\.layers\\.%s\\.self_attn\\.(%s)$" % (rng0, "|".join(attn4)))
    if shared4: g2_targets.append("re:.*\\.layers\\.%s\\.mlp\\.shared_experts\\.(%s)$" % (rng3, "|".join(shared4)))
    if dense4: g2_targets.append("re:.*\\.layers\\.[0-2]\\.mlp\\.(%s)$" % "|".join(dense4))
    g2 = {"format": "nvfp4-pack-quantized", "input_activations": None, "output_activations": None, "targets": g2_targets, "weights": copy.deepcopy(qc["config_groups"]["group_0"]["weights"])}
    new_cfg = copy.deepcopy(cfg); new_cfg["quantization_config"]["config_groups"]["group_1"]["targets"] = g1_targets
    new_cfg["quantization_config"]["config_groups"]["group_2"] = g2
    json.dump(new_cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    report["linked"] = linked; report["index_tensors"] = len(wm); report["total_size"] = total; report["group_1_targets"] = g1_targets; report["group_2"] = g2; report["finished"] = time.strftime("%FT%T")
    json.dump(report, open(report_json, "w"), indent=1)
    log("done: shards", len(shards), "linked", linked, "index tensors", len(wm), "total_size GB", round(total / 1e9, 2))


if __name__ == "__main__":
    main()
