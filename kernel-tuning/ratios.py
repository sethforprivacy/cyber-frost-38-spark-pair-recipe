#!/usr/bin/env python3
"""Per-expert gate/up global scales (weight_scale_2) of a local NVFP4 checkpoint, stdlib only.
  python3 ratios.py CHECKPOINT_DIR 0,12,24,36,47 > v1_ratios.json      (all 512 experts of each listed layer)
"""
import json, struct, sys, os
root = sys.argv[1]; layers = [int(x) for x in sys.argv[2].split(",")]
wm = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
hdr = {}
def tensor_bytes(k):
    f = wm[k]
    if f not in hdr:
        with open(f"{root}/{f}", "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr[f] = (8 + n, json.loads(fh.read(n)))
    off, h = hdr[f]; a, b = h[k]["data_offsets"]
    with open(f"{root}/{f}", "rb") as fh:
        fh.seek(off + a); return fh.read(b - a)
out = {}
for L in layers:
    r = []
    for e in range(512):
        p = f"model.language_model.layers.{L}.mlp.experts.{e}"
        g = struct.unpack("<f", tensor_bytes(f"{p}.gate_proj.weight_scale_2"))[0]
        u = struct.unpack("<f", tensor_bytes(f"{p}.up_proj.weight_scale_2"))[0]
        r.append([e, g, u])
    out[L] = r
json.dump(out, sys.stdout)
