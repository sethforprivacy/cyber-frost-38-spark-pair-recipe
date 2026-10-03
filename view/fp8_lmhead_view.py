#!/usr/bin/env python3
"""Derive a view from the FP8-dense view whose output head (lm_head) is also FP8 per output channel.

The target's BF16 head is 636 MB per rank and is read once per decode step (2.7 ms of a ~43 ms step on GB10).
Same format as fp8_dense_view.py (ModelOpt FP8_PER_CHANNEL_PER_TOKEN: e4m3 weight, fp32 scale per vocab row,
dynamic per-token activation). The draft head (dh4, NVFP4) is unchanged. Only the shard holding lm_head.weight is
rewritten; "lm_head" leaves the ignore list and gets a quantized_layers entry.

  python3 fp8_lmhead_view.py SRC DST          (inside the vLLM image: torch + safetensors)
"""
import hashlib, json, os, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
ALGO = "FP8_PER_CHANNEL_PER_TOKEN"
KEY = "lm_head.weight"
os.makedirs(DST, exist_ok=True)
for n in sorted(os.listdir(SRC)):
    s, d = f"{SRC}/{n}", f"{DST}/{n}"
    if n in ("config.json", "model.safetensors.index.json", "hf_quant_config.json") or os.path.lexists(d):
        continue
    if os.path.isdir(s):
        os.symlink(os.path.realpath(s), d)
    else:
        os.link(os.path.realpath(s), d)
idx = json.load(open(f"{SRC}/model.safetensors.index.json")); wm = idx["weight_map"]
f = wm[KEY]; t0 = time.time(); out = {}
with safe_open(f"{SRC}/{f}", "pt") as h:
    meta = h.metadata()
    for k in h.keys():
        w = h.get_tensor(k)
        if k == KEY:   # row chunks: a full fp32 copy of the 248K x 2560 head does not fit a 6 GB container
            q = torch.empty(w.shape, dtype=torch.float8_e4m3fn); scale = torch.empty(w.shape[0], dtype=torch.float32)
            e2 = 0.0; r2 = 0.0
            for r0 in range(0, w.shape[0], 8192):
                x = w[r0:r0 + 8192].float()
                s = (x.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
                qc = (x / s[:, None]).to(torch.float8_e4m3fn)
                e2 += float((qc.float() * s[:, None] - x).pow(2).sum()); r2 += float(x.pow(2).sum())
                q[r0:r0 + 8192] = qc; scale[r0:r0 + 8192] = s
            err = (e2 / r2) ** 0.5
            out[k] = q; out["lm_head.weight_scale"] = scale; wm["lm_head.weight_scale"] = f
        else:
            out[k] = w
save_file(out, f"{DST}/{f}.tmp", metadata=meta)
if os.path.exists(f"{DST}/{f}"):
    os.unlink(f"{DST}/{f}")
os.rename(f"{DST}/{f}.tmp", f"{DST}/{f}")
idx["weight_map"] = wm
json.dump(idx, open(f"{DST}/model.safetensors.index.json", "w"), indent=2)
for name in ("config.json", "hf_quant_config.json"):
    c = json.load(open(f"{SRC}/{name}"))
    for qc in [c[k] for k in ("quantization_config", "quantization") if isinstance(c.get(k), dict)]:
        for key in ("ignore", "exclude_modules"):
            if isinstance(qc.get(key), list):
                qc[key] = [e for e in qc[key] if e != "lm_head"]
        if isinstance(qc.get("quantized_layers"), dict):
            qc["quantized_layers"]["lm_head"] = {"quant_algo": ALGO}
    json.dump(c, open(f"{DST}/{name}", "w"), indent=2)
h = hashlib.sha256()
with open(f"{DST}/{f}", "rb") as fh:
    for chunk in iter(lambda: fh.read(64 << 20), b""):
        h.update(chunk)
man = {"source": SRC, "algo": ALGO, "tensor": KEY, "file": f, "sha256": h.hexdigest(), "rel_rms_err": err,
       "seconds": round(time.time() - t0, 1)}
json.dump(man, open(f"{DST}/fp8-lmhead-manifest.json", "w"), indent=1)
print("DONE", json.dumps(man), flush=True)
