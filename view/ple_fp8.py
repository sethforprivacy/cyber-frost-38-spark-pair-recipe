#!/usr/bin/env python3
"""Derive a CYBER-FROST checkpoint view whose PLE / n-gram table is FP8 (e4m3fn, one global scale).

vLLM 0.30's own path: text_config.ple_embedding_dtype == "float8_e4m3fn" selects
Qwen4ExpPLEFp8EmbeddingMethod (ngram_embedding.shard_<k>.weight in FP8 + ngram_embedding.weight_scale,
one fp32 scalar, dequant = fp8.to(bf16) * scale.to(bf16)). Same layout family as nvidia/Qwen3.8-Flash-Next-NVFP4
and Blackfrost-AI/CYBER-FROST-3.8-FP8 (FP8 table, one scale). The table itself is stock Qwen (the PLE-only
shards hash-match base), so this is the same table NVIDIA ships in FP8.

Only the 33 files holding PLE shards are rewritten (same names; every other tensor in them copied
byte-for-byte); the other 98 shards and all tokenizer/template files are symlinked to the source.
Deterministic: both ranks produce identical files (compare manifest.json sha256s).

  python3 ple_fp8.py SRC DST          (inside the vLLM image: torch + safetensors)
"""
import hashlib, json, os, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST = sys.argv[1], sys.argv[2]
PFX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
torch.set_num_threads(int(os.environ.get("THREADS", "16")))
os.makedirs(DST, exist_ok=True)
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wm = idx["weight_map"]
ple = sorted((k for k in wm if k.startswith(PFX + "shard_") and k.endswith(".weight")),
             key=lambda k: int(k[len(PFX) + 6:-7]))
files = sorted({wm[k] for k in ple})
print(f"{len(ple)} PLE shards in {len(files)} files", flush=True)

# pass 1: global amax
t0 = time.time(); amax = 0.0; n = 0
for f in files:
    with safe_open(f"{SRC}/{f}", "pt") as h:
        for k in h.keys():
            if k in ple:
                w = h.get_tensor(k)
                amax = max(amax, float(w.abs().max().float())); n += w.numel()
print(f"pass1 amax {amax:.6g} over {n} elements, {time.time()-t0:.0f}s", flush=True)
# scale: smallest bf16 >= amax/448, so q <= 448 and the kernel's bf16 cast of the scale is exact
s = torch.tensor([amax / 448.0], dtype=torch.float32)
sb = s.to(torch.bfloat16).float()
if sb.item() < s.item():
    sb = torch.nextafter(s.to(torch.bfloat16), torch.tensor([float("inf")], dtype=torch.bfloat16)).float()
scale = sb.item()
print(f"scale {scale:.8g} (amax/448 = {amax/448:.8g})", flush=True)

# pass 2: rewrite
err2 = 0.0; ref2 = 0.0; sub = 0; total_size = 0
scale_file = wm[ple[0]]
for f in files:
    t1 = time.time(); out = {}; meta = None
    with safe_open(f"{SRC}/{f}", "pt") as h:
        meta = h.metadata()
        for k in h.keys():
            w = h.get_tensor(k)
            if k in ple:
                x = w.float()
                q = (x / scale).to(torch.float8_e4m3fn)
                d = q.to(torch.bfloat16) * torch.tensor(scale, dtype=torch.bfloat16)
                e = d.float() - x
                err2 += float((e * e).sum()); ref2 += float((x * x).sum())
                sub += int((q.float().abs() < 2 ** -6).sum() - (x == 0).sum())
                out[k] = q.contiguous()
            else:
                out[k] = w.contiguous()
    if f == scale_file:
        out[PFX + "weight_scale"] = torch.tensor([scale], dtype=torch.float32)
    save_file(out, f"{DST}/{f}.tmp", metadata=meta)
    os.replace(f"{DST}/{f}.tmp", f"{DST}/{f}")
    print(f"wrote {f} ({time.time()-t1:.0f}s)", flush=True)
print(f"relative RMS error {((err2/ref2)**.5):.5f}; subnormal-or-flushed nonzeros {sub}", flush=True)

for name in os.listdir(SRC):
    p = f"{DST}/{name}"
    if name in files or name in ("config.json", "model.safetensors.index.json") or os.path.lexists(p):
        continue
    os.symlink(os.path.join(SRC, name), p)

cfg = json.load(open(f"{SRC}/config.json"))
cfg["text_config"]["ple_embedding_dtype"] = "float8_e4m3fn"
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
wm2 = dict(wm); wm2[PFX + "weight_scale"] = scale_file
sizes = 0
for f in sorted(set(wm2.values())):
    with open(f"{DST}/{f}", "rb") as fh:
        hn = int.from_bytes(fh.read(8), "little"); hdr = json.loads(fh.read(hn))
    sizes += sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in hdr.items() if k != "__metadata__")
json.dump({"metadata": {"total_size": sizes}, "weight_map": wm2}, open(f"{DST}/model.safetensors.index.json", "w"), indent=2)

man = {}
for f in files + ["config.json", "model.safetensors.index.json"]:
    hs = hashlib.sha256()
    with open(f"{DST}/{f}", "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            hs.update(b)
    man[f] = hs.hexdigest()
json.dump({"scale": scale, "amax": amax, "rel_rms_err": (err2 / ref2) ** .5, "total_size": sizes, "sha256": man},
          open(f"{DST}/ple-fp8-manifest.json", "w"), indent=1)
print(f"DONE total_size {sizes} ({sizes/2**30:.2f} GiB) in {time.time()-t0:.0f}s", flush=True)
