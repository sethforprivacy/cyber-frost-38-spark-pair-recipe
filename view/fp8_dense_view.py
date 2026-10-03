#!/usr/bin/env python3
"""Derive a CYBER-FROST checkpoint view whose dense BF16 projections are FP8 E4M3 with one scale per output channel.

The NVFP4 export keeps every non-expert projection in BF16 (ModelOpt ignore list), and on GB10 those BF16 GEMMs were
half of a decode step (2026-10-01 profile). vLLM 0.30's ModelOpt mixed precision already serves
"FP8_PER_CHANNEL_PER_TOKEN" per layer (W8A8: static per-channel FP8 weight, dynamic per-token FP8 activation, CUTLASS
scaled_mm on SM12x), which ran 1.9x faster than BF16 cuBLAS at decode sizes and 1.4x at prefill sizes in
fp8_linear_bench.py. This view switches these layers to that format:

  linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}     36 GDN layers
  self_attn.{q_proj,k_proj,v_proj,o_proj}                              12 QSA layers
  mlp.shared_expert.{gate_proj,up_proj,down_proj}                      48 layers
  {attn,mlp}_hyper_connection.input_mix_weight_up                      48 x 2 (myllmbox passes quant_config here)
Unchanged (BF16): router gates, shared-expert sigmoid gates, the QSA indexer, HC down/injection (built with
quant_config=None), PLE projections, embeddings, lm_head, vision, the MTP layer ("mtp*" stays ignored).

Per row: scale = amax / 448 (fp32), q = round(w / scale) in e4m3fn. Each "<m>.weight" becomes FP8 and a
"<m>.weight_scale" [N] fp32 is added beside it. Only the shards holding these tensors are rewritten (same names,
other tensors copied byte-for-byte); every other file is hardlinked from SRC. config.json: the layers get
quantized_layers entries and the ignore wildcards that covered them are dropped. Deterministic on both ranks.

  python3 fp8_dense_view.py SRC DST          (inside the vLLM image: torch + safetensors)
"""
import hashlib, json, os, re, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
ALGO = "FP8_PER_CHANNEL_PER_TOKEN"
PAT = re.compile(r"^model\.language_model\.layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)"
                 r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.shared_expert\.(gate_proj|up_proj|down_proj)"
                 r"|(attn|mlp)_hyper_connection\.input_mix_weight_up)\.weight$")
DROP_IGNORE = re.compile(r"^model\.language_model\.layers\.\d+\.(linear_attn|self_attn|mlp\.shared_expert"
                         r"|attn_hyper_connection|mlp_hyper_connection)\*$")
torch.set_num_threads(int(os.environ.get("THREADS", "8")))
os.makedirs(DST, exist_ok=True)

for n in sorted(os.listdir(SRC)):   # hardlink everything first; rewritten shards replace their links below
    s, d = f"{SRC}/{n}", f"{DST}/{n}"
    if n in ("config.json", "model.safetensors.index.json", "hf_quant_config.json") or os.path.lexists(d):
        continue
    if os.path.isdir(s):
        os.symlink(os.path.realpath(s), d)   # asset dirs: link to the host path, as fix_links.sh does
    else:
        os.link(os.path.realpath(s), d)

idx = json.load(open(f"{SRC}/model.safetensors.index.json")); wm = idx["weight_map"]
targets = sorted(k for k in wm if PAT.match(k))
files = sorted({wm[k] for k in targets})
print(f"{len(targets)} tensors in {len(files)} files", flush=True)
errs = {}; t0 = time.time(); added = 0
for f in files:
    t1 = time.time(); out = {}
    with safe_open(f"{SRC}/{f}", "pt") as h:
        meta = h.metadata()
        for k in h.keys():
            w = h.get_tensor(k)
            if k in targets:
                assert w.dtype == torch.bfloat16 and w.dim() == 2, (k, w.dtype, w.shape)
                x = w.float()
                scale = (x.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
                q = (x / scale[:, None]).to(torch.float8_e4m3fn)
                deq = q.float() * scale[:, None]
                errs[k] = float((deq - x).norm() / x.norm().clamp_min(1e-30))
                out[k] = q
                out[k[:-len("weight")] + "weight_scale"] = scale.contiguous()
                wm[k[:-len("weight")] + "weight_scale"] = f; added += 1
            else:
                out[k] = w
    tmp = f"{DST}/{f}.tmp"
    save_file(out, tmp, metadata=meta)
    if os.path.exists(f"{DST}/{f}"):
        os.unlink(f"{DST}/{f}")         # drop the hardlink, never write through it
    os.rename(tmp, f"{DST}/{f}")
    print(f"{f}: {sum(1 for k in out if k in targets)} fp8 tensors, {time.time() - t1:.1f}s", flush=True)

total = 0
for f in sorted(set(wm.values())):
    total += os.path.getsize(f"{DST}/{f}")
idx["weight_map"] = wm; idx.setdefault("metadata", {})["total_size"] = total
json.dump(idx, open(f"{DST}/model.safetensors.index.json", "w"), indent=2)

for name in ("config.json", "hf_quant_config.json"):
    p = f"{SRC}/{name}"
    if not os.path.exists(p):
        continue
    c = json.load(open(p))
    qcs = [c[k] for k in ("quantization_config", "quantization") if isinstance(c.get(k), dict)]
    if "text_config" in c and isinstance(c["text_config"].get("quantization_config"), dict):
        qcs.append(c["text_config"]["quantization_config"])
    for qc in qcs:
        for key in ("ignore", "exclude_modules"):
            if isinstance(qc.get(key), list):
                before = len(qc[key]); qc[key] = [e for e in qc[key] if not DROP_IGNORE.match(e)]
                print(f"{name}: {key} {before} -> {len(qc[key])}", flush=True)
        if isinstance(qc.get("quantized_layers"), dict):
            for k in targets:
                qc["quantized_layers"][k[:-len(".weight")]] = {"quant_algo": ALGO}
            print(f"{name}: quantized_layers -> {len(qc['quantized_layers'])}", flush=True)
    json.dump(c, open(f"{DST}/{name}", "w"), indent=2)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


man = {"source": SRC, "algo": ALGO, "tensors": len(targets), "scales_added": added,
       "rel_rms_err": {"mean": sum(errs.values()) / len(errs), "max": max(errs.values()),
                       "max_tensor": max(errs, key=errs.get)},
       "files": {f: sha256(f"{DST}/{f}") for f in files},
       "seconds": round(time.time() - t0, 1)}
json.dump(man, open(f"{DST}/fp8-dense-manifest.json", "w"), indent=1)
print("DONE", json.dumps({k: v for k, v in man.items() if k != "files"}), flush=True)
