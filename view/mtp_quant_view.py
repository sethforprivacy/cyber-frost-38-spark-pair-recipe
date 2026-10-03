#!/usr/bin/env python3
"""Derive the 2026-10-03 CyberFrost views from the V2 production view (hardlinks + rewritten shards).

  python3 derive_view_1003.py SRC DST [--hc8] [--mtpq]       (inside the vLLM image: torch + safetensors)

--hc8   The target's hyper-connection down/inject projections (input_mix_weight_down [320, 10240] and
        block_inject_weight [4, 10240] of every attn/mlp hyper-connection, merged at load into one replicated
        GEMM) become FP8 E4M3 with one scale per output channel, the format fp8_dense_view.py gives the other dense
        projections (FP8_PER_CHANNEL_PER_TOKEN). Needs the hc8 hyperconnection.py overlay (the image builds this
        projection with quant_config=None).
--mtpq  The MTP layer, which the export kept in BF16 (ignore "mtp*"). Draft-only: the target verifies every token,
        so outputs are unchanged and only acceptance and speed move.
          routed experts: the fused BF16 gate_up_proj [512, 1280, 2560] / down_proj [512, 2560, 640] become
            per-expert ModelOpt NVFP4 tensors served W4A16 (Marlin): group 16, E4M3 block scales, one FP32 global
            scale per expert shared by gate and up (as V2 does), no activation scales
          dense projections: self_attn q/k/v/o, shared_expert gate/up/down, the attn/mlp hyper-connection
            input_mix_weight_up, fc_embedding and fc_hidden become FP8 per channel; with --hc8 also the MTP layer's
            hyper-connection down/inject
          the router gate, shared_expert_gate, QSA indexer and the final mixer stay BF16
        The "mtp*" ignore wildcard is replaced by explicit quantized_layers entries (checkpoint names, layer 0;
        vLLM remaps them to the draft model's layer 48).
Only the shards holding these tensors are rewritten (other tensors byte-for-byte); every other file is hardlinked.
Deterministic on both ranks.
"""
import hashlib, json, os, re, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
HC8, MTPQ = "--hc8" in sys.argv, "--mtpq" in sys.argv
assert HC8 or MTPQ
FP8 = "FP8_PER_CHANNEL_PER_TOKEN"
pats = []
if HC8:
    pats.append(r"model\.language_model\.layers\.\d+\.(attn|mlp)_hyper_connection\.(input_mix_weight_down|block_inject_weight)")
if MTPQ:
    pats += [r"mtp\.layers\.0\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
             r"mtp\.layers\.0\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)",
             r"mtp\.layers\.0\.(attn|mlp)_hyper_connection\.input_mix_weight_up",
             r"mtp\.(fc_embedding|fc_hidden)"]
    if HC8:
        pats.append(r"mtp\.layers\.0\.(attn|mlp)_hyper_connection\.(input_mix_weight_down|block_inject_weight)")
FP8_PAT = re.compile(r"^(" + "|".join(pats) + r")\.weight$")
MTP_EXPERTS = {"mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"} if MTPQ else set()
torch.set_num_threads(int(os.environ.get("THREADS", "16")))
os.makedirs(DST, exist_ok=True)

for n in sorted(os.listdir(SRC)):   # hardlink everything first; rewritten shards replace their links below
    s, d = f"{SRC}/{n}", f"{DST}/{n}"
    if n in ("config.json", "model.safetensors.index.json", "hf_quant_config.json") or os.path.lexists(d):
        continue
    if os.path.islink(s) or os.path.isdir(s):
        os.symlink(os.path.realpath(s), d)
    else:
        os.link(os.path.realpath(s), d)

idx = json.load(open(f"{SRC}/model.safetensors.index.json")); wm = idx["weight_map"]
fp8_targets = sorted(k for k in wm if FP8_PAT.match(k))
files = sorted({wm[k] for k in fp8_targets} | {wm[k] for k in MTP_EXPERTS})
print(f"{len(fp8_targets)} fp8 tensors, {len(MTP_EXPERTS)} fused MTP expert tensors, in {len(files)} files", flush=True)

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
BOUNDS = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def nvfp4(w, gs):
    """w [E, R, K] fp32, gs [E] fp32 global (dequant) scales -> packed uint8 [E, R, K/2], E4M3 [E, R, K/16], rel err."""
    E, R, K = w.shape
    wb = w.view(E, R, K // 16, 16)
    bs = (wb.abs().amax(-1) / 6.0 / gs.view(E, 1, 1)).clamp(max=448.0).to(torch.float8_e4m3fn)
    eff = bs.float() * gs.view(E, 1, 1)
    x = wb / eff.unsqueeze(-1).clamp_min(1e-30)
    code = torch.bucketize(x.abs().clamp(max=6.0), BOUNDS).to(torch.uint8)
    deq = E2M1[code.long()] * torch.sign(x) * eff.unsqueeze(-1)
    nib = code | ((x < 0) & (code > 0)).to(torch.uint8) << 3
    nib = nib.view(E, R, K)
    packed = nib[..., 0::2] | (nib[..., 1::2] << 4)
    err = ((deq.view(E, R, K) - w).flatten(1).norm(dim=1) / w.flatten(1).norm(dim=1).clamp_min(1e-30))
    return packed.contiguous(), bs.contiguous(), err


errs = {}; nv_errs = []; t0 = time.time()
for f in files:
    t1 = time.time(); out = {}
    with safe_open(f"{SRC}/{f}", "pt") as h:
        meta = h.metadata()
        for k in h.keys():
            w = h.get_tensor(k)
            if k in fp8_targets:
                assert w.dtype == torch.bfloat16 and w.dim() == 2, (k, w.dtype, w.shape)
                x = w.float()
                scale = (x.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
                q = (x / scale[:, None]).to(torch.float8_e4m3fn)
                errs[k] = float((q.float() * scale[:, None] - x).norm() / x.norm().clamp_min(1e-30))
                out[k] = q
                sk = k[:-len("weight")] + "weight_scale"
                out[sk] = scale.contiguous(); wm[sk] = f
            elif k in MTP_EXPERTS:
                assert w.dtype == torch.bfloat16 and w.dim() == 3, (k, w.dtype, w.shape)
                base = "mtp.layers.0.mlp.experts"
                del wm[k]
                for e0 in range(0, w.shape[0], 32):
                    x = w[e0:e0 + 32].float()
                    if k.endswith("gate_up_proj"):          # [E, 2I, H]: gate rows first, then up
                        half = x.shape[1] // 2
                        gs = x.abs().flatten(1).amax(1) / (6.0 * 448.0)       # one scale for the gate/up pair
                        parts = {"gate_proj": x[:, :half], "up_proj": x[:, half:]}
                    else:                                    # down_proj [E, H, I]
                        gs = x.abs().flatten(1).amax(1) / (6.0 * 448.0)
                        parts = {"down_proj": x}
                    for name, p in parts.items():
                        packed, bs, err = nvfp4(p.contiguous(), gs)
                        nv_errs += err.tolist()
                        for j in range(p.shape[0]):
                            pre = f"{base}.{e0 + j}.{name}"
                            out[f"{pre}.weight"] = packed[j].clone()
                            out[f"{pre}.weight_scale"] = bs[j].clone()
                            out[f"{pre}.weight_scale_2"] = gs[j].clone().reshape(())
                            for s in ("weight", "weight_scale", "weight_scale_2"):
                                wm[f"{pre}.{s}"] = f
            else:
                out[k] = w
    tmp = f"{DST}/{f}.tmp"
    save_file(out, tmp, metadata=meta)
    if os.path.exists(f"{DST}/{f}"):
        os.unlink(f"{DST}/{f}")         # drop the hardlink, never write through it
    os.rename(tmp, f"{DST}/{f}")
    print(f"{f}: rewritten, {time.time() - t1:.1f}s", flush=True)

if nv_errs:
    m = sum(nv_errs) / len(nv_errs)
    print(f"MTP experts NVFP4: {len(nv_errs)} matrices, rel L2 error mean {m:.4f} max {max(nv_errs):.4f}", flush=True)
    assert m < 0.11 and max(nv_errs) < 0.13, "NVFP4 error out of the expected ~0.095 range"
total = sum(os.path.getsize(f"{DST}/{f}") for f in sorted(set(wm.values())))
idx["weight_map"] = wm; idx.setdefault("metadata", {})["total_size"] = total
json.dump(idx, open(f"{DST}/model.safetensors.index.json", "w"), indent=2)

new_layers = {k[:-len(".weight")]: {"quant_algo": FP8} for k in fp8_targets}
if MTPQ:
    new_layers["mtp.layers.0.mlp.experts"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
for name in ("config.json", "hf_quant_config.json"):
    p = f"{SRC}/{name}"
    if not os.path.exists(p):
        continue
    c = json.load(open(p))
    qcs = [c[k] for k in ("quantization_config", "quantization") if isinstance(c.get(k), dict)]
    if "text_config" in c and isinstance(c["text_config"].get("quantization_config"), dict):
        qcs.append(c["text_config"]["quantization_config"])
    for qc in qcs:
        if MTPQ:
            for key in ("ignore", "exclude_modules"):
                if isinstance(qc.get(key), list) and "mtp*" in qc[key]:
                    qc[key] = [e for e in qc[key] if e != "mtp*"]
                    print(f"{name}: dropped 'mtp*' from {key}", flush=True)
        if isinstance(qc.get("quantized_layers"), dict):
            qc["quantized_layers"].update(new_layers)
            print(f"{name}: quantized_layers -> {len(qc['quantized_layers'])}", flush=True)
    json.dump(c, open(f"{DST}/{name}", "w"), indent=2)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


man = {"source": SRC, "flags": {"hc8": HC8, "mtpq": MTPQ}, "fp8_tensors": len(fp8_targets),
       "fp8_rel_rms_err": {"mean": sum(errs.values()) / max(1, len(errs)), "max": max(errs.values(), default=0)},
       "mtp_experts_nvfp4": ({"matrices": len(nv_errs), "rel_l2_mean": sum(nv_errs) / len(nv_errs),
                              "rel_l2_max": max(nv_errs)} if nv_errs else None),
       "quantized_layers_added": len(new_layers),
       "files": {f: sha256(f"{DST}/{f}") for f in files}, "seconds": round(time.time() - t0, 1)}
json.dump(man, open(f"{DST}/view-1003-manifest.json", "w"), indent=1)
print("DONE", json.dumps({k: v for k, v in man.items() if k != "files"}), flush=True)
