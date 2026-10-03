#!/usr/bin/env python3
"""Add a DRAFT-ONLY W4A16 NVFP4 copy of CYBER-FROST's lm_head to a checkpoint view (run inside the vLLM image).

The MTP drafter shares the target's BF16 lm_head (636 MB/rank), read once per draft token (K=3 -> 3x/step on
top of the verify). A 4-bit copy used only for drafting cuts those reads ~3.5x. Exact for outputs: the verifier
still uses the BF16 head and rejection sampling keeps the target distribution; only acceptance can move.
Format = hibrid48's model-lmhead-nvfp4.safetensors (W4A16_NVFP4: U8 [V, H/2] e2m1 pairs low-nibble-first,
F8_E4M3 [V, H/16] block scales, F32 scalar global = amax/(6*448)), under names mtp.draft_head.* (the target's
loader skips mtp.*; the MTP remap sends them to model.draft_head.*).

  python3 draft_head.py VIEW_DIR          (VIEW_DIR: an existing checkpoint view; files written into it)
"""
import json, os, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

V = sys.argv[1]
idx = json.load(open(f"{V}/model.safetensors.index.json"))
wm = idx["weight_map"]
with safe_open(f"{V}/{wm['lm_head.weight']}", "pt") as h:
    w = h.get_tensor("lm_head.weight").float()          # [248320, 2560]
t0 = time.time()
LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
g = w.abs().max().item() / (6.0 * 448.0)
R, H = w.shape
packed = torch.empty(R, H // 2, dtype=torch.uint8); scales = torch.empty(R, H // 16, dtype=torch.float8_e4m3fn)
err2 = ref2 = 0.0
for r0 in range(0, R, 8192):
    x = w[r0:r0 + 8192]; n = x.shape[0]
    xb = x.view(n, H // 16, 16); amax = xb.abs().amax(-1)
    s = (amax / (6.0 * g)).to(torch.float8_e4m3fn); sf = (s.float() * g).clamp_min(1e-30)
    y = xb / sf.unsqueeze(-1)
    idx_ = (y.abs().unsqueeze(-1) - LUT).abs().argmin(-1)
    code = (idx_ | ((y < 0).long() << 3)).view(n, H)
    packed[r0:r0 + n] = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    scales[r0:r0 + n] = s
    deq = (torch.where(code >= 8, -LUT[code & 7], LUT[code & 7]).view(n, H // 16, 16) * sf.unsqueeze(-1)).view(n, H)
    err2 += float(((deq - x) ** 2).sum()); ref2 += float((x * x).sum())
out = {"mtp.draft_head.weight": packed, "mtp.draft_head.weight_scale": scales,
       "mtp.draft_head.weight_scale_2": torch.tensor(g, dtype=torch.float32)}
fn = "draft-head-nvfp4.safetensors"
save_file(out, f"{V}/{fn}", metadata={"format": "pt"})
for k in out:
    wm[k] = fn
idx["metadata"]["total_size"] += sum(t.numel() * t.element_size() for t in out.values())
json.dump(idx, open(f"{V}/model.safetensors.index.json", "w"), indent=2)
cfg = json.load(open(f"{V}/config.json"))
ql = cfg["quantization_config"]["quantized_layers"]
for p in ("mtp.draft_head", "model.mtp.draft_head", "draft_head", "model.draft_head", "language_model.mtp.draft_head"):
    ql[p] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
json.dump(cfg, open(f"{V}/config.json", "w"), indent=2)
print(f"draft head: global {g:.4e}, rel RMS err {(err2/ref2)**.5:.4f}, {time.time()-t0:.0f}s -> {V}/{fn}")
