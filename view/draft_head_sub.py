#!/usr/bin/env python3
"""Reduced-vocabulary draft head (FR-Spec style) for the CyberFrost MTP drafter.

The draft-only NVFP4 head (draft_head.py) still covers all 248,320 tokens: 179 MB per rank read on each of the 3
draft passes per step (~2.3 ms of a ~42 ms step on GB10). Tokens the drafter almost never proposes correctly do not
need to be in it. This keeps the K most frequent ids (draft_vocab_stats.py: code/English corpus + the model's own
generations) plus every added/special token, quantizes those rows of the BF16 lm_head exactly as draft_head.py does,
and writes them as a new draft head file in the view. The id list goes to draft-vocab-ids.json; the mtp-drafthead
overlay (MBX_DRAFT_VOCAB) builds a K-row head and scatters its logits into the full vocabulary with -inf elsewhere.
Outputs stay exact (the target verifies every draft); only acceptance can drop, by the target's mass outside the set.

  python3 draft_head_sub.py SRC_WITH_BF16_LM_HEAD VIEW K vocab_stats.json      (inside the vLLM image)
"""
import json, os, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, V, K, STATS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
order = json.load(open(STATS))["order"]
tj = json.load(open(f"{V}/tokenizer.json"))
special = sorted({t["id"] for t in tj.get("added_tokens", [])})
keep = list(dict.fromkeys(special + order))[:max(K, len(special))]
seen = set(keep)
ids = sorted(seen)
assert len(ids) == K, (len(ids), K)
idx_src = json.load(open(f"{SRC}/model.safetensors.index.json"))
with safe_open(f"{SRC}/{idx_src['weight_map']['lm_head.weight']}", "pt") as h:
    lm = h.get_tensor("lm_head.weight")
assert lm.dtype == torch.bfloat16, lm.dtype
w = lm[torch.tensor(ids)].float()
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
    q = (y.abs().unsqueeze(-1) - LUT).abs().argmin(-1)
    code = (q | ((y < 0).long() << 3)).view(n, H)
    packed[r0:r0 + n] = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    scales[r0:r0 + n] = s
    deq = (torch.where(code >= 8, -LUT[code & 7], LUT[code & 7]).view(n, H // 16, 16) * sf.unsqueeze(-1)).view(n, H)
    err2 += float(((deq - x) ** 2).sum()); ref2 += float((x * x).sum())
out = {"mtp.draft_head.weight": packed, "mtp.draft_head.weight_scale": scales,
       "mtp.draft_head.weight_scale_2": torch.tensor(g, dtype=torch.float32)}
fn = f"draft-head-nvfp4-v{K}.safetensors"
save_file(out, f"{V}/{fn}", metadata={"format": "pt"})
idx = json.load(open(f"{V}/model.safetensors.index.json")); wm = idx["weight_map"]
old = {wm[k] for k in out if k in wm}
for k in out:
    wm[k] = fn
json.dump(idx, open(f"{V}/model.safetensors.index.json", "w"), indent=2)
for o in old - {fn}:                       # drop the full head's link from this view only
    if os.path.lexists(f"{V}/{o}") and o not in set(wm.values()):
        os.unlink(f"{V}/{o}")
json.dump({"k": K, "special": len(special), "ids": ids}, open(f"{V}/draft-vocab-ids.json", "w"))
print(json.dumps({"k": K, "special": len(special), "rel_rms_err": (err2 / ref2) ** 0.5, "file": fn,
                  "dropped": sorted(old - {fn}), "seconds": round(time.time() - t0, 1)}))
