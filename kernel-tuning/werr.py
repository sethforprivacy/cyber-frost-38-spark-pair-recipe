"""Weight error of routed experts vs the BF16 source: V1 as exported, V1 as served (w13 fold), V2.

Reads only the tensors it needs, by HTTP range requests against the Hub (no full download; numpy needed).
  DIR must hold: bf16_index.json (Blackfrost-AI/CYBER-FROST-3.8-BF16's model.safetensors.index.json at the pinned
  revision), nv_index.json (the NVFP4 repos' index; V1 and V2 share it), v1_ratios.json (ratios.py over a local V1
  copy: per layer, [expert, gate weight_scale_2, up weight_scale_2] for all 512 experts).
  python3 werr.py DIR        -> DIR/werr.json; per layer: the 3 lowest and 3 highest gate/up ratios + 4 random experts
"""
import json, struct, sys, random, urllib.request
import numpy as np
HF = "https://huggingface.co/Blackfrost-AI"
REPOS = {"bf16": ("CYBER-FROST-3.8-BF16", "5321904427c4ef54df8a667edcbc2d1184e4286e"),
         "v1": ("CYBER-FROST-3.8-NVFP4", "ac0e3c69af06f85a743afcd95a15e1984d81e774"),
         "v2": ("CYBER-FROST-3.8-NVFP4-V2", "67fced339ed73eb4f21d31ac0a81c06f479dcb83")}
S = sys.argv[1]
idx = {"bf16": json.load(open(f"{S}/bf16_index.json"))["weight_map"],
       "nv": json.load(open(f"{S}/nv_index.json"))["weight_map"]}
hdrs = {}
def rng(url, a, b):
    for t in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b-1}"}), timeout=120).read()
        except Exception as ex:
            err = ex
    raise err
def url(r, f): return f"{HF}/{REPOS[r][0]}/resolve/{REPOS[r][1]}/{f}"
def header(r, f):
    if (r, f) not in hdrs:
        n = struct.unpack("<Q", rng(url(r, f), 0, 8))[0]
        hdrs[(r, f)] = (8 + n, json.loads(rng(url(r, f), 8, 8 + n)))
    return hdrs[(r, f)]
def tensor(r, k, sub=None):
    f = (idx["bf16"] if r == "bf16" else idx["nv"])[k]
    off, h = header(r, f); a, b = h[k]["data_offsets"]
    if sub: a, b = a + sub[0], a + sub[1]
    return rng(url(r, f), off + a, off + b), h[k]
# tables
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
def e4m3_val(b):
    s, e, m = b >> 7, (b >> 3) & 15, b & 7
    if e == 15 and m == 7: return np.nan
    v = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
    return -v if s else v
E4M3 = np.array([e4m3_val(i) for i in range(256)], np.float32)
POS = np.sort(np.unique(E4M3[(E4M3 >= 0) & ~np.isnan(E4M3)]))
def to_e4m3(x):  # round to nearest representable (positive scales only)
    i = np.clip(np.searchsorted(POS, x), 1, len(POS) - 1)
    lo, hi = POS[i - 1], POS[i]
    return np.where(x - lo <= hi - x, lo, hi)
def nvfp4(r, p):
    w, _ = tensor(r, f"{p}.weight"); s, _ = tensor(r, f"{p}.weight_scale"); g, _ = tensor(r, f"{p}.weight_scale_2")
    q = np.frombuffer(w, np.uint8).reshape(640, 1280)
    codes = np.stack([q & 15, q >> 4], -1).reshape(640, 2560)
    bs = E4M3[np.frombuffer(s, np.uint8)].reshape(640, 160)
    return E2M1[codes], bs, struct.unpack("<f", g)[0]
def deq(vals, bs, g): return vals * np.repeat(bs, 16, 1) * g
def rel(a, ref): return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))
ratios = json.load(open(f"{S}/v1_ratios.json"))
rows = []
random.seed(0)
for L, r in ratios.items():
    by = sorted(r, key=lambda x: x[1] / x[2])
    pick = [x[0] for x in by[:3] + by[-3:]] + random.sample([x[0] for x in by[3:-3]], 4)
    k = f"model.language_model.layers.{L}.mlp.experts.gate_up_proj"
    for e in pick:
        nb = 1280 * 2560 * 2
        raw, _ = tensor("bf16", k, (e * nb, (e + 1) * nb))
        ref = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(1280, 2560)
        p = f"model.language_model.layers.{L}.mlp.experts.{e}"
        (v1g, b1g, g1g), (v1u, b1u, g1u) = nvfp4("v1", f"{p}.gate_proj"), nvfp4("v1", f"{p}.up_proj")
        (v2g, b2g, g2g), (v2u, b2u, g2u) = nvfp4("v2", f"{p}.gate_proj"), nvfp4("v2", f"{p}.up_proj")
        rg, ru = ref[:640], ref[640:]
        gmax = max(g1g, g1u)
        fg = deq(v1g, to_e4m3(b1g * g1g / gmax), gmax); fu = deq(v1u, to_e4m3(b1u * g1u / gmax), gmax)
        row = dict(layer=int(L), expert=e, ratio=g1g / g1u, v2_scales_equal=g2g == g2u,
                   v1_gate=rel(deq(v1g, b1g, g1g), rg), v1_up=rel(deq(v1u, b1u, g1u), ru),
                   fold_gate=rel(fg, rg), fold_up=rel(fu, ru),
                   unfixed_up=rel(deq(v1u, b1u, g1g), ru),
                   v2_gate=rel(deq(v2g, b2g, g2g), rg), v2_up=rel(deq(v2u, b2u, g2u), ru),
                   v1_sub=float(np.mean(b1g * g1g / gmax < 2 ** -6) + np.mean(b1u * g1u / gmax < 2 ** -6)) / 2,
                   v2_sub=float(np.mean(b2g < 2 ** -6) + np.mean(b2u < 2 ** -6)) / 2)
        rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(f"{S}/werr.json", "w"), indent=1)
