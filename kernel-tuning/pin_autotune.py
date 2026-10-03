#!/usr/bin/env python3
"""Build a pinned FlashInfer autotune file for the CyberFrost TP2 pair from one rank-0 tuning result.

  python3 pin_autotune.py <tuned autotune_configs.json> <out.json> [max_decode_bucket=32]

1. Decode buckets (M <= max_decode_bucket) of the NVFP4 MoE GEMMs (trtllm::fused_moe::gemm1/gemm2, uint8 weights)
   all take the tactic most of those buckets chose. The tuner times tactics with hot caches and sometimes picks a
   different tile for one bucket (2026-10-01: gemm1 at M=4 got the 128x64 tile, 167 us/layer in a real decode step,
   against 130 us for the 128x32x256 tile its neighbours chose).
2. Every MoE key is duplicated for tp_rank 1 (cache-key extras: x, w, out dtypes, top_k, tp_size, tp_rank, ...),
   so both ranks hit every entry and none of them tunes alone (vllm#57423).
"""
import collections, json, re, sys

src, dst = sys.argv[1], sys.argv[2]
max_b = int(sys.argv[3]) if len(sys.argv) > 3 else 32
d = json.load(open(src))
KEY = re.compile(r"^\('(?P<op>[^']+)', '(?P<runner>[^']+)', \(\((?P<m>\d+),")
groups = collections.defaultdict(list)
for k, v in d.items():
    mo = KEY.match(k)
    if mo and mo["op"].startswith("trtllm::fused_moe::gemm") and "torch.uint8" in k and int(mo["m"]) <= max_b:
        groups[mo["op"]].append(k)
changed = []
for op, keys in groups.items():
    tactic = collections.Counter(d[k][1] for k in keys).most_common(1)[0][0]
    for k in keys:
        if d[k][1] != tactic:
            changed.append((op, int(KEY.match(k)["m"]), d[k][1], tactic)); d[k] = [d[k][0], tactic]
out = {}
dup = 0
for k, v in d.items():
    out[k] = v
    if k.startswith("('trtllm::fused_moe::"):
        r1 = re.sub(r"(, \d+, 2), 0, (1, 0, 1, 0, )", r"\1, 1, \2", k, count=1)   # top_k, tp_size=2, tp_rank 0 -> 1
        assert r1 != k, k
        out[r1] = v; dup += 1
json.dump(out, open(dst, "w"), indent=1)
print(json.dumps({"entries": len(out), "rank1_duplicates": dup, "decode_pins": changed}))
