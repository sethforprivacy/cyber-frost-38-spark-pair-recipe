#!/usr/bin/env python3
"""Summarise a vLLM torch-profiler trace (Chrome JSON, .gz ok): GPU busy vs idle, time by kernel family, top kernels.

  python3 prof_analyze.py <trace.json.gz> [--steps N] [--top 25]

--steps divides the totals into per-step figures (pass the capture's spec-decode draft count for decode captures).
"""
import gzip, json, re, sys
from collections import defaultdict

path = sys.argv[1]
steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 0
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
op = gzip.open if path.endswith(".gz") else open
ev = json.load(op(path, "rt"))["traceEvents"]
k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
if not k:
    sys.exit("no GPU events")
FAM = [  # first match wins; order matters
    ("nccl", r"nccl"),
    ("moe-quant/route", r"expandInputRows|finalizeMoe|computeStrides|moe_align|topkGating|routing|permute|unpermute|"
                        r"buildExpertMaps|fused_moe.*(sort|align)|moe_sum|count_and_sort|MoeRouting|doActivation"),
    ("moe-gemm", r"moe|MoE|grouped|Grouped"),
    ("fp4/fp8 quant", r"fp4|Fp4|FP4|quant|e4m3|scaled_mm_scale|cvt_"),
    ("gdn", r"recurrent|gated_delta|delta_rule|chunk_|causal_conv|conv1d|gdn|GDN|mamba|ssm"),
    ("qsa/attn", r"qsa|QSA|indexer|topk|TopK|sparse|flash|fmha|attention|Attention|paged|kv_cache|reshape_and_cache"),
    ("dense-gemm", r"nvjet|gemm|Gemm|GEMM|cutlass|sm90|sm100|sm120|matmul|splitK|cublas|Kernel2"),
    ("norm", r"rms|norm|Norm"),
    ("sample", r"sampl|softmax|argmax|rejection|multinomial|exponential|logits|penalt|top_p|top_k"),
    ("ple/embed", r"ngram|ple|PLE|embedding|Embedding|index_select|gather|indexSelect"),
    ("copy/memset", r"[Mm]emcpy|[Mm]emset|copy|Copy|fill|Fill"),
    ("elementwise", r"elementwise|vectorized|triton_poi|triton_red|triton_per|unrolled|reduce|Reduce|cat|where|silu|"
                    r"sigmoid|mul|add"),
]
fam_t = defaultdict(float); fam_n = defaultdict(int); name_t = defaultdict(float); name_n = defaultdict(int)
for e in k:
    n = e["name"]; d = float(e.get("dur", 0))
    f = next((f for f, rx in FAM if re.search(rx, n)), "other")
    fam_t[f] += d; fam_n[f] += 1; name_t[n] += d; name_n[n] += 1
iv = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))) for e in k)
busy = 0.0; cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce:
        busy += ce - cs; cs, ce = s, e
    else:
        ce = max(ce, e)
busy += ce - cs
span = iv[-1][1] - iv[0][0]; tot = sum(fam_t.values())
streams = defaultdict(float)
for e in k:
    streams[e.get("args", {}).get("stream", "?")] += float(e.get("dur", 0))
print(f"{path.rsplit('/', 1)[-1]}: {len(k)} GPU events, span {span/1e3:.1f} ms, busy {busy/1e3:.1f} ms "
      f"({100*busy/span:.1f} %), kernel-time sum {tot/1e3:.1f} ms; streams "
      + ", ".join(f"{s}:{t/1e3:.0f}ms" for s, t in sorted(streams.items(), key=lambda x: -x[1])[:4]))
per = f" | per step ({steps})" if steps else ""
print(f"{'family':18s} {'ms':>9s} {'%':>6s} {'count':>8s}{per}")
for f, t in sorted(fam_t.items(), key=lambda x: -x[1]):
    ps = f" | {t/steps/1e3:8.3f} ms" if steps else ""
    print(f"{f:18s} {t/1e3:9.1f} {100*t/tot:6.1f} {fam_n[f]:8d}{ps}")
print(f"top {top} kernels:")
for n, t in sorted(name_t.items(), key=lambda x: -x[1])[:top]:
    print(f"  {t/1e3:8.1f} ms {100*t/tot:5.1f} % n={name_n[n]:6d} avg {t/name_n[n]:7.1f} us  {n[:150]}")
