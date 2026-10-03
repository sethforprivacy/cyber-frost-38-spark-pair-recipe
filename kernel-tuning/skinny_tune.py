#!/usr/bin/env python3
"""Tune vLLM's CuTe-DSL skinny GEMM (Qwen4Exp low-latency decode GEMM) for GB10 / SM121 at TP2.

vLLM 0.30 only enables qwen4_exp_low_latency_gemm on SM103 / SM90, with plans keyed by TP4 shapes, so on GB10 every
BF16 projection at decode falls back to cuBLAS, which picks SM80 WMMA 16x16 kernels (half of a decode step on B,
2026-10-01 profile). This times cuBLAS (F.linear) against skinny-GEMM configs for each local (N, K) at TP2 and
M in {1, 2, 4, 8, 16}, inside CUDA graphs, rotating weight copies so every call reads from DRAM (L2 cold), and prints
a plan of the configs that beat cuBLAS. Run inside a serving container on an idle arm (needs ~1 GB GPU memory):

  docker exec -i <container> python3 - <shape indices, comma separated | all> < skinny_tune.py
"""
import itertools, json, sys, time

import torch
import torch.nn.functional as F
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import SkinnyGemmConfig, shape_dynamic_skinny_gemm as SG

SHAPES = [  # local (N, K) at TP2, calls per verify forward
    (8192, 2560),    # GDN fused QKVZ, 36
    (6656, 2560),    # QSA fused Q/gate/K/V, 12 (+1 MTP)
    (2560, 3072),    # GDN out_proj + QSA o_proj, 48 (+1)
    (336, 10240),    # HC merged down/injection (replicated), 98
    (10240, 320),    # HC up (replicated), 100
    (640, 2560),     # shared-expert gate/up + QSA indexer Q/K, 62
    (512, 2560),     # router gate (replicated), 49
    (2560, 320),     # shared-expert down, 49
    (48, 2560),      # GDN fused B/A, 36
    (1, 2560),       # shared-expert sigmoid gate, 49
    (124160, 2560),  # LM head (target, M = 4 x seqs)
    (320, 10240),    # final HC down, 2
    (1280, 2560),    # MTP fc_embedding / fc_hidden
]
MS = [1, 2, 4, 8, 16]
DEV = torch.device("cuda", 0)
torch.cuda.set_device(DEV)
sel = sys.argv[1] if len(sys.argv) > 1 else "all"
idx = range(len(SHAPES)) if sel == "all" else [int(i) for i in sel.split(",")]


def timed(fn, ws, x, reps=4):
    """Per-call microseconds of fn(x, w) over a CUDA graph that cycles through the weight copies."""
    for w in ws[:2]:
        fn(x, w)
    torch.cuda.synchronize()
    calls = max(48, len(ws))
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(x, ws[0])
        with torch.cuda.graph(g, stream=s):
            for i in range(calls):
                fn(x, ws[i % len(ws)])
    torch.cuda.current_stream().wait_stream(s)
    g.replay(); torch.cuda.synchronize()
    best = 1e30
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / calls)
    del g
    return best


def candidates(n, k, m):
    out = []
    for bs, opb, ku in itertools.product((32, 64, 128, 256), (1, 2, 4, 8), (1, 2, 4)):
        if n % opb:
            continue
        for vw in (8, 4, 2):
            if k % (bs * vw) == 0:
                break
        else:
            continue
        tiles = k // (bs * vw)
        for sk in ((None, k) if tiles >= 2 else (None,)):
            if sk is not None and ku != 1:
                continue      # static K unrolls the whole loop; k_unroll is unused there
            out.append(SkinnyGemmConfig(m, bs, opb, k_unroll=ku, vector_width=vw, static_k=sk))
    return out


# reference DRAM read bandwidth (sum over a 1 GiB tensor)
big = torch.empty(512 * 1024 * 1024, dtype=torch.bfloat16, device=DEV).normal_()
torch.cuda.synchronize(); t = time.time()
for _ in range(5):
    big.sum()
torch.cuda.synchronize(); print(f"read BW ref: {5 * big.numel() * 2 / (time.time() - t) / 1e9:.0f} GB/s", flush=True)
del big; torch.cuda.empty_cache()

plans = {}
for si in idx:
    n, k = SHAPES[si]
    wbytes = n * k * 2
    copies = max(1, min(1024, (192 << 20) // wbytes))   # > 8x the 24 MB L2: every call reads DRAM
    ws = [torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.02 for _ in range(copies)]
    plan = {}
    shortlist = None   # stage 1 (M=4) searches everything; other M re-time the 5 best M=4 structures
    for m in [4] + [mm for mm in MS if mm != 4]:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
        ref = F.linear(x, ws[0])
        base = timed(lambda a, w: F.linear(a, w), ws, x)
        best = None; scored = []
        cands = candidates(n, k, m) if shortlist is None else [
            SkinnyGemmConfig(m, c.block_size, c.outputs_per_block, k_unroll=c.k_unroll, vector_width=c.vector_width,
                             static_k=c.static_k) for c in shortlist]
        for cfg in cands:
            try:
                y = SG(x, ws[0], cfg)
                err = ((y.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6)).item()
                if err > 2e-2:
                    continue
                us = timed(lambda a, w, c=cfg: SG(a, w, c), ws, x)
            except Exception as ex:  # invalid combination for this shape
                continue
            scored.append((us, cfg))
            if best is None or us < best[0]:
                best = (us, cfg)
        if shortlist is None:
            shortlist = [c for _, c in sorted(scored, key=lambda s: s[0])[:5]]
        gbps = lambda us: wbytes / us / 1e3
        if best and best[0] < base * 0.97:
            c = best[1]
            plan[m] = c
            print(f"({n},{k}) M={m}: cuBLAS {base:7.1f}us ({gbps(base):4.0f} GB/s) -> skinny {best[0]:7.1f}us "
                  f"({gbps(best[0]):4.0f} GB/s) {base / best[0]:.2f}x  cfg bs={c.block_size} opb={c.outputs_per_block} "
                  f"ku={c.k_unroll} vw={c.vector_width} sk={c.static_k}", flush=True)
        else:
            print(f"({n},{k}) M={m}: cuBLAS {base:7.1f}us ({gbps(base):4.0f} GB/s) keeps "
                  f"(best skinny {best[0] if best else float('nan'):.1f}us)", flush=True)
    plans[f"{n},{k}"] = {str(m): [c.num_rows, c.block_size, c.outputs_per_block, c.k_unroll, c.vector_width, c.static_k]
                         for m, c in plan.items()}
    del ws; torch.cuda.empty_cache()
print("PLANS_JSON " + json.dumps(plans), flush=True)
