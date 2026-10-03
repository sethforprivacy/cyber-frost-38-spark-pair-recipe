#!/usr/bin/env python3
"""Can FP8 weights make CyberFrost's BF16 projections cheaper on GB10? Times, per local (N, K) at TP2 and per M:
  bf16-cublas   F.linear (what the model runs today)
  bf16-skinny   the tuned CuTe-DSL skinny GEMM (M <= 16; plan file from skinny_tune.py)
  fp8-marlin    W8A16, per-output-channel FP8 E4M3 weights (vLLM Marlin FP8 GEMM)
  fp8-cutlass   W8A8, per-channel weights + dynamic per-token FP8 activations (vLLM CUTLASS scaled_mm)
CUDA-graph replay, weight copies rotated past L2. Also reports each FP8 path's output error against BF16.
  docker exec -i <container> python3 - <plans.json path inside container | -> < fp8_linear_bench.py
"""
import json, sys

import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import SkinnyGemmConfig, shape_dynamic_skinny_gemm as SG
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear, marlin_quant_fp8_torch

DEV = torch.device("cuda", 0); torch.cuda.set_device(DEV)
SHAPES = [(8192, 2560), (6656, 2560), (2560, 3072), (336, 10240), (10240, 320), (640, 2560), (512, 2560),
          (2560, 320), (124160, 2560)]
MS = [1, 4, 8, 16, 32, 2048]
plans = {}
if len(sys.argv) > 1 and sys.argv[1] != "-":
    for line in open(sys.argv[1]):
        if "PLANS_JSON" in line:
            plans.update(json.loads(line.split("PLANS_JSON ", 1)[1]))


def timed(fn, nargs, reps=4):
    for a in nargs[:2]:
        fn(*a)
    torch.cuda.synchronize()
    calls = max(24, len(nargs))
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(*nargs[0])
        with torch.cuda.graph(g, stream=s):
            for i in range(calls):
                fn(*nargs[i % len(nargs)])
    torch.cuda.current_stream().wait_stream(s); g.replay(); torch.cuda.synchronize()
    best = 1e30
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / calls)
    return best


ws_buf = marlin_make_workspace_new(DEV)
for n, k in SHAPES:
    copies = max(1, min(256, (192 << 20) // (n * k * 2)))
    W = [torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.02 for _ in range(copies)]
    MQ = []; CQ = []
    for w in W:
        _, qw, qs = marlin_quant_fp8_torch(w, -1)
        MQ.append((qw, qs))
        wq, ws = ops.scaled_fp8_quant(w, use_per_token_if_dynamic=True)   # per-row (output channel) scales
        CQ.append((wq.t(), ws.reshape(1, -1).float()))
    for m in (MS if n * k < 100_000_000 else [mm for mm in MS if mm <= 32]):
        x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
        ref = F.linear(x, W[0]).float(); nrm = ref.abs().max().clamp_min(1e-6)
        row = {"bf16-cublas": timed(lambda a, w: F.linear(a, w), [(x, w) for w in W])}
        p = plans.get(f"{n},{k}", {}).get(str(m))
        if p:
            cfg = SkinnyGemmConfig(p[0], p[1], p[2], k_unroll=p[3], vector_width=p[4], static_k=p[5])
            row["bf16-skinny"] = timed(lambda a, w: SG(a, w, cfg), [(x, w) for w in W])
        try:
            fm = lambda a, q, s: apply_fp8_marlin_linear(a, q, s, ws_buf, n, k, None)
            err_m = ((fm(x, *MQ[0]).float() - ref).abs().max() / nrm).item()
            row["fp8-marlin"] = timed(fm, [(x, q, s) for q, s in MQ])
        except Exception as ex:
            err_m = None; row["fp8-marlin"] = f"ERR {type(ex).__name__}: {str(ex)[:80]}"
        try:
            def fc(a, wq, ws):
                aq, as_ = ops.scaled_fp8_quant(a, use_per_token_if_dynamic=True)
                return ops.cutlass_scaled_mm(aq, wq, as_, ws, out_dtype=torch.bfloat16)
            err_c = ((fc(x, *CQ[0]).float() - ref).abs().max() / nrm).item()
            row["fp8-cutlass"] = timed(fc, [(x, wq, ws) for wq, ws in CQ])
        except Exception as ex:
            err_c = None; row["fp8-cutlass"] = f"ERR {type(ex).__name__}: {str(ex)[:80]}"
        fmt = lambda v: f"{v:8.1f}" if isinstance(v, float) else v
        print(f"({n},{k}) M={m:5d} " + " ".join(f"{kk}={fmt(vv)}" for kk, vv in row.items()) +
              f"  err marlin={err_m if err_m is None else round(err_m, 4)} cutlass={err_c if err_c is None else round(err_c, 4)}",
              flush=True)
    del W, MQ, CQ; torch.cuda.empty_cache()
print("FP8_BENCH_DONE", flush=True)
