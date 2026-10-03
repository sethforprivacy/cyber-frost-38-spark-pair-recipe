#!/usr/bin/env python3
"""Is the hyper-connection down/inject projection (merged, replicated: N=336 = 320 low-rank + 4 inject + 12 pad,
K=10240) cheaper in FP8 on GB10? It is the largest BF16 GEMM left in the decode step (~2.9 ms per step, 98 calls
per forward). fp8_linear_bench.py (10-01) stopped at this shape: Marlin's repack needs N % 64 == 0.

Times per M, CUDA-graph replay, weight copies rotated past L2:
  bf16-cublas   F.linear
  bf16-skinny   the production CuTe-DSL skinny GEMM with the SM12x TP2 plan for (336, 10240) (M <= 16)
  fp8-cutlass   W8A8 per-channel weights + dynamic per-token FP8 activations (what FP8_PER_CHANNEL_PER_TOKEN runs)
  fp8-marlin    W8A16 per-channel (N padded to 384 for the repack)
Output error vs BF16 is reported for each FP8 path.
  docker run --rm --gpus all -i <image> python3 - < hc_fp8_bench.py
"""
import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import SkinnyGemmConfig, shape_dynamic_skinny_gemm as SG
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear, marlin_quant_fp8_torch

DEV = torch.device("cuda", 0); torch.cuda.set_device(DEV)
N, K = 336, 10240
PLAN = {1: SkinnyGemmConfig(1, 64, 2, static_k=10240), 2: SkinnyGemmConfig(2, 64, 1, static_k=10240),
        4: SkinnyGemmConfig(4, 64, 2, static_k=10240), 8: SkinnyGemmConfig(8, 64, 1, static_k=10240),
        16: SkinnyGemmConfig(16, 64, 2, static_k=10240)}
MS = [1, 2, 4, 8, 16, 32, 64, 2048]


def timed(fn, nargs, reps=5):
    for a in nargs[:2]:
        fn(*a)
    torch.cuda.synchronize()
    calls = max(48, len(nargs))
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


copies = 48   # 48 x 6.9 MB BF16 > L2
W = [torch.randn(N, K, dtype=torch.bfloat16, device=DEV) * 0.02 for _ in range(copies)]
CQ = []
for w in W:
    wq, ws = ops.scaled_fp8_quant(w, use_per_token_if_dynamic=True)   # per-row (output-channel) scales
    CQ.append((wq.t(), ws.reshape(1, -1).float()))
ws_buf = marlin_make_workspace_new(DEV)
MQ = []
for w in W:
    wp = torch.cat([w, torch.zeros(384 - N, K, dtype=w.dtype, device=DEV)])
    _, qw, qs = marlin_quant_fp8_torch(wp, -1)
    MQ.append((qw, qs))
for m in MS:
    x = torch.randn(m, K, dtype=torch.bfloat16, device=DEV)
    ref = F.linear(x, W[0]).float(); nrm = ref.abs().max().clamp_min(1e-6)
    row = {"bf16-cublas": timed(lambda a, w: F.linear(a, w), [(x, w) for w in W])}
    if m in PLAN:
        row["bf16-skinny"] = timed(lambda a, w: SG(a, w, PLAN[m]), [(x, w) for w in W])

    def fc(a, wq, ws):
        aq, as_ = ops.scaled_fp8_quant(a, use_per_token_if_dynamic=True)
        return ops.cutlass_scaled_mm(aq, wq, as_, ws, out_dtype=torch.bfloat16)
    err_c = ((fc(x, *CQ[0]).float() - ref).abs().max() / nrm).item()
    row["fp8-cutlass"] = timed(fc, [(x, wq, ws) for wq, ws in CQ])
    fm = lambda a, q, s: apply_fp8_marlin_linear(a, q, s, ws_buf, 384, K, None)
    err_m = ((fm(x, *MQ[0])[:, :N].float() - ref).abs().max() / nrm).item()
    row["fp8-marlin384"] = timed(fm, [(x, q, s) for q, s in MQ])
    gbs = N * K * 2 / (row.get("bf16-skinny", row["bf16-cublas"]) * 1e-6) / 1e9
    print(f"M={m:5d} " + " ".join(f"{k}={v:8.1f}" for k, v in row.items()) +
          f"  (bf16 best {gbs:.0f} GB/s)  err cutlass={err_c:.4f} marlin={err_m:.4f}", flush=True)
print("HC_FP8_BENCH_DONE", flush=True)
