# Kernel tuning and analysis

The measurements behind the profile's kernel choices. GPU scripts run inside the serving image on an idle node
(`docker run --rm --gpus all --ipc host --entrypoint python3 -i <image> - < script.py`); the rest are stdlib Python.

| script | what it measures | decision it backs |
|---|---|---|
| `prof_run.py`, `prof_analyze.py` | torch-profiler traces of a decode step at 1 and 8 streams (server booted with `--profiler-config`), grouped by kernel family | where the step goes: 55 % BF16 dense projections, 22 % NVFP4 MoE, 7 % all-reduce (10-01) |
| `skinny_tune.py` | the Qwen4Exp CuTe-DSL skinny GEMM against cuBLAS for every TP2 projection shape at M = 1-16, in CUDA graphs, weights rotated past L2; prints the plan table | `overlays/vllm-gb10` plans: 240-255 GB/s vs 160-225 GB/s for cuBLAS |
| `fp8_linear_bench.py` | BF16 (cuBLAS, skinny) vs FP8 W8A16 Marlin vs FP8 W8A8 CUTLASS per shape and M | FP8 per-channel for the dense projections and the head (1.9x at decode sizes, 1.4x at M = 2048) |
| `hc_fp8_bench.py` | the same for the hyper-connection down/inject GEMM (N = 336, K = 10,240) | rejected: 24 vs 27.5 µs at decode, 368 vs 246 µs at M = 2048 |
| `pin_autotune.py` | turns one boot's FlashInfer tactic cache into a pinned file: decode buckets smoothed, keys for both TP ranks | `launcher/pinned-autotune-moe-v1.json` |
| `draft_vocab_gen.py`, `draft_vocab_stats.py` | the model's own generations plus a local code/docs corpus, tokenized and counted; coverage of the top-K ids on held-out text | the 49,152-id draft vocabulary (99.94 % coverage) |
| `werr.py`, `ratios.py` | per-expert gate/up global scales of an NVFP4 checkpoint; relative weight error against the BF16 source by HTTP range reads | the V1 mis-scale (ratios 0.21-4.83), the fold's extra error (+3.8 %), V2 matching the per-projection export |
