# Overlay provenance

Every overlay is a complete file that replaces the image's copy at
`/usr/local/lib/python3.12/dist-packages/vllm/<path>`. Each set has a `SHA256SUMS` (the launcher refuses a mismatch)
and `image-diffs/` with the unified diff of every file against the pinned image
(`myllmbox/qwen38-flash-next-cluster-vllm@sha256:861ac752…`, vLLM 0.30.0 + myllmbox patches). Files derived from
vLLM keep vLLM's Apache-2.0 licence (`../LICENSES/Apache-2.0.txt`).

## vllm-upstream: upstream vLLM fixes, backported to the image

| File | Upstream change | What it fixes on this model |
|---|---|---|
| `models/qwen4_exp/nvidia/ple_layer.py`, `v1/attention/backends/mamba_attn.py`, `v1/attention/backends/short_conv_attn.py` | [vllm#58114](https://github.com/vllm-project/vllm/pull/58114) | PLE metadata overhead per step |
| `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | [vllm#57039](https://github.com/vllm-project/vllm/pull/57039) | redundant GDN prefill copies |
| `models/qwen4_exp/nvidia/ops/qsa_indexer.py` | [vllm#57105](https://github.com/vllm-project/vllm/pull/57105) | QSA indexer workspace |
| `models/qwen4_exp/common/qsa_cache.py` | [vllm#58961](https://github.com/vllm-project/vllm/pull/58961), [vllm#58040](https://github.com/vllm-project/vllm/pull/58040) (hand-ported onto myllmbox's file), [vllm#59528](https://github.com/vllm-project/vllm/pull/59528) | KV memory held after profiling; QSA graph-padding clamp; QSA ring kept off the null block |
| `models/qwen4_exp/nvidia/qsa.py`, `models/qwen4_exp/nvidia/ops/qsa.py` | [vllm#55557](https://github.com/vllm-project/vllm/pull/55557) (merged after the v0.30.0 branch) | `--kv-cache-dtype fp8` on the QSA layers (per-tensor scale 1.0) |

`upstream-prs/` holds the upstream diffs these were ported from (core files only).

## vllm-gb10: GB10 kernel selection

| File | Change |
|---|---|
| `models/qwen4_exp/nvidia/low_latency_gemm.py` | adds SM12x (GB10) plans for the Qwen4Exp CuTe-DSL skinny decode GEMM at TP2 (`kernel-tuning/skinny_tune.py`); upstream ships plans only for SM103 / SM90 at TP4, so on GB10 every decode-size BF16 projection fell back to SM80 WMMA cuBLAS kernels. Reported as [vllm#59605](https://github.com/vllm-project/vllm/issues/59605). |
| `model_executor/warmup/flashinfer_autotune_cache.py` | a fixed FlashInfer tactic-cache directory (`MBX_AUTOTUNE_HASH`), so a pinned tactic file that holds both ranks' keys is used on every boot: no per-boot autotune drift and no warm-cache TP2 deadlock ([vllm#57423](https://github.com/vllm-project/vllm/issues/57423)) |

## vllm-draft: the MTP drafter's output head

| File | Change |
|---|---|
| `models/qwen4_exp/nvidia/mtp.py` | `MBX_DRAFT_HEAD=1`: the drafter computes its logits with a draft-only NVFP4 copy of the output head (quant prefix `dh4`); `MBX_DRAFT_VOCAB=<ids.json>`: that head covers only the listed 49,152 token ids and scatters into the full vocabulary with -inf elsewhere. The verifier keeps the target's head, so outputs are unchanged; only acceptance can move. |

## vllm-roce: RoCEnante all-reduce for the TP2 pair

| File | Change |
|---|---|
| `distributed/device_communicators/cyber_roce_ar.py` | a 2-rank one-shot all-reduce over both halves of the ConnectX-7 port, adapted from b12x's RoCEnante ([local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) @ `e4084d2e`, Apache-2.0): each rank RDMA-writes its input into the peer's pinned host memory (no GPUDirect on GB10), a C proxy posts the writes, one kernel sums in a fixed rank order. Needs the b12x checkout on `PYTHONPATH`. |
| `distributed/device_communicators/cuda_communicator.py` | four hooks: all-reduces up to `CYBER_ROCE_AR_MAX` bytes (1 MiB) go to `cyber_roce_ar`; larger ones and all-gathers stay on NCCL |
