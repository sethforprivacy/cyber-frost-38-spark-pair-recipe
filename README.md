# CYBER-FROST-3.8 on two DGX Sparks: a TP2 recipe for vLLM

By Seth For Privacy.

This recipe serves [Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2](https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2)
at revision `67fced339ed73eb4f21d31ac0a81c06f479dcb83` with tensor parallelism 2 across two NVIDIA DGX Spark (GB10)
nodes joined by one direct QSFP cable. CYBER-FROST-3.8 is Blackfrost's fine-tune of Qwen3.8-Flash-Next
(`qwen4_exp`: 48 layers of gated DeltaNet and QSA attention, 512 routed experts with 10 active, hyper-connections, a
per-layer n-gram embedding table, one MTP layer). The checkpoint has NVFP4 routed experts and everything else in BF16.
It accepts images.

The stack:
- **Engine:** vLLM 0.30.0 from the public image `myllmbox/qwen38-flash-next-cluster-vllm`, pinned by digest.
- **Overlays:** four sets of file overlays, sha-checked and mounted read-only:
  - upstream vLLM fixes backported to the image (FP8 KV for QSA among them);
  - GB10 decode-GEMM plans and a pinned FlashInfer tactic cache;
  - a draft-only, reduced-vocabulary output head for the MTP drafter;
  - a two-rank RDMA all-reduce built on b12x's RoCEnante.
- **A serving view of the checkpoint**, built on each node by `view/build-view.sh`: FP8 for the n-gram table, the
  dense projections and the output head; an NVFP4 draft head; the MTP drafter in W4A16 NVFP4 / FP8; YaRN to 524,288
  tokens. The checkpoint itself is never modified and the weights are not redistributed here.

It is the production profile we have served since 2026-10-03, with the benchmark and microbenchmark scripts behind
every number below. Every number was measured on our own pair.

**Status:** a research-grade recipe from one pair of nodes. **Licence:** MIT for our own files; third-party pieces
keep their own licences (see [Licence](#licence)).

## Contents

- [Results: launch day vs today](#results-launch-day-2026-09-29-vs-today-2026-10-03)
- [Hardware and cabling](#hardware-and-cabling)
- [Software pins](#software-pins)
- [Bring-up](#bring-up)
- [The production profile](#the-production-profile)
- [What each change does](#what-each-change-does)
- [Tried and rejected](#tried-and-rejected)
- [Rollback](#rollback)
- [Limitations and caveats](#limitations-and-caveats)
- [Reproducing the numbers](#reproducing-the-numbers)
- [Layout](#layout) · [Credits](#credits) · [Licence](#licence)

## Results: launch day (2026-09-29) vs today (2026-10-03)

The hardware and cables are the same in both columns. "Launch" is the 2026-09-29 profile on the checkpoint as
first published (V1: TP2, MTP K=3, 2 seats, 262,144 context, BF16 KV 8 GB per rank, 2,048 batched tokens).
"Today" is `launcher/profile.env` on the V2 checkpoint, on freshly rebooted nodes.

| | launch | today | change |
|---|---:|---:|---:|
| Held-out NLL, 29,166 tokens of code and prose (lower is better) | 0.4709 | **0.3777** | -20 % |
| Top-1 agreement with the held-out text | 0.896 | 0.916 | |
| Max context / seats | 262,144 / 2 | **524,288 / 16** | 2x / 8x |
| KV cache per rank | 8 GB BF16 | 24 GB FP8 (2,825,969 tokens) | |
| Single stream: median decode tok/s; ms per decode step | 46.4; 59.8 | **75.1; 34.2** | +62 %; -43 % |
| Every seat busy: aggregate tok/s | 76 (2 streams) | **270** (16 streams) | 3.5x |
| 8 streams: aggregate tok/s | (2 seats) | 189 | |
| Cold prefill 7K / 56K / 175K tokens (tok/s) | 3,220 / 3,038 / 2,855 | 3,666 / 3,717 / 3,552 | +14 % / +22 % / +24 % |
| ~500K-token needle (prefill + answer) | not servable | 500,136 tokens, 159 s | new |
| Model memory per rank | 88.6 GiB | 61.8 GiB | |

- NLL: teacher-forced, on fixed chunks of code and prose from the host OS (`bench/cyber_bench.py`). Most of the
  launch-day gap is V1's mis-scaled experts (see [What each change does](#what-each-change-does)). On the larger
  165,433-token set (`bench/nll_big.py`), today's three runs: 0.28901 / 0.28934 / 0.28960.
- Decode: thinking on, the published sampling (T=1, top-p 0.95, top-k 20), six code-review prompts of 1,200
  tokens each. Tokens per second move with how many draft tokens are accepted (2.77 per step at launch, 2.38-2.50
  today: the corrected target agrees less with the provisional MTP layer); ms per step does not, so it is the
  number to compare.
- Prefill: one cold request at a time (unique prefix), `max_tokens` 1.
- Needle: a passphrase in the middle of ~500K tokens of filler, retrieved exactly; a 530K-token prompt is refused
  with 400 and the server stays healthy.

## Hardware and cabling

- **Nodes.** Two DGX Spark (GB10, 128 GB unified memory, 10 Cortex-X925 + 10 Cortex-A725 cores). Each node has a
  ConnectX-7 with two QSFP ports; each port appears as **two PCIe functions on two PCIe root domains**:

  | QSFP port | PCIe domain 0 | PCIe domain 2 |
  |---|---|---|
  | port 0 (cabled) | `rocep1s0f0` (netdev `enp1s0f0np0`) | `roceP2p1s0f0` (netdev `enP2p1s0f0np0`) |
  | port 1 (unused) | `rocep1s0f1` | `roceP2p1s0f1` |

- **Cable.** One QSFP cable between the two nodes' port 0. Both PCIe halves of that port come up at 200 Gb/s.
  We call them rail 1 (domain 0) and rail 2 (domain 2).
- **Addressing.** Each rail gets its own point-to-point /24 (example: `10.100.1.0/24` on rail 1, `10.100.2.0/24` on
  rail 2). Rail 1 also carries torch.distributed, gloo and the NCCL bootstrap (`RANK0_FABRIC_IP`, `FABRIC_IFNAME`).
- **Fabric settings.** MTU 9000 on both rails; RoCE active MTU 4096. **GID index 3 must be the RoCE v2 IPv4-mapped
  GID** on both functions (the launcher sets `NCCL_IB_GID_INDEX=3`). Use static addressing: a DHCP / IPv6 profile on
  a fabric function can take the GID slots you expect for IPv4.
- **Management.** ssh from rank 0 to rank 1 (key-based) and the API on rank 0 can use any network; nothing else
  needs a switch.

Checks per node (read-only):

```bash
ip -br addr show enp1s0f0np0 enP2p1s0f0np0                                   # two addresses, UP, mtu 9000
for d in rocep1s0f0 roceP2p1s0f0; do
  echo "$d $(cat /sys/class/infiniband/$d/ports/1/gids/3) $(cat /sys/class/infiniband/$d/ports/1/gid_attrs/types/3)"
done                                                                          # 0000:...:ffff:<ipv4 in hex> and "RoCE v2"
ping -M do -s 8972 -c 3 <the other node's address on each rail>              # jumbo frames end to end
```

## Software pins

| piece | pin |
|---|---|
| Serving image | `myllmbox/qwen38-flash-next-cluster-vllm@sha256:861ac752164e0d723c5eff3f876586c6678c26ad4a516112c48745f6a101ff4d`, image ID (config digest) `sha256:4d03fc80b2c28088808115eec524cb36732c8181b13ba15085095fa927b91815`, arm64. vLLM 0.30.0, torch 2.13.0+cu130, FlashInfer 0.6.18.post1, plus myllmbox's Qwen3.8-Flash-Next patches. |
| Checkpoint | `Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2` @ `67fced339ed73eb4f21d31ac0a81c06f479dcb83` (2026-10-02 export with one global scale per expert's gate/up pair), identical on both nodes (`launcher/fetch-checkpoint.py` verifies every file against the Hub's digests) |
| b12x (RoCEnante) | [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) @ `e4084d2eef4932e0fa06db3f7a94deb83ad132d7` (b12x 1.5.0, Apache-2.0). A checkout is mounted **read-only** into the container on `PYTHONPATH`; nothing is installed. The launcher refuses to start if the checkout is not at that commit. |
| Overlays | `overlays/` = the four production overlay sets, identical to production apart from one comment line (see `overlays/NOTICE`), each with `SHA256SUMS` (checked at every start), the diff against the image and its upstream source ([`overlays/PROVENANCE.md`](overlays/PROVENANCE.md)) |
| Host | DGX OS (Ubuntu 24.04.4, glibc 2.39), kernel 6.17.0-1029-nvidia, NVIDIA driver 580.173.02, Docker 29.2.1 with the NVIDIA container runtime, python3 and git on the host |

The image is the one we run with only an entrypoint layer of our own on top (it reads the key file and redacts keys
from the logs); the launcher here passes the keys with `--api-key` instead, so you can use the public image as is.

## Bring-up

Both nodes need the same user, the same recipe path, and key-based non-interactive ssh from rank 0 to rank 1.

1. **Fabric.** Cable and address the pair as above. Run the checks on both nodes.
2. **Image, on both nodes** (or pull once and relay with `docker save | ssh <node> docker load`; the launcher checks
   the image ID, which survives a relay):
   ```bash
   docker pull myllmbox/qwen38-flash-next-cluster-vllm@sha256:861ac752164e0d723c5eff3f876586c6678c26ad4a516112c48745f6a101ff4d
   docker image inspect --format '{{.Id}}' myllmbox/qwen38-flash-next-cluster-vllm@sha256:861ac752164e0d723c5eff3f876586c6678c26ad4a516112c48745f6a101ff4d
   # sha256:4d03fc80b2c2...
   ```
   With Docker's containerd image store, `.Id` can be the manifest digest instead of the config digest. In that case
   set `VLLM_IMAGE_ID` to what your daemon prints for the image you pulled by digest.
3. **Checkpoint, staged once and copied** (174 GiB):
   ```bash
   python3 launcher/fetch-checkpoint.py --dest /srv/models      # needs huggingface_hub + hf_xet
   # copy /srv/models/CYBER-FROST-3.8-NVFP4-V2/ to rank 1 (rsync over the fabric), then on each node:
   cd /srv/models/CYBER-FROST-3.8-NVFP4-V2/67fced339ed73eb4f21d31ac0a81c06f479dcb83 && sha256sum -c --quiet ../67fced339ed73eb4f21d31ac0a81c06f479dcb83.sha256
   ```
4. **Serving view, on both nodes** (same command on each; the result is identical):
   ```bash
   view/build-view.sh /srv/models/CYBER-FROST-3.8-NVFP4-V2/67fced339ed73eb4f21d31ac0a81c06f479dcb83 \
                      /srv/models/CYBER-FROST-3.8-NVFP4-V2-serving
   ```
   It runs the tensor steps in the serving image (CPU only, about 10 minutes on a DGX Spark) and writes 77 GiB of
   new files next to 42 GiB of hardlinks to the checkpoint. Check the result on each node:
   ```bash
   cd /srv/models/CYBER-FROST-3.8-NVFP4-V2-serving && sha256sum -c --quiet /path/to/recipe/view/VIEW.sha256
   ```
   `view/VIEW.sha256` lists every file of the view we serve except the four `*-manifest.json` build records (they
   hold timings). Our two nodes and our production view match it; `config.json` differs from production only in
   its `_name_or_path` string, because our production view was first derived from V1.
5. **b12x, on both nodes:**
   ```bash
   git clone https://github.com/local-inference-lab/b12x ~/b12x/e4084d2e
   git -C ~/b12x/e4084d2e checkout --detach e4084d2eef4932e0fa06db3f7a94deb83ad132d7
   ```
6. **Site file and keys.**
   - `cp launcher/hosts.env.example launcher/hosts.env` and edit the addresses, `LINK_CHECKS`, `VIEW` and paths.
     The file must be identical on both nodes.
   - Put the API key file (one key per line) at `API_KEY_FILE` on **both** nodes. Both ranks parse the same command
     line, though only rank 0 serves HTTP. The file is bind-mounted read-only and read at exec time, so no key
     appears in `docker inspect`.
7. **Dry run.** `launcher/pair-up.sh check` prints both ranks' `docker run` commands. On each rank it verifies the
   image ID, the view, the b12x commit and every overlay's checksums. It starts nothing.
8. **Boot.** Reboot both nodes first. GB10 unified memory fragments over time; an un-rebooted node can lose a large
   share of its throughput while passing every memory check. Then run, on rank 0:
   ```bash
   launcher/pair-up.sh up        # or from rank 0's crontab: @reboot sleep 60 && bash .../launcher/pair-up.sh up >> ~/cyberfrost-run/owner.log 2>&1
   ```
   - What `pair-up.sh up` does:
     - waits for ssh to rank 1, one fabric ping per rail and > 100 GiB MemAvailable on both nodes; if rail 2 is down
       it starts on rail 1 alone with RoCEnante off, and if rail 1 is down it refuses;
     - starts rank 1 (headless) and then rank 0, which serves the API on `PORT`, each with a host memory guard;
     - waits up to 30 min for `/health`.
   - Timing: about 11 minutes from start to healthy (most of it is the lazy safetensors load).
9. **Verify:** `bench/cyber_qualify.py <out-dir>` with `KEYFILE`, `LONG_TOKENS=500000` and `REFUSE_TOKENS=530000`:
   every key answers, unknown keys get 401, vision, a forced tool call and its continuation, a 500K-token needle,
   a 530K-token prompt refused with 400 (server still healthy), 8 concurrent requests, strict JSON.

A request:

```bash
curl -s http://<rank0>:8901/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"CYBER-FROST-3.8-NVFP4","messages":[{"role":"user","content":"Hello"}]}'
```

Thinking is on (the chat template forces it; `--reasoning-parser qwen3` returns it separately). Tool calls use
`--tool-call-parser qwen3_xml`. The published sampling is T=1.0, top_p=0.95, top_k=20.

## The production profile

`launcher/profile.env`, passed to both ranks by `pair-up.sh`:

| knob | value | why |
|---|---|---|
| `--moe-backend` | `marlin` | W4A16 routed experts: BF16 activations instead of NVFP4 activations (NLL −0.006) and a faster decode step; prefill pays for it (see below) |
| `--max-model-len` | 524,288 | YaRN ×2 over the native 262,144, set in the view so the MTP drafter gets the same RoPE |
| `--kv-cache-dtype` / `--kv-cache-memory-bytes` | `fp8` / 24 GB per rank | 2,825,969 tokens of KV: 5.39 full-length requests at once |
| `--max-num-seqs` / `--max-num-batched-tokens` | 16 / 8,192 | 16,384 tested (below) |
| `--speculative-config` | MTP, 3 draft tokens, block verification, probabilistic drafts | the 4th draft position is accepted too rarely to pay for itself |
| `--compilation-config` | full and piecewise CUDA graphs, capture sizes 1-64 | |
| `--safetensors-load-strategy` | `lazy` | fastsafetensors' MTP pass spikes host memory on GB10 |
| `--gpu-memory-utilization` | 0.70 | with an explicit KV size this only sets vLLM's start-up free-memory check |
| NCCL | both rails (`NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0`, `NCCL_IB_MERGE_NICS=1`), 4 channels, GID 3 | +6 % prefill vs one rail |
| RoCEnante | all-reduces ≤ 1 MiB (every decode-size one) | ~110 decode all-reduces per step at ~30 µs each on NCCL; RoCEnante saves ~1.1 ms per step |
| `MBX_DRAFT_HEAD` / `MBX_DRAFT_VOCAB` | 1 / `/model/draft-vocab-ids.json` | the draft-only NVFP4 head over 49,152 tokens |
| `MBX_FUSED_DRAFT`, `MBX_PLE_REPLICATE=0` | myllmbox switches | fused draft path; the n-gram table is sharded, not replicated |
| `PINNED_TACTICS` | `pinned-autotune-moe-v1.json` | FlashInfer MoE tactics for the rollback profile; unused with Marlin |

## What each change does

Every row was measured as an A/B on this pair, one change at a time (the 10-03 rows on freshly rebooted nodes);
"step" is the time of one decode step at one stream (independent of draft acceptance). Dates are when the change
went into production (UTC).

| date | change | effect |
|---|---|---|
| 09-30 | **w1/w3 scale fold** (V1 only; vllm#55073's method) | V1 shipped gate and up with different global scales; vLLM keeps one per expert, so every up-projection was mis-scaled. NLL 0.4755 → 0.3866. V2 fixes it in the checkpoint: no fold needed, and V2 is 0.0011 better again (no extra rounding) |
| 09-30 | **FP8 n-gram table** (`ple_fp8.py`) | 48 → 24 GiB per rank (one global scale, 2.65 % RMS error); 0.0073 better NLL than hibrid48's NVFP4 table. This is what makes room for 16 seats and 24 GB of KV |
| 09-30 | **draft-only NVFP4 output head** (`draft_head.py`, overlay `vllm-draft`) | +21 % decode at equal acceptance: the drafter reads a 4-bit copy of the 248,320-row head three times per step |
| 09-30 | **upstream backports** (overlay `vllm-upstream`) | overhead and correctness fixes (PLE metadata, GDN prefill copies, KV held after profiling, QSA indexer workspace and graph padding): prefill +1-2 %, decode unchanged |
| 09-30 | **YaRN ×2** (`view_steps.py yarn512k`) | 524,288-token context; needles at 342K / 422K / 500K retrieved; short-context NLL unchanged |
| 10-01 | **FP8 KV on QSA** (vllm#55557 in `vllm-upstream`) | KV pool 1.59M → 2.83M tokens at the same 24 GB/rank; NLL within run-to-run noise |
| 10-01 | **skinny decode GEMM plans for GB10** (overlay `vllm-gb10`) | decode-size BF16 projections ran SM80 WMMA cuBLAS kernels at 125-205 GB/s; the CuTe-DSL skinny GEMM with TP2 plans reaches 240-255 GB/s. Dense GEMM time −11 % |
| 10-01 | **FP8 dense projections** (`fp8_dense_view.py`) | GDN in/out, QSA q/k/v/o, shared expert and hyper-connection up in FP8 per channel: step −16 % (51.0 → 43.0 ms), prefill +5-6 %, NLL unchanged |
| 10-01 | **pinned MoE tactics** (`vllm-gb10` + `pin_autotune.py`) | FlashInfer's per-boot autotune picked different tiles boot to boot (a 3.5 % decode swing); one pinned file with both ranks' keys removes that and the warm-cache TP2 deadlock (vllm#57423) |
| 10-01 | **dual-rail NCCL** | +6 % prefill (prefill all-reduces are 42 MB) |
| 10-01 | **FP8 output head** (`fp8_lmhead_view.py`) | step −1.8 ms, NLL unchanged |
| 10-01 | **48K draft vocabulary** (`draft_head_sub.py` + `draft-vocab-ids.json`) | the draft head covers the 49,152 most frequent tokens of a code/English corpus plus all special tokens (99.94 % of held-out generations): step −2.7 ms, acceptance unchanged |
| 10-01 | **16 seats** (from 8) | 233 tok/s aggregate at 16 streams (before RoCEnante) |
| 10-01 | **RoCEnante all-reduce** (overlay `vllm-roce`) | step 39.8 → 38.7 ms, 16 streams +5 % |
| 10-03 | **CYBER-FROST V2 checkpoint** | NLL −0.0011 (chunk-paired over 165K tokens); step unchanged |
| 10-03 | **Marlin W4A16 routed experts** | NLL −0.0063 (165K tokens), step −3.4 %, 8 / 16 streams +4.8 / +4.2 %, prefill −3 to −15 % |
| 10-03 | **MTP drafter quantized** (`mtp_quant_view.py`) | routed experts W4A16 NVFP4 (0.0946 relative L2 error, as in Blackfrost's own export), dense projections FP8: step −4.0 %, 8 streams +6 %, 1.7 GiB less per rank; acceptance unchanged |

## Tried and rejected

| what | result |
|---|---|
| 16,384 batched tokens with Marlin (10-03) | prefill +0.2 to +2.0 %, decode unchanged, but 4.4 GiB less memory headroom at the floor of the bench (14.3 GiB on rank 0) with 16 seats still to fill |
| hyper-connection down/inject projection in FP8 (10-03) | at its shape (N = 336, K = 10,240) FP8 CUTLASS takes 24.4 µs vs 27.5 µs for the BF16 skinny GEMM at decode sizes (it is parallelism-bound, not bandwidth-bound), but 368 vs 246 µs at M = 2048: ~0.3 ms per step gained for ~2.3 % prefill lost (`kernel-tuning/hc_fp8_bench.py`) |
| b12x MoE kernels (10-01) | crashed in CUDA-graph capture (padding rows with expert id −1); with padding disabled, decode −10 % vs CUTLASS and the first long prefill killed the worker |
| 4 draft tokens (10-01) | the 4th position is accepted too rarely: step +12 % for +4 % tokens per step |
| draft top-k / top-p (vllm#56724 port, 10-01) | no change in tokens per step |
| Triton for the MTP layer's BF16 experts (10-01) | same kernel time as FlashInfer CUTLASS (now moot: the experts are NVFP4) |
| vllm#57128 (09-30) | on v0.30 it removes nearly every GDN prefix-cache hit: hot-turn TTFT on a cached ~60K-token prefix 1.1-1.8 s → 16-17 s |
| fastsafetensors (09-30) | the MTP pass stages whole shard files and dipped host memory below 1 GiB on one boot |
| hibrid48's packed NVFP4 n-gram table (09-30) | 13.4 GiB per rank, but NLL +0.0045; FP8 is better by 0.0073 |
| `--moe-backend marlin` before 10-03 | did not boot: the flag also reached the MTP layer's BF16 experts, which Marlin cannot run. Quantizing the drafter (10-03) removes the last unquantized MoE |

## Rollback

- `launcher/profile.rollback-20261001.env`: the 2026-10-01 profile on V2 (FlashInfer CUTLASS W4A4 experts with the
  pinned tactics, BF16 MTP drafter). Build its view with `MTP=0 view/build-view.sh RAW OUT` and point `VIEW` at it.
- Any profile: `launcher/pair-up.sh down`, then `PROFILE=launcher/profile.rollback-20261001.env launcher/pair-up.sh up`.

## Limitations and caveats

- **One pair, one benchmark set.** Every number comes from our two nodes. Expect differences on other kernels,
  drivers, cables or firmware.
- **GB10-specific tuning.** The skinny-GEMM plans, the pinned tactics and the RoCEnante transport were tuned for GB10
  at TP2. On other hardware drop `vllm-gb10` and `vllm-roce` first.
- **Prefill vs decode.** Marlin costs cold prefill (1-10 % in the production profile, up to 15 % at 28K tokens in its
  own A/B) for better NLL and decode; for prefill-bound use (many cold long prompts, short answers)
  `profile.rollback-20261001.env` is faster.
- **Beyond the native context.** 262,144 to 524,288 tokens is YaRN ×2. Our needles retrieve at 500K, but quality at
  that length is less characterised than at native lengths.
- **FP8 KV** uses per-tensor scales of 1.0 (no calibration); we measured no NLL change, but long-context quality was
  only spot-checked with needles.
- **The MTP layer is provisional** (Blackfrost's model card): it predates the final behavioural training. Acceptance
  is ~2.4 tokens per step at T=1 on code review prompts; a refreshed MTP layer would help every runtime.
- **The draft vocabulary is biased to code and English.** Text dominated by other scripts drafts worse and decodes
  slower; outputs are unchanged, because the target verifies every token.
- **Memory.** GB10 memory is unified and an exhausted Spark can hang. The memory guard stops a rank below 4 GiB
  available (or three samples below 6 GiB); it cannot catch an instantaneous allocation.
- **The model.** Use it within Blackfrost's and Qwen's licence terms. This recipe claims nothing about its behaviour
  or capability.

## Reproducing the numbers

```bash
export KEYFILE=/etc/cyberfrost/api-keys
python3 bench/cyber_bench.py out/today nll decode decode8 decode16 prefill     # NLL (29K), 1/8/16-stream decode, cold prefill 7K-175K
python3 bench/nll_big.py out/today/nll-big.json 3                             # NLL over 165K held-out tokens, 3 runs, per chunk
LONG_TOKENS=500000 REFUSE_TOKENS=530000 python3 bench/cyber_qualify.py out/qual
```

Reboot both nodes before each measurement (see Bring-up). The NLL sets are files of the host OS (Python 3.12's
standard library, zlib's and libtool's examples, two kernel-header scripts, the GPL-3 text) as shipped with DGX OS
on Ubuntu 24.04.4 and kernel 6.17.0-1029-nvidia; `nll_big.py` prints a hash of its chunk set (ours:
`52ea55dc9630c92d`). On another OS release the text differs and so do the numbers; compare arms on one host.
The prompts in `bench/` are reworded from the ones we measured with (generic code-review wording, same code samples
and lengths). NLL, prefill and ms per step do not depend on them; decode tokens per second can differ slightly,
because draft acceptance depends on the text. `kernel-tuning/` holds the microbenchmarks behind the
kernel choices (`skinny_tune.py`, `fp8_linear_bench.py`, `hc_fp8_bench.py`), the tactic pinning (`pin_autotune.py`),
the profiler capture and its analysis (`prof_run.py`, `prof_analyze.py`), the draft-vocabulary statistics
(`draft_vocab_stats.py`, `draft_vocab_gen.py`) and the weight-error check against the BF16 source (`werr.py`,
`ratios.py`).

## Layout

```
launcher/       pair-up.sh (owner: checks, start, health), vllm-rank.sh (one rank's docker run), hosts.env.example,
                profile.env (production), profile.rollback-20261001.env, memory-guard.py, fetch-checkpoint.py,
                pinned-autotune-moe-v1.json
overlays/       vllm-upstream, vllm-gb10, vllm-draft, vllm-roce (each: files + SHA256SUMS + image-diffs/),
                PROVENANCE.md, NOTICE
view/           build-view.sh and its steps, draft-vocab-ids.json, VIEW.sha256 (the files of our serving view)
bench/          cyber_bench.py, nll_big.py, cyber_qualify.py, needle_long.py
kernel-tuning/  the microbenchmarks and analysis scripts above
```

## Credits

- **Blackfrost-AI:** [CYBER-FROST-3.8](https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2), and the V2
  re-export within a day of our report ([discussion #1](https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-NVFP4/discussions/1)).
- **Qwen:** Qwen3.8-Flash-Next, the base model.
- **myllmbox:** the Qwen3.8-Flash-Next cluster image and its patches.
- **vLLM:** the engine and the upstream fixes in `overlays/vllm-upstream` (authors in each linked PR).
- **local-inference-lab:** [b12x](https://github.com/local-inference-lab/b12x) and its RoCEnante RDMA collectives.
- **NVIDIA:** FlashInfer and TensorRT Model Optimizer (the NVFP4 format).

## Licence

- **Our files** (launcher, view tools, bench, kernel-tuning, docs): MIT, see `LICENSE`.
- **`overlays/`**: modified vLLM source files and our additions to them, Apache-2.0 (`LICENSES/Apache-2.0.txt`,
  `overlays/NOTICE`). `vllm-roce/…/cyber_roce_ar.py` is adapted from b12x (Apache-2.0).
- **Not included:** the checkpoint (Qwen Community License, via Blackfrost-AI), the image, b12x. Each keeps its own
  terms.
