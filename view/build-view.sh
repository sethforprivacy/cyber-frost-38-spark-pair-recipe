#!/usr/bin/env bash
# Build the serving view from the verified CYBER-FROST-3.8-NVFP4-V2 checkpoint (launcher/fetch-checkpoint.py), on each
# node. A view is a directory of hardlinks to the checkpoint plus the few shards and files that change; the
# checkpoint itself is never modified. The steps are deterministic: both nodes produce identical files.
#
#   build-view.sh RAW OUT        e.g. build-view.sh /srv/models/CYBER-FROST-3.8-NVFP4-V2/67fced33... \
#                                                   /srv/models/CYBER-FROST-3.8-NVFP4-V2-serving
#
# RAW and OUT must be on one filesystem (hardlinks). The tensor steps run in the serving image (CPU only, ~10 min on
# a DGX Spark, 77 GiB of new files beside 42 GiB of hardlinks); the rest is stdlib Python on the host. Steps:
#   1  ple_fp8.py           PLE / n-gram table BF16 -> FP8 E4M3, one global scale (vLLM 0.30's own FP8 PLE path)
#   2  draft_head.py        a draft-only W4A16 NVFP4 copy of the output head (+ the "dh4" quant key)
#   3  view_steps yarn512k  524,288-token context (YaRN x2)
#   4  fp8_dense_view.py    dense projections (GDN, QSA, shared expert, hyper-connection up) FP8 per channel
#   5  fp8_lmhead_view.py   the output head FP8 per channel
#   6  draft_head_sub.py    the draft head cut to the 49,152 ids of draft-vocab-ids.json
#   7  mtp_quant_view.py    the MTP drafter: routed experts W4A16 NVFP4, dense projections FP8 (MTP=0 skips it)
# Intermediate views live in OUT.work and are removed at the end (KEEP_WORK=1 keeps them).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RAW=$(cd "${1:?RAW}" && pwd); OUT=${2:?OUT}
case "$OUT" in /*) ;; *) OUT=$PWD/$OUT;; esac
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 1; }
IMAGE=${VLLM_IMAGE:-myllmbox/qwen38-flash-next-cluster-vllm@sha256:861ac752164e0d723c5eff3f876586c6678c26ad4a516112c48745f6a101ff4d}
W=$OUT.work; mkdir -p "$W"
ROOT=$(python3 -c 'import os,sys; print(os.path.commonpath(sys.argv[1:]))' "$RAW" "$OUT")
[ "$ROOT" != / ] || { echo "RAW and OUT share no directory below /" >&2; exit 1; }
[ "$(stat -c %d "$RAW")" = "$(stat -c %d "$W")" ] || { echo "RAW and OUT must be on one filesystem" >&2; exit 1; }
py() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e "THREADS=${THREADS:-16}" --entrypoint python3 \
         -v "$ROOT:$ROOT" -v "$HERE:$HERE:ro" "$IMAGE" "$@"; }
step() { echo "=== $(date -u +%T) $*"; }

step "1 FP8 PLE";            py "$HERE/ple_fp8.py" "$RAW" "$W/1-plefp8"; python3 "$HERE/view_steps.py" relink "$W/1-plefp8"
step "2 NVFP4 draft head";   python3 "$HERE/view_steps.py" clone "$W/1-plefp8" "$W/2-dh"
                             py "$HERE/draft_head.py" "$W/2-dh"; python3 "$HERE/view_steps.py" dh4 "$W/2-dh"
step "3 YaRN 512K";          python3 "$HERE/view_steps.py" clone "$W/2-dh" "$W/3-512k"
                             python3 "$HERE/view_steps.py" yarn512k "$W/3-512k"
step "4 FP8 dense";          py "$HERE/fp8_dense_view.py" "$W/3-512k" "$W/4-fp8d"
step "5 FP8 output head";    py "$HERE/fp8_lmhead_view.py" "$W/4-fp8d" "$W/5-lmh"
step "6 48K draft vocabulary"
python3 "$HERE/view_steps.py" clone "$W/5-lmh" "$W/6-dv48k"
python3 "$HERE/view_steps.py" vocab "$HERE/draft-vocab-ids.json" "$W/vocab-order.json"
py "$HERE/draft_head_sub.py" "$W/4-fp8d" "$W/6-dv48k" 49152 "$W/vocab-order.json"
if [ "${MTP:-1}" = 1 ]; then
  step "7 MTP drafter: NVFP4 experts, FP8 dense"; py "$HERE/mtp_quant_view.py" "$W/6-dv48k" "$OUT" --mtpq
else
  mv "$W/6-dv48k" "$OUT"
fi
[ "${KEEP_WORK:-0}" = 1 ] || rm -rf -- "$W"
step "done: $OUT ($(ls "$OUT" | wc -l) entries)"
