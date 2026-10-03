#!/usr/bin/env bash
# One vLLM rank of CYBER-FROST-3.8-NVFP4-V2 on a pair of DGX Sparks (GB10), TP=2 over one directly cabled
# ConnectX-7 port (both PCIe halves).
# Image: myllmbox/qwen38-flash-next-cluster-vllm (vLLM 0.30.0 + myllmbox's Qwen3.8-Flash-Next patches), pinned by
# digest; the overlays under overlays/ are mounted read-only over it after a checksum check.
#   vllm-rank.sh <rank> {check|up}      site facts from hosts.env, profile knobs from the environment (profile.env
#                                       via pair-up.sh); rank 0 serves the API. check prints the docker command only.
set -euo pipefail
R=${1:?rank 0..1}; MODE=${2:-check}
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=load-hosts.sh
. "$HERE/load-hosts.sh"
case "$R" in 0|1) ipv=RANK${R}_FABRIC_IP; IP=${!ipv:?$ipv not set (hosts.env)};; *) exit 2;; esac
MASTER_IP=${RANK0_FABRIC_IP:?RANK0_FABRIC_IP not set (hosts.env)}
OVERLAYS=${OVERLAYS:-$(cd "$HERE/.." && pwd)/overlays}
NAME=${CONTAINER:-vllm_cyberfrost}
# Pinned: myllmbox/qwen38-flash-next-cluster-vllm@sha256:861ac752164e0d723c5eff3f876586c6678c26ad4a516112c48745f6a101ff4d,
# whose config digest (the image ID below) is checked before every start. A docker save | docker load relay drops the
# repo digest, so identity is checked by image ID; pull by digest, tag it, set VLLM_IMAGE to the tag if you rename it.
IMAGE=${VLLM_IMAGE:-myllmbox/qwen38-flash-next-cluster-vllm@sha256:861ac752164e0d723c5eff3f876586c6678c26ad4a516112c48745f6a101ff4d}
VLLM_IMAGE_ID=${VLLM_IMAGE_ID:-sha256:4d03fc80b2c28088808115eec524cb36732c8181b13ba15085095fa927b91815}
VIEW=${VIEW:?VIEW not set (hosts.env): the serving view built by view/build-view.sh}
STATE=${STATE:-$HOME/vllm-cyberfrost-state}
API_KEY_FILE=${API_KEY_FILE:?API_KEY_FILE not set (hosts.env): one API key per line}
PORT=${PORT:-8901}
MODEL_NAME=${MODEL_NAME:-CYBER-FROST-3.8-NVFP4}
KV_BYTES=${KV_BYTES:-24000000000}
MAX_LEN=${MAX_LEN:-524288}
MAX_SEQS=${MAX_SEQS:-16}
BATCHED=${BATCHED:-8192}
MOE_BACKEND=${MOE_BACKEND:-marlin}
SPEC=${SPEC:-'{"method": "mtp", "num_speculative_tokens": '"${SPEC_K:-3}"', "rejection_sample_method": "block", "draft_sample_method": "probabilistic"}'}
CC=${VLLM_CC:-'{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 48, 64], "compile_ranges_endpoints": [64]}'}
FABRIC_IFNAME=${FABRIC_IFNAME:-enp1s0f0np0}
# Both PCIe halves of the cabled CX-7 port (DUAL_RAIL=1, production); DUAL_RAIL=0 = the first half only, RoCEnante off.
HCAS=${HCAS:-rocep1s0f0,roceP2p1s0f0}
fail(){ echo "vllm-rank $R: $*" >&2; exit 1; }

# OVERLAY_SETS=dir[,dir]: overlay sets under overlays/ (paths relative to the vllm package + SHA256SUMS), each
# sha-checked, then every listed file mounted read-only over the image's copy. Their files must not overlap.
PATCH_MOUNTS=()
for pd in ${OVERLAY_SETS//,/ }; do
  PD="$OVERLAYS/$pd"; (cd "$PD" && sha256sum -c --quiet SHA256SUMS) || fail "$pd checksum mismatch"
  while read -r _ rel; do PATCH_MOUNTS+=(-v "$PD/$rel:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro"); done < "$PD/SHA256SUMS"
done
# VLLM_ENV=KEY=VALUE[,KEY=VALUE]: extra container environment.
VE="${VLLM_ENV:-}"; ENV_EXTRA=(); for kv in ${VE//,/ }; do ENV_EXTRA+=(-e "$kv"); done
NET_ENV=(-e "NCCL_IB_HCA==$HCAS" -e NCCL_IB_MERGE_NICS=1)
if [ "${DUAL_RAIL:-1}" = 1 ] && [ "${ROCE_AR:-1}" = 1 ]; then
  # B12X=<short sha>: the b12x package of $B12X_ROOT/<sha> (a local-inference-lab/b12x checkout, Apache-2.0),
  # mounted read-only on PYTHONPATH for the RoCEnante all-reduce (overlay set vllm-roce); nothing installed.
  BD="${B12X_ROOT:-$HOME/b12x}/${B12X:?B12X not set (profile.env)}"
  [ "$(git -C "$BD" rev-parse --short=8 HEAD 2>/dev/null)" = "$B12X" ] || fail "b12x checkout $BD is not at $B12X"
  NET_ENV+=(-v "$BD/b12x:/opt/b12x/b12x:ro" -e PYTHONPATH=/opt/b12x -e B12X_ROCE_CACHE_DIR=/root/.cache/b12x-roce
            -e CYBER_ROCE_AR=1 -e "CYBER_ROCE_AR_MAX=${ROCE_AR_MAX:-1048576}")
elif [ "${DUAL_RAIL:-1}" != 1 ]; then
  NET_ENV=(-e "NCCL_IB_HCA==${HCAS%%,*}" -e CYBER_ROCE_AR=0)
fi
[ "$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null)" = "$VLLM_IMAGE_ID" ] || fail "image identity mismatch"
[ -s "$VIEW/model.safetensors.index.json" ] || fail "view missing: $VIEW"
mkdir -p "$STATE/cache"
# PINNED_TACTICS=<file in launcher/>: one FlashInfer MoE tactic file holding both ranks' keys
# (kernel-tuning/pin_autotune.py), installed as the whole tactic cache under the fixed directory the vllm-gb10
# overlay reads (MBX_AUTOTUNE_HASH). No tuning at boot, the same tactics every boot, and no warm-cache TP2 deadlock
# (vllm#57423). Only FlashInfer MoE layers use it (the rollback profile); harmless with Marlin.
if [ -n "${PINNED_TACTICS:-}" ] && [ "$MODE" = up ]; then
  D=/c/vllm/flashinfer_autotune_cache/0.6.18.post1/121a/pinned-moe-v1
  docker run --rm --network none -v "$STATE/cache:/c" -v "$(readlink -f "$HERE/$PINNED_TACTICS"):/pinned.json:ro" \
    --entrypoint sh "$IMAGE" -c "rm -rf /c/vllm/flashinfer_autotune_cache && mkdir -p $D && cp /pinned.json $D/autotune_configs.json" \
    || fail "installing $PINNED_TACTICS"
fi
[ -n "${PINNED_TACTICS:-}" ] && ENV_EXTRA+=(-e MBX_AUTOTUNE_HASH=pinned-moe-v1)
serve=(vllm serve /model --tensor-parallel-size 2 --distributed-executor-backend mp
  --master-addr "$MASTER_IP" --master-port "${MASTER_PORT:-50299}" --nnodes 2 --node-rank "$R"
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "${GMU:-0.70}" --kv-cache-memory-bytes "$KV_BYTES"
  --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCHED" --enable-prefix-caching --kv-cache-dtype fp8
  --mm-encoder-tp-mode data --load-format safetensors --safetensors-load-strategy lazy --dtype bfloat16
  --block-size 1632 --async-scheduling --engram-config '{"cpu_offload": false}' --moe-backend "$MOE_BACKEND"
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
  --served-model-name "$MODEL_NAME" --host 0.0.0.0 --port "$PORT"
  --speculative-config "$SPEC" --compilation-config "$CC")
# VLLM_EXTRA=--flag=value[,--flag=value]: extra vLLM flags (none in the production profile).
VX="${VLLM_EXTRA:-}"; for f in ${VX//,/ }; do serve+=("$f"); done
[ "$R" != 0 ] && serve+=(--headless)
cmd=(docker run -d --name "$NAME" --restart no --label family=cyberfrost --label engine=vllm
  --network host --ipc host --gpus all --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --device /dev/infiniband:/dev/infiniband ${CPUSET:+--cpuset-cpus "$CPUSET"}
  -v "$VIEW:/model:ro" -v "$STATE/cache:/root/.cache" -v "$API_KEY_FILE:/run/secrets/api-keys:ro"
  "${PATCH_MOUNTS[@]}"
  -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_MARLIN_USE_ATOMIC_ADD=1
  -e MBX_PLE_REPLICATE=0 -e MBX_FUSED_DRAFT=1 -e MBX_DRAFT_HEAD=1 -e MBX_DRAFT_VOCAB=/model/draft-vocab-ids.json
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180
  -e "VLLM_HOST_IP=$IP" -e "NCCL_SOCKET_IFNAME=$FABRIC_IFNAME" -e "GLOO_SOCKET_IFNAME=$FABRIC_IFNAME"
  -e NCCL_NET=IB -e NCCL_IB_GID_INDEX="${GID_INDEX:-3}" -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1
  -e NCCL_DEBUG=WARN "${NET_ENV[@]}" "${ENV_EXTRA[@]}"
  --entrypoint bash "$IMAGE" -c 'exec "$@" --api-key $(tr "\n" " " < /run/secrets/api-keys)' _ "${serve[@]}")
idle(){
  [ -z "$(docker ps -q --filter "name=$NAME")" ] || fail "$NAME is already running"
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || fail "GPU has active compute processes"
  awk '/MemAvailable/{exit !($2 > 100*1048576)}' /proc/meminfo || fail "MemAvailable < 100 GiB"
}
case "$MODE" in
 check) printf '%q ' "${cmd[@]}"; printf '\n';;
 up)
  idle; docker rm "$NAME" >/dev/null 2>&1 || true
  nohup python3 "$HERE/memory-guard.py" --state "$STATE" --container "$NAME" >> "$STATE/memory-guard.log" 2>&1 < /dev/null &
  "${cmd[@]}";;
 *) fail "usage: vllm-rank.sh <rank> {check|up}";;
esac
