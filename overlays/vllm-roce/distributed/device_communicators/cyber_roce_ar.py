# SPDX-License-Identifier: Apache-2.0
"""RoCEnante one-shot all-reduce for TP=2 on the B pair (CyberFrost tuning 2026-10-01, CYBER_ROCE_AR=1).

Adapted from the C+D MiMo lane's ring-only recursive doubling (mimo_roce_rd.py in the public MiMo recipe,
github.com/sethforprivacy/mimo-v26-flash-spark-ring-recipe, overlays/vllm-patches-roce; same day).
b12x's RoCEnante (``b12x.comm.roce``, local-inference-lab/b12x @ e4084d2e, Apache-2.0;
mounted read-only on PYTHONPATH, nothing installed) RDMA-writes each rank's input into pinned host slots of the peer
(GB10 unified memory, no GPUDirect), a C proxy thread posts the writes, and one kernel sums in a fixed rank order,
so both ranks get bit-identical outputs. With two ranks on one direct link no recursive doubling is needed: one
2-rank runtime over both PCIe functions of the cabled port. Eligible all-reduces (bf16/fp16/fp32, contiguous,
16-byte multiple, <= CYBER_ROCE_AR_MAX bytes, decided from shape/dtype only, so both ranks route alike) use it in
eager mode and inside CUDA graphs; larger ones stay on NCCL. Fail-stop: a timed-out wait freezes the runtime and
``check_health`` (called from the eager logits all-gather of every step) raises.
"""

import os

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


class CyberRoceAR:
    def __init__(self, cpu_group, device: torch.device, max_size: int) -> None:
        self.disabled = True
        self.rank = dist.get_rank(cpu_group)
        world = dist.get_world_size(cpu_group)
        if world != 2:
            logger.warning("CYBER_ROCE_AR needs a 2-rank TP group (got %d); disabled", world)
            return
        from b12x.comm import roce
        from b12x.comm.roce import _preparation
        from b12x.preparation import PreparationSession, PreparedCall

        hcas = os.environ.get("CYBER_ROCE_HCAS", "rocep1s0f0,roceP2p1s0f0").split(",")
        self.device = device
        self.max_size = int(max_size)
        self.rt = roce.AllReduce(exchange_group=cpu_group, device=device, max_size=self.max_size,
                                 max_gather_bytes=1 << 20, hca_names=hcas)
        query = roce.query_from_runtime(
            self.rt, surface="AllReduce.all_reduce", call={"dtypes": ("float16", "bfloat16", "float32")},
            topology="roce_rdma", peer_hosts=("cyber-ar-0", "cyber-ar-1"))
        self.plan = roce.plan(query, runtime=self.rt)
        seeds = [torch.zeros(16 // dt.itemsize, dtype=dt, device=device)
                 for dt in (torch.float16, torch.bfloat16, torch.float32)]

        def prepare(state):
            calls = [_preparation.prepared_call(state, inp=s) for s in seeds]
            return PreparedCall(run=lambda: [c.run() for c in calls])

        PreparationSession(device=device, autotune=False, compile_workers=2).prepare(
            (self.plan.request(name="cyber-ar", prepare_call=prepare),))
        self.disabled = False
        logger.info("CYBER_ROCE_AR: RoCEnante 2-rank all-reduce active (rank %d, %s, max %d bytes)",
                    self.rank, ",".join(hcas), self.max_size)

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        return not self.disabled and self.rt.should_allreduce(inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(inp)
        self.rt.all_reduce(inp, out=out, plan=self.plan)
        return out

    def check_health(self) -> None:
        if not self.disabled:
            self.rt.check_health()

    def close(self) -> None:
        if not self.disabled:
            self.disabled = True
            self.rt.close()


def maybe_create(unique_name: str, cpu_group, device: torch.device, world_size: int):
    if os.environ.get("CYBER_ROCE_AR", "0") != "1" or unique_name.split(":")[0] != "tp" or world_size != 2:
        return None
    return CyberRoceAR(cpu_group, device, int(os.environ.get("CYBER_ROCE_AR_MAX", str(1 << 20))))
