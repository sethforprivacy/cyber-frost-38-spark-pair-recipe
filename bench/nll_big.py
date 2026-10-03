#!/usr/bin/env python3
"""Paired held-out NLL for the V1 -> V2 weight swap (stdlib only; run on rank 0 against :8901).

cyber_bench.py's NLL (29K tokens) moves ~+-0.003 run to run on one boot, the size of the effect expected here
(the w13 fold adds ~4 % relative weight error to one half of every expert's w13; V2 removes it). This eval is
~6x larger (165K tokens) and keeps per-chunk results, so two runs can be compared chunk by chunk.

  python3 nll_big.py OUT.json [repeats]          Env: API, MODEL, KEYFILE (as cyber_bench.py)

Chunks: every /usr/lib/python3.12/*.py over 30 KB (sorted), three 3,600-char chunks each, plus the C and
prose sources of cyber_bench.py. Same on every run; the chunk list hash is recorded.
"""
import glob, hashlib, json, os, sys, time, urllib.request

API = os.environ.get("API", "http://127.0.0.1:8901")
MODEL = os.environ.get("MODEL", "CYBER-FROST-3.8-NVFP4")
KEY = open(os.path.expanduser(os.environ.get(
    "KEYFILE", "/etc/cyberfrost/api-keys"))).readline().strip()
OUT, REPEATS = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
CH = 3600

py = sorted(p for p in glob.glob("/usr/lib/python3.12/*.py") if os.path.getsize(p) > 30_000)
extra = ["/usr/share/doc/zlib1g-dev/examples/gun.c", "/usr/share/doc/zlib1g-dev/examples/zran.c",
         "/usr/src/linux-nvidia-6.17-headers-6.17.0-1029/scripts/mod/modpost.c",
         "/usr/src/linux-nvidia-6.17-headers-6.17.0-1029/scripts/unifdef.c",
         "/usr/share/libtool/ltdl.c", "/usr/share/common-licenses/GPL-3"]
chunks = []
for p in py + extra:
    if not os.path.exists(p):   # another OS release: the set hash below will differ
        print(f"{p} missing, skipped", flush=True); continue
    s = open(p, errors="replace").read(); start = min(2000, len(s) // 4)
    for i in range(3 if p in py else 4):
        c = s[start + i * CH: start + (i + 1) * CH]
        if len(c) > 1000:
            chunks.append((f"{p}#{i}", c))
digest = hashlib.sha256("".join(n + c for n, c in chunks).encode()).hexdigest()[:16]
print(f"{len(chunks)} chunks, set {digest}", flush=True)


def nll(text):
    body = {"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    req = urllib.request.Request(API + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    pl = json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["prompt_logprobs"]
    s = n = 0
    for d in pl[1:]:   # prompt_logprobs=0: the prompt token's own entry first (as cyber_bench.py reads it)
        if d:
            s -= next(iter(d.values()))["logprob"]; n += 1
    return s, n


runs = []
for r in range(REPEATS):
    t0 = time.time(); per = []
    for name, text in chunks:
        s, n = nll(text); per.append({"chunk": name, "sum": s, "n": n})
    tot = sum(x["sum"] for x in per) / sum(x["n"] for x in per)
    runs.append({"mean_nll": tot, "tokens": sum(x["n"] for x in per), "secs": round(time.time() - t0, 1),
                 "chunks": per})
    print(f"run {r}: NLL {tot:.5f} over {runs[-1]['tokens']} tokens ({runs[-1]['secs']} s)", flush=True)
json.dump({"set": digest, "model": MODEL, "api": API, "at": time.time(), "runs": runs}, open(OUT, "w"), indent=1)
