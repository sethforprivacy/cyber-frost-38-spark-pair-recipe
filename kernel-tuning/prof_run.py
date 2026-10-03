#!/usr/bin/env python3
"""Capture torch-profiler traces of the three request shapes on a server booted with --profiler-config (rank 0).

  python3 prof_run.py <outdir> [decode1] [decode8] [prefill]      (default: all three)

Each capture: POST /start_profile, run the requests, POST /stop_profile, wait for the trace files to settle, and
record which new trace files appeared on each rank (the profiler dir is root-owned, so files are listed, not moved).
decode1   one code-review request, thinking on, published sampling, 200 tokens
decode8   eight such requests at once, 120 tokens each
prefill   one 20K-token unique prompt, max_tokens 1 (three 8,192-token chunks)
Writes <outdir>/captures.json with per-capture wall time, token counts, MTP deltas and the trace file names.
"""
import json, os, random, subprocess, sys, threading, time, urllib.request

API = os.environ.get("API", "http://127.0.0.1:8901")
MODEL = "CYBER-FROST-3.8-NVFP4"
KEY = open(os.path.expanduser(os.environ.get("KEYFILE", "/etc/cyberfrost/api-keys"))).readline().strip()
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
WHAT = sys.argv[2:] or ["decode1", "decode8", "prefill"]
PDIR = os.path.expanduser(os.environ.get("PROF_HOST_DIR", "~/vllm-cyberfrost-state/cache/prof"))
WORKER = os.environ["RANK1_SSH"]   # ssh target of rank 1 (its traces are copied back)
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
CODE = """def download():
    fn = request.args.get('file', '')
    path = os.path.join(BASE_DIR, fn)
    if not os.path.exists(path):
        abort(404)
    with open(path, 'rb') as f:
        data = f.read()
    log.info('served %s to %s' % (fn, request.remote_addr))
    return Response(data, mimetype=mimetypes.guess_type(path)[0])"""


def post(path, body=None, timeout=1800):
    req = urllib.request.Request(API + path, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def metrics():
    t = urllib.request.urlopen(API + "/metrics", timeout=10).read().decode(); out = {}
    for line in t.splitlines():
        for k in ("spec_decode_num_drafts_total", "spec_decode_num_accepted_tokens_total",
                  "spec_decode_num_draft_tokens_total", "generation_tokens_total", "prompt_tokens_total"):
            if line.startswith("vllm:" + k + "{"):
                out[k] = out.get(k, 0) + float(line.rsplit(" ", 1)[1])
    return out


def chat(i, max_tokens, prompt=None):
    msgs = [{"role": "system", "content": "You are a senior software engineer reviewing code. Be precise."},
            {"role": "user", "content": prompt or f"Review this Python code for bugs and robustness problems (variant {i}). "
             f"For each issue give the exact line, the impact, and a fixed version.\n\n```python\n{CODE}\n```"}]
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "seed": 1000 + i, **SAMPLING}
    return json.loads(post("/v1/chat/completions", body))["usage"]


def listing():
    loc = sorted(os.listdir(PDIR)) if os.path.isdir(PDIR) else []
    rem = subprocess.run(["ssh", "-o", "BatchMode=yes", WORKER, f"ls {PDIR} 2>/dev/null"], capture_output=True,
                         text=True).stdout.split()
    return set(loc), set(rem)


def capture(name, work):
    l0, r0 = listing(); m0 = metrics()
    post("/start_profile"); t0 = time.time()
    usage = work()
    wall = time.time() - t0
    post("/stop_profile")
    for _ in range(60):              # wait until both ranks' new files stop growing
        time.sleep(5)
        l1, r1 = listing()
        if l1 - l0 and r1 - r0:
            time.sleep(10); break
    l1, r1 = listing(); m1 = metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    rec = {"capture": name, "wall_s": round(wall, 2), "usage": usage, "metrics_delta": d,
           "head_files": sorted(l1 - l0), "worker_files": sorted(r1 - r0)}
    print(json.dumps(rec), flush=True)
    return rec


def decode1():
    return chat(0, 200)


def decode8():
    out = [None] * 8
    th = [threading.Thread(target=lambda j=j: out.__setitem__(j, chat(j, 120))) for j in range(8)]
    [t.start() for t in th]; [t.join() for t in th]
    return out


def prefill():
    rnd = random.Random(time.time())
    words = "array kernel buffer socket parser token stream frame alpha beta gamma delta".split()
    text = f"nonce {rnd.random()}\n" + " ".join(rnd.choice(words) for _ in range(20000))
    return chat(0, 1, prompt=text + "\n\nSummarize in one word.")


for w in range(2):                  # warm the decode path outside any capture
    chat(100 + w, 48)
caps = [capture(n, {"decode1": decode1, "decode8": decode8, "prefill": prefill}[n]) for n in WHAT]
json.dump(caps, open(os.path.join(OUT, "captures.json"), "w"), indent=1)
print("PROF_DONE", flush=True)
