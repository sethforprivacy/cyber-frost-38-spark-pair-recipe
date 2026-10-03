#!/usr/bin/env python3
"""Collect CyberFrost generations for draft-vocabulary statistics: the model explains 48 excerpts of the CPython
standard library (thinking on, published sampling, 8 requests at a time). Writes one JSON line per generation.

  python3 draft_vocab_gen.py <out.jsonl>      (rank 0, against the server on :8901)
"""
import glob, json, os, random, sys, threading, urllib.request

API = "http://127.0.0.1:8901"; MODEL = "CYBER-FROST-3.8-NVFP4"
KEY = open(os.path.expanduser(os.environ.get("KEYFILE", "/etc/cyberfrost/api-keys"))).readline().strip()
OUT = sys.argv[1]
files = sorted(glob.glob("/usr/lib/python3.12/*.py"))
rnd = random.Random(20261001)
picks = rnd.sample(files, 48)


def excerpt(path):
    s = open(path, encoding="utf-8", errors="ignore").read()
    if len(s) <= 3000:
        return s
    i = rnd.randrange(0, len(s) - 3000)
    return s[i:i + 3000]


jobs = [(os.path.basename(p), excerpt(p)) for p in picks]
out = open(OUT, "w"); lock = threading.Lock()


def one(name, code):
    body = {"model": MODEL, "max_tokens": 1200, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "messages": [{"role": "user", "content": f"Explain what this code from Python's {name} does, step by "
                          f"step, and suggest one improvement.\n\n```python\n{code}\n```"}]}
    req = urllib.request.Request(API + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]
    text = (m.get("reasoning_content") or m.get("reasoning") or "") + "\n" + (m.get("content") or "")
    with lock:
        out.write(json.dumps({"file": name, "text": text}) + "\n"); out.flush()


for k in range(0, len(jobs), 8):
    th = [threading.Thread(target=one, args=j) for j in jobs[k:k + 8]]
    [t.start() for t in th]; [t.join() for t in th]
    print(f"{min(k + 8, len(jobs))}/{len(jobs)}", flush=True)
print("GEN_DONE", flush=True)
