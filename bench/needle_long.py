#!/usr/bin/env python3
"""Long-context retrieval past the native 262K window (YaRN check). Each prompt plants two passphrases, one early
(~8 % depth) and one late (~92 % depth), in code-like filler; the model must return both.
  python3 needle_long.py <out.json> [targets, default 300000,400000,500000]
"""
import json, os, random, string, sys, time, urllib.request
OUT = sys.argv[1]
TARGETS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "300000,400000,500000").split(",")]
KEY = open(os.path.expanduser(os.environ.get("KEYFILE", "/etc/cyberfrost/api-keys"))).readline().strip()
API = os.environ.get("API", "http://127.0.0.1:8901")
SRC = "".join(open(p, errors="replace").read() for p in ["/usr/lib/python3.12/tarfile.py", "/usr/lib/python3.12/http/client.py",
      "/usr/src/linux-nvidia-6.17-headers-6.17.0-1029/scripts/mod/modpost.c", "/usr/lib/python3.12/subprocess.py",
      "/usr/lib/python3.12/zipfile/__init__.py", "/usr/lib/python3.12/ssl.py", "/usr/lib/python3.12/email/_header_value_parser.py"])
rows = []
for t in TARGETS:
    rnd = random.Random(t)
    chars = int(t * 3.4)
    blob = (SRC * (chars // len(SRC) + 1))
    start = rnd.randrange(0, len(SRC)); blob = blob[start:start + chars]
    a = "".join(rnd.choices(string.ascii_uppercase + string.digits, k=8)); b = "".join(rnd.choices(string.ascii_uppercase + string.digits, k=8))
    i1, i2 = int(len(blob) * 0.08), int(len(blob) * 0.92)
    text = blob[:i1] + f"\n# MARKER-ONE = {a}\n" + blob[i1:i2] + f"\n# MARKER-TWO = {b}\n" + blob[i2:]
    q = "\n\nTwo lines in the code above set MARKER-ONE and MARKER-TWO. Reply with exactly: ONE=<value> TWO=<value>"
    body = {"model": "CYBER-FROST-3.8-NVFP4", "messages": [{"role": "user", "content": text + q}], "temperature": 0,
            "max_tokens": 2048, "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(API + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    try:
        j = json.load(urllib.request.urlopen(req, timeout=3600)); dt = time.time() - t0
        c = j["choices"][0]["message"].get("content") or ""
        r = {"target": t, "prompt_tokens": j["usage"]["prompt_tokens"], "seconds": round(dt, 1), "one_found": a in c,
             "two_found": b in c, "answer": c.strip()[:80]}
    except Exception as e:
        r = {"target": t, "error": f"{type(e).__name__}: {e}"[:300], "seconds": round(time.time() - t0, 1)}
    rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
