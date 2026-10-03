#!/usr/bin/env python3
"""Qualification gates for a CyberFrost profile, run on rank 0 (credentials never printed or written).

Gates: every key in KEYFILE answers READY (thinking is template-forced; low effort); a retired key (RETIRED_KEYFILE,
optional), a missing key and an invalid key get 401; vision (red square); forced tool call + continuation; a
LONG_TOKENS-token needle retrieval (default 255,000; 500,000 at the 524,288 context); a REFUSE_TOKENS prompt
(default 270,000; 530,000 at 524,288) -> 400 and the server stays healthy; 8 concurrent short requests; strict JSON
3/3. Receipts -> <out>/qualification.json.
  KEYFILE=/etc/cyberfrost/api-keys LONG_TOKENS=500000 REFUSE_TOKENS=530000 \
    python3 cyber_qualify.py <out_dir> [--base http://127.0.0.1:8901/v1] [--skip-long]
"""
import base64, json, random, string, struct, sys, threading, time, urllib.error, urllib.request, zlib
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
BASE = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8901/v1"
SKIP_LONG = "--skip-long" in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
import os
KEYS = Path(os.environ.get("KEYFILE", "/etc/cyberfrost/api-keys")).read_text().split()
# RETIRED_KEYFILE (optional): keys that must now be refused (one per line)
OLD = (Path(os.environ["RETIRED_KEYFILE"]).read_text().split() or [None])[0] if os.environ.get("RETIRED_KEYFILE") else None
MODEL = "CYBER-FROST-3.8-NVFP4"
assert KEYS and len(set(KEYS)) == len(KEYS) and OLD not in KEYS
res = {"base": BASE, "started": time.time(), "gates": {}}


def chat(key, body, timeout=900):
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps({"model": MODEL, **body}).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.load(f)


def gate(name, fn):
    t0 = time.time()
    try:
        info = fn() or {}
        res["gates"][name] = {"pass": True, "seconds": round(time.time() - t0, 2), **info}
    except Exception as e:  # record, keep going
        res["gates"][name] = {"pass": False, "seconds": round(time.time() - t0, 2), "error": f"{type(e).__name__}: {e}"[:400]}
    print(name, json.dumps(res["gates"][name])[:300], flush=True)


LOW = {"chat_template_kwargs": {"reasoning_effort": "low"}}


def keys():
    out = []
    for i, k in enumerate(KEYS):
        d = chat(k, {"messages": [{"role": "user", "content": "Reply exactly READY."}], "temperature": 0, "max_tokens": 512, **LOW})
        c = d["choices"][0]["message"].get("content") or ""
        assert d["model"] == MODEL and "READY" in c, c[:80]
        out.append(i + 1)
    return {"keys_ok": len(out)}


def rejected():
    codes = []
    for k in ([OLD] if OLD else []) + [None, "sk-invalid-" + "x" * 20]:
        try:
            chat(k, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
            codes.append(200)
        except urllib.error.HTTPError as e:
            codes.append(e.code)
    assert codes and all(c == 401 for c in codes), codes
    return {"codes": codes}


def png():
    def ch(t, d): return struct.pack("!I", len(d)) + t + d + struct.pack("!I", zlib.crc32(t + d))
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack("!2I5B", 64, 64, 8, 2, 0, 0, 0)) + \
        ch(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 64) * 64)) + ch(b"IEND", b"")


def vision():
    url = "data:image/png;base64," + base64.b64encode(png()).decode()
    d = chat(KEYS[0], {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
             {"type": "text", "text": "What single color fills this image? Answer with one word."}]}], "temperature": 0, "max_tokens": 1024, **LOW})
    c = (d["choices"][0]["message"].get("content") or "").lower()
    assert "red" in c, c[:80]
    return {"answer": c.strip()[:40]}


def tools():
    tool = {"type": "function", "function": {"name": "get_status", "description": "Read target status.",
            "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}}}
    msgs = [{"role": "user", "content": "Check the status of target alpha with the tool, then tell me the status."}]
    d = chat(KEYS[0], {"messages": msgs, "tools": [tool], "tool_choice": "auto", "temperature": 0, "max_tokens": 2048, **LOW})
    m = d["choices"][0]["message"]; calls = m.get("tool_calls") or []
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_status", m
    msgs += [{"role": "assistant", "content": m.get("content") or "", "tool_calls": calls},
             {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"target": "alpha", "status": "DEGRADED-7731"})}]
    d = chat(KEYS[0], {"messages": msgs, "tools": [tool], "temperature": 0, "max_tokens": 2048, **LOW})
    c = d["choices"][0]["message"].get("content") or ""
    assert "7731" in c or "DEGRADED" in c.upper(), c[:120]
    return {"calls": 1, "answer": c.strip()[:60]}


def filler(n):
    rnd = random.Random(n); w = "alpha beta gamma delta array kernel buffer socket parser token stream frame".split()
    return " ".join(rnd.choice(w) for _ in range(n))


def long_ctx():
    needle = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    words = int(__import__("os").environ.get("LONG_TOKENS", "255000"))   # this filler is ~1 token/word
    text = filler(words // 2) + f"\nThe passphrase is {needle}.\n" + filler(words // 2)
    t0 = time.time()
    d = chat(KEYS[0], {"messages": [{"role": "user", "content": text + "\n\nWhat is the passphrase? Reply with it only."}],
                       "temperature": 0, "max_tokens": 1024, **LOW}, timeout=3600)
    c = d["choices"][0]["message"].get("content") or ""
    assert needle in c, c[:80]
    return {"prompt_tokens": d["usage"]["prompt_tokens"], "wall_s": round(time.time() - t0, 1)}


def refuse():
    text = filler(int(__import__("os").environ.get("REFUSE_TOKENS", "270000")))
    try:
        chat(KEYS[0], {"messages": [{"role": "user", "content": text}], "max_tokens": 8})
        raise AssertionError("over-limit prompt accepted")
    except urllib.error.HTTPError as e:
        assert e.code == 400, e.code
    with urllib.request.urlopen(BASE.rsplit("/v1", 1)[0] + "/health", timeout=10) as f:
        assert f.status == 200
    return {"status": 400, "health_after": 200}


def conc():
    out = [None] * 8
    def one(i):
        try:
            d = chat(KEYS[i % 7], {"messages": [{"role": "user", "content": f"Reply exactly READY {i}."}], "temperature": 0, "max_tokens": 512, **LOW})
            out[i] = "READY" in (d["choices"][0]["message"].get("content") or "")
        except Exception:
            out[i] = False
    th = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    [t.start() for t in th]; [t.join() for t in th]
    assert all(out), out
    return {"ok": sum(out)}


def strict_json():
    ok = 0
    for i in range(3):
        d = chat(KEYS[0], {"messages": [{"role": "user", "content": f"Return a JSON object with keys id (the integer {i}) and ok (true). JSON only."}],
                           "temperature": 0, "max_tokens": 1024, "response_format": {"type": "json_object"}, **LOW})
        j = json.loads(d["choices"][0]["message"]["content"]); ok += j.get("id") == i and j.get("ok") is True
    assert ok == 3, ok
    return {"ok": ok}


for n, f in [("keys", keys), ("rejected", rejected), ("vision", vision), ("tools", tools), ("concurrency8", conc),
             ("strict_json", strict_json)] + ([] if SKIP_LONG else [("long_context", long_ctx), ("refuse_over_limit", refuse)]):
    if ONLY is None or n in ONLY:
        gate(n, f)
res["finished"] = time.time(); res["all_pass"] = all(g["pass"] for g in res["gates"].values())
(OUT / "qualification.json").write_text(json.dumps(res, indent=1) + "\n")
print("ALL_PASS" if res["all_pass"] else "SOME_FAILED", flush=True)
