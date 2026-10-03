#!/usr/bin/env python3
"""CyberFrost TP2 bench (stdlib only). Run on rank 0 against the server, with nothing else on it.

  python3 cyber_bench.py <outdir> [nll] [decode] [pair] [prefill] [hotturn] [longdecode]   (default: all)

nll        held-out teacher-forced NLL via /v1/completions prompt_logprobs=0 on fixed local texts
           (python stdlib, C sources, GPL-3 prose). Same chunks every arm; the quality gate.
decode     single-stream, thinking ON (template default), published sampling (T=1, top-p .95, top-k 20),
           fixed seeds, code-review prompts; decode tok/s + MTP acceptance from /metrics deltas.
pair       the same prompts two at a time (the lane's seat count today).
prefill    cold prefill (unique nonce) at 8K..200K, max_tokens 1.
hotturn    multi-turn hot turns: ~60K cold context, then 3 turns each appending ~2K tokens of new file content
           (prefix-cache hits); TTFT per turn + decode of 256 tokens.
longdecode decode tok/s at ~120K context (thinking on).
Env: API (default http://127.0.0.1:8901), MODEL, KEYFILE.
"""
import hashlib, json, os, random, string, sys, threading, time, urllib.request

API = os.environ.get("API", "http://127.0.0.1:8901")
MODEL = os.environ.get("MODEL", "CYBER-FROST-3.8-NVFP4")
KEY = open(os.path.expanduser(os.environ.get(
    "KEYFILE", "/etc/cyberfrost/api-keys"))).readline().strip()
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
WHAT = sys.argv[2:] or ["nll", "decode", "pair", "prefill", "hotturn", "longdecode"]
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def post(path, body, timeout=1800):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def metrics():
    try:
        t = urllib.request.urlopen(API + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    want = ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
            "spec_decode_num_accepted_tokens_total", "num_preemptions_total",
            "generation_tokens_total", "prompt_tokens_total", "prefix_cache_hits_total",
            "prefix_cache_queries_total")
    out = {}
    for line in t.splitlines():
        for w in want:
            if line.startswith("vllm:" + w + "{"):
                out[w] = out.get(w, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def delta(a, b):
    return {k: b.get(k, 0) - a.get(k, 0) for k in b}


def spec(d):
    drafts = d.get("spec_decode_num_drafts_total", 0)
    acc = d.get("spec_decode_num_accepted_tokens_total", 0)
    dt = d.get("spec_decode_num_draft_tokens_total", 0)
    return {"tokens_per_step": round(1 + acc / drafts, 3) if drafts else None,
            "accept_rate": round(acc / dt, 3) if dt else None, "drafts": drafts}


def stream_chat(messages, max_tokens, seed=None, extra=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, **SAMPLING}
    if seed is not None:
        body["seed"] = seed
    if extra:
        body.update(extra)
    t0 = time.time(); first = None; last = None; usage = None; text = []; reasoning = []; finish = None
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.time()
                    first = first or now
                    last = now
                    (reasoning if (d.get("reasoning_content") or d.get("reasoning")) else text).append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    ct = (usage or {}).get("completion_tokens", 0)
    pt = (usage or {}).get("prompt_tokens", 0)
    dec = (ct - 1) / (last - first) if first and last and last > first and ct > 1 else None
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft": round(first - t0, 3) if first else None,
            "wall": round(time.time() - t0, 2), "decode_tok_s": round(dec, 2) if dec else None,
            "finish": finish, "content_chars": len("".join(text)), "reasoning_chars": len("".join(reasoning))}


# ---------------------------------------------------------------- nll
NLL_SOURCES = [
    ("/usr/lib/python3.12/tarfile.py", 3), ("/usr/lib/python3.12/zipfile/__init__.py", 2),
    ("/usr/lib/python3.12/http/client.py", 2), ("/usr/lib/python3.12/ssl.py", 2),
    ("/usr/lib/python3.12/subprocess.py", 2), ("/usr/lib/python3.12/urllib/parse.py", 2),
    ("/usr/lib/python3.12/email/_header_value_parser.py", 2),
    ("/usr/share/doc/zlib1g-dev/examples/gun.c", 2), ("/usr/share/doc/zlib1g-dev/examples/zran.c", 2),
    ("/usr/src/linux-nvidia-6.17-headers-6.17.0-1029/scripts/mod/modpost.c", 3),
    ("/usr/src/linux-nvidia-6.17-headers-6.17.0-1029/scripts/unifdef.c", 2),
    ("/usr/share/libtool/ltdl.c", 2), ("/usr/share/common-licenses/GPL-3", 4),
]
CHUNK_CHARS = 3600


def nll_chunks():
    chunks = []
    for path, n in NLL_SOURCES:
        if not os.path.exists(path):   # another OS release: the set (and its numbers) differ
            print(f"nll: {path} missing, skipped", flush=True); continue
        s = open(path, errors="replace").read()
        start = min(2000, len(s) // 4)
        for i in range(n):
            c = s[start + i * CHUNK_CHARS: start + (i + 1) * CHUNK_CHARS]
            if len(c) > 1000:
                chunks.append((f"{os.path.basename(path)}#{i}", c))
    return chunks


def run_nll():
    res = []; tot_lp = 0.0; tot_n = 0; top1 = 0
    for name, text in nll_chunks():
        body = {"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
        j = json.load(post("/v1/completions", body, timeout=600))
        pl = j["choices"][0]["prompt_logprobs"]
        lps = []; r1 = 0
        for entry in pl[1:]:
            if not entry:
                continue
            (tok, info), = [next(iter(entry.items()))]
            lps.append(info["logprob"]); r1 += info.get("rank") == 1
        n = len(lps); s = -sum(lps)
        res.append({"chunk": name, "sha256": hashlib.sha256(text.encode()).hexdigest()[:16], "tokens": n,
                    "nll": round(s / n, 5), "top1": round(r1 / n, 4)})
        tot_lp += s; tot_n += n; top1 += r1
        print(f"nll {name:34s} tok {n:5d} nll {s/n:.4f} top1 {r1/n:.3f}", flush=True)
    summ = {"tokens": tot_n, "mean_nll": round(tot_lp / tot_n, 5), "top1": round(top1 / tot_n, 4)}
    print("NLL", summ, flush=True)
    json.dump({"summary": summ, "chunks": res}, open(f"{OUT}/nll.json", "w"), indent=1)


# ---------------------------------------------------------------- decode prompts
SNIPPETS = [
    ("C", """static int parse_record(const char *buf, size_t len, struct rec *out) {
    char name[64];
    uint16_t nlen;
    if (len < 2) return -1;
    memcpy(&nlen, buf, 2);
    memcpy(name, buf + 2, nlen);
    name[nlen] = 0;
    out->id = atoi(name);
    strcpy(out->label, buf + 2 + nlen);
    return 0;
}"""),
    ("Python", """@app.route('/download')
def download():
    fn = request.args.get('file', '')
    path = os.path.join(BASE_DIR, fn)
    if not os.path.exists(path):
        abort(404)
    with open(path, 'rb') as f:
        data = f.read()
    log.info('served %s to %s' % (fn, request.remote_addr))
    return Response(data, mimetype=mimetypes.guess_type(path)[0])"""),
    ("Go", """func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
    user := r.FormValue("user"); pass := r.FormValue("pass")
    q := fmt.Sprintf("SELECT id, hash FROM users WHERE name = '%s'", user)
    row := s.db.QueryRow(q)
    var id int; var hash string
    if err := row.Scan(&id, &hash); err != nil { http.Error(w, err.Error(), 500); return }
    if hash == md5hex(pass) { setSession(w, id) }
}"""),
    ("Rust", """pub fn read_frame(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut hdr = [0u8; 4];
    stream.read_exact(&mut hdr)?;
    let n = u32::from_be_bytes(hdr) as usize;
    let mut buf = Vec::with_capacity(n);
    unsafe { buf.set_len(n); }
    stream.read_exact(&mut buf)?;
    Ok(buf)
}"""),
    ("Dart", """Future<void> importWallet(String backupJson) async {
  final data = jsonDecode(backupJson);
  final seed = data['seed'] as String;
  await secureStorage.write(key: 'seed', value: seed);
  debugPrint('imported wallet seed: $seed');
  final url = Uri.parse('http://' + data['node'] + '/json_rpc');
  await http.post(url, body: jsonEncode({'method': 'restore', 'params': {'seed': seed}}));
}"""),
    ("JavaScript", """app.post('/api/profile', async (req, res) => {
  const profile = Object.assign({}, defaults, req.body);
  if (profile.isAdmin && !req.user.isAdmin) delete profile.isAdmin;
  await db.collection('users').updateOne({ _id: req.user.id }, { $set: profile });
  res.send(`<h1>Saved ${profile.displayName}</h1>`);
});"""),
]


def review_prompt(lang, code):
    return [{"role": "system", "content": "You are a senior software engineer reviewing code. Be precise."},
            {"role": "user", "content": f"Review this {lang} code for bugs and robustness problems. For each issue give the "
             f"exact line, the impact, and a fixed version.\n\n```{lang.lower()}\n{code}\n```"}]


def run_decode(tag="decode", conc=1, max_tokens=1200, rounds=1):
    rows = []
    jobs = [(i, lang, code) for r in range(rounds) for i, (lang, code) in enumerate(SNIPPETS)]
    m0 = metrics(); t0 = time.time()
    for k in range(0, len(jobs), conc):
        batch = jobs[k:k + conc]; out = [None] * len(batch); mb = metrics()
        def go(j, i, lang, code):
            out[j] = stream_chat(review_prompt(lang, code), max_tokens, seed=1000 + i)
        th = [threading.Thread(target=go, args=(j, *b)) for j, b in enumerate(batch)]
        tb = time.time()
        for t in th: t.start()
        for t in th: t.join()
        wall = time.time() - tb; me = metrics(); sp = spec(delta(mb, me))
        agg = sum(o["completion_tokens"] for o in out) / wall
        for (i, lang, _), o in zip(batch, out):
            o.update(prompt=lang, **{"batch_" + k2: v for k2, v in sp.items()})
            rows.append(o)
        print(f"{tag} c{conc} {','.join(b[1] for b in batch):16s} " + " | ".join(
            f"{o['completion_tokens']}tok {o['decode_tok_s']}t/s ttft {o['ttft']}" for o in out) +
            f" | agg {agg:.1f} tok/s  tok/step {sp['tokens_per_step']}", flush=True)
    m1 = metrics(); tot = spec(delta(m0, m1)); wall = time.time() - t0
    gen = sum(r["completion_tokens"] for r in rows)
    dec = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
    summ = {"conc": conc, "requests": len(rows), "gen_tokens": gen, "wall": round(wall, 1),
            "agg_tok_s": round(gen / wall, 2), "median_stream_decode_tok_s": sorted(dec)[len(dec) // 2] if dec else None,
            "mean_stream_decode_tok_s": round(sum(dec) / len(dec), 2) if dec else None, **tot,
            "preemptions": delta(m0, m1).get("num_preemptions_total")}
    print(tag.upper(), summ, flush=True)
    json.dump({"summary": summ, "rows": rows}, open(f"{OUT}/{tag}.json", "w"), indent=1)


# ---------------------------------------------------------------- prefill
WORDS = ("the quick brown fox jumps over the lazy dog while distributed systems exchange heartbeats across "
         "lossy links and schedulers weigh timeouts against false positives").split()


def filler(n_words, seed):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n_words))


def run_prefill(targets=(8000, 32000, 64000, 128000, 200000)):
    rows = []
    for t in targets:
        nonce = "".join(random.choices(string.ascii_lowercase, k=12))
        text = f"nonce {nonce}. " + filler(int(t * 0.75), t)
        body = {"model": MODEL, "prompt": text + "\nReply OK.", "max_tokens": 1, "temperature": 0}
        t0 = time.time(); j = json.load(post("/v1/completions", body)); dt = time.time() - t0
        pt = j["usage"]["prompt_tokens"]
        rows.append({"prompt_tokens": pt, "seconds": round(dt, 2), "prefill_tok_s": round(pt / dt)})
        print(f"prefill {pt:>7} tok {dt:7.2f} s {pt/dt:8.0f} tok/s", flush=True)
    json.dump(rows, open(f"{OUT}/prefill.json", "w"), indent=1)


# ---------------------------------------------------------------- hot turns
def code_blob(n_chars, seed):
    srcs = ["/usr/lib/python3.12/tarfile.py", "/usr/lib/python3.12/http/client.py",
            "/usr/src/linux-nvidia-6.17-headers-6.17.0-1029/scripts/mod/modpost.c",
            "/usr/lib/python3.12/subprocess.py", "/usr/lib/python3.12/zipfile/__init__.py",
            "/usr/lib/python3.12/ssl.py", "/usr/lib/python3.12/email/_header_value_parser.py"]
    s = "".join(open(p, errors="replace").read() for p in srcs)
    rnd = random.Random(seed); start = rnd.randrange(0, max(1, len(s) - n_chars))
    return s[start:start + n_chars]


def run_hotturn(ctx_chars=200000, turns=3, add_chars=7000):
    nonce = "".join(random.choices(string.ascii_lowercase, k=12))
    msgs = [{"role": "system", "content": f"Session {nonce}. You are a senior engineer working through a repository "
             "with tools. Keep answers short."},
            {"role": "user", "content": "Here is the first file:\n```\n" + code_blob(ctx_chars, 7) +
             "\n```\nList the three most complex functions in one line each."}]
    rows = []
    for t in range(turns + 1):
        r = stream_chat(msgs, 256, seed=77 + t)
        r["turn"] = t; rows.append(r)
        print(f"hot turn {t}: prompt {r['prompt_tokens']} ttft {r['ttft']} s decode {r['decode_tok_s']} t/s", flush=True)
        msgs.append({"role": "assistant", "content": "Noted; continuing the review."})
        msgs.append({"role": "user", "content": f"Tool output {t}:\n```\n" + code_blob(add_chars, 100 + t) +
                     "\n```\nAny new issue? One line."})
    json.dump(rows, open(f"{OUT}/hotturn.json", "w"), indent=1)


def run_longdecode(ctx_chars=420000):
    nonce = "".join(random.choices(string.ascii_lowercase, k=12))
    msgs = [{"role": "user", "content": f"Session {nonce}.\n```\n" + code_blob(ctx_chars, 11) +
             "\n```\nReview the code above: explain in detail the most complex control-flow path you can find."}]
    m0 = metrics(); r = stream_chat(msgs, 800, seed=5); sp = spec(delta(m0, metrics()))
    r.update(sp)
    print(f"longdecode prompt {r['prompt_tokens']} ttft {r['ttft']} decode {r['decode_tok_s']} t/s tok/step {sp['tokens_per_step']}", flush=True)
    json.dump(r, open(f"{OUT}/longdecode.json", "w"), indent=1)


def run_multiturn(n=24, conc=4, p_chars=5200, gen1=700, gen2=160):
    """Prefix-cache + MTP corruption probe (vllm#53912/#57128). Turn 1 samples ~700 tokens with MTP from a
    ~1.4K-token prompt, so its answer crosses the 1632-token state-checkpoint boundary. Turn 2 re-sends the exact
    token ids (prompt + generated + question): it reuses the GDN snapshot taken inside the MTP-generated span.
    Greedy turn 2 on the cached path vs the same ids with a fresh cache_salt (cold recompute): identical or
    long-common-prefix = healthy; garbage/empty/early divergence on the cached path = corrupted snapshot."""
    def tok(text):
        j = json.load(post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False}))
        return j["tokens"]
    def comp(prompt, max_tokens, temp, salt=None, seed=None):
        body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": temp,
                "return_token_ids": True}
        if temp:
            body.update(top_p=0.95, top_k=20)
        if salt:
            body["cache_salt"] = salt
        if seed is not None:
            body["seed"] = seed
        t0 = time.time(); j = json.load(post("/v1/completions", body)); dt = time.time() - t0
        c = j["choices"][0]
        return c.get("token_ids") or [], c["text"], j.get("prompt_token_ids") or c.get("prompt_token_ids"), dt
    q_ids = tok("\n\nNow, in one short paragraph, state the single most important issue above and how to fix it:\n")
    rows = [None] * n
    def conv(i):
        nonce = "".join(random.Random(900 + i).choices(string.ascii_lowercase, k=10))
        p = f"[{nonce}] Review notes for the following source file.\n" + code_blob(p_chars, 300 + i) + \
            "\n\nReview, finding by finding:\n"
        a_ids, a_txt, p_ids, _ = comp(p, gen1, 1.0, seed=i)
        if not p_ids:
            p_ids = tok(p)
        ids2 = list(p_ids) + list(a_ids) + q_ids
        b_hit, t_hit, _, dt_hit = comp(ids2, gen2, 0.0)
        b_ref, t_ref, _, dt_ref = comp(ids2, gen2, 0.0, salt="ref-" + nonce)
        common = 0
        for x, y in zip(b_hit, b_ref):
            if x != y:
                break
            common += 1
        def garbage(t):
            if not t.strip():
                return True
            run = best = 1
            for a, b in zip(t, t[1:]):
                run = run + 1 if a == b and not a.isspace() else 1
                best = max(best, run)
            return best >= 16
        rows[i] = {"i": i, "prompt_tokens": len(p_ids), "gen1": len(a_ids), "turn2_prompt": len(ids2),
                   "common_prefix": common, "len_hit": len(b_hit), "len_ref": len(b_ref), "identical": b_hit == b_ref,
                   "garbage_hit": garbage(t_hit), "garbage_ref": garbage(t_ref),
                   "ttft_proxy_hit_s": round(dt_hit, 2), "ttft_proxy_ref_s": round(dt_ref, 2),
                   "hit_head": t_hit[:80], "ref_head": t_ref[:80]}
    for k in range(0, n, conc):
        th = [threading.Thread(target=conv, args=(i,)) for i in range(k, min(n, k + conc))]
        for t in th: t.start()
        for t in th: t.join()
        for r in rows[k:k + conc]:
            print(f"mt {r['i']:2d} p{r['prompt_tokens']} g{r['gen1']} -> common {r['common_prefix']:3d}/{r['len_ref']} "
                  f"ident {int(r['identical'])} garbage hit/ref {int(r['garbage_hit'])}/{int(r['garbage_ref'])} "
                  f"t hit/ref {r['ttft_proxy_hit_s']}/{r['ttft_proxy_ref_s']}s", flush=True)
    ok = [r for r in rows if r]
    summ = {"conversations": len(ok), "identical": sum(r["identical"] for r in ok),
            "garbage_hit": sum(r["garbage_hit"] for r in ok), "garbage_ref": sum(r["garbage_ref"] for r in ok),
            "early_divergence_lt16": sum(r["common_prefix"] < 16 for r in ok),
            "median_common_prefix": sorted(r["common_prefix"] for r in ok)[len(ok) // 2]}
    print("MULTITURN", summ, flush=True)
    json.dump({"summary": summ, "rows": ok}, open(f"{OUT}/multiturn.json", "w"), indent=1)


if __name__ == "__main__":
    t0 = time.time()
    for w in WHAT:
        print(f"== {w} {time.strftime('%H:%M:%S', time.gmtime())}Z", flush=True)
        if w == "nll": run_nll()
        elif w == "decode": run_decode("decode", 1)
        elif w == "pair": run_decode("pair", 2)
        elif w == "prefill": run_prefill()
        elif w == "hotturn": run_hotturn()
        elif w == "longdecode": run_longdecode()
        elif w == "multiturn": run_multiturn()
        elif w == "decode8": run_decode("decode8", 8, rounds=2)
        elif w == "decode16": run_decode("decode16", 16, rounds=4)
    print(f"BENCH_DONE {time.time()-t0:.0f}s", flush=True)
