#!/usr/bin/env python3
"""Token frequencies for a reduced draft vocabulary (FR-Spec style) for the CyberFrost MTP draft head.

Counts tokenizer ids over (1) a local code/English corpus: CPython stdlib, the vLLM / torch / transformers Python
sources and their bundled docs, C/C++ headers; and (2) the model's own outputs, collected from the serving lane
(code-review prompts with thinking on). Writes <out>/vocab_stats.json with the ids
sorted by count and the coverage of the top-K sets on the held-out model outputs.

  docker exec -i <container> python3 - <out_dir> [<generations.jsonl>] < draft_vocab_stats.py
"""
import collections, glob, json, os, sys

from tokenizers import Tokenizer

OUT = sys.argv[1]; GEN = sys.argv[2] if len(sys.argv) > 2 else None
os.makedirs(OUT, exist_ok=True)
tok = Tokenizer.from_file("/model/tokenizer.json")
vocab = tok.get_vocab_size(with_added_tokens=True)
cnt = collections.Counter(); files = 0; nbytes = 0
pats = ["/usr/lib/python3.12/**/*.py", "/usr/local/lib/python3.12/dist-packages/vllm/**/*.py",
        "/usr/local/lib/python3.12/dist-packages/transformers/**/*.py",
        "/usr/local/lib/python3.12/dist-packages/torch/**/*.py",
        "/usr/local/lib/python3.12/dist-packages/torch/include/**/*.h",
        "/usr/local/lib/python3.12/dist-packages/**/*.md", "/usr/local/lib/python3.12/dist-packages/**/*.rst",
        "/usr/share/doc/**/*.txt", "/usr/include/**/*.h"]
for pat in pats:
    for f in glob.glob(pat, recursive=True):
        try:
            s = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if not s or len(s) > 2_000_000:
            continue
        cnt.update(tok.encode(s, add_special_tokens=False).ids); files += 1; nbytes += len(s)
        if nbytes > 400_000_000:
            break
gen_ids = []
if GEN and os.path.exists(GEN):
    for line in open(GEN):
        t = json.loads(line)["text"]
        gen_ids.append(tok.encode(t, add_special_tokens=False).ids)
held = [ids for i, ids in enumerate(gen_ids) if i % 4 == 0]      # every 4th generation is held out
for i, ids in enumerate(gen_ids):
    if i % 4:
        cnt.update(ids * 20)                                          # weight the lane's own outputs
order = [t for t, _ in cnt.most_common()]
held_n = sum(len(x) for x in held)
cov = {}
for k in (16384, 32768, 49152, 65536, 98304, 131072):
    s = set(order[:k])
    cov[k] = {"corpus_mass": sum(cnt[t] for t in order[:k]) / max(1, sum(cnt.values())),
              "heldout_gen_coverage": (sum(1 for x in held for t in x if t in s) / held_n) if held_n else None}
json.dump({"vocab": vocab, "files": files, "bytes": nbytes, "distinct": len(cnt), "tokens": sum(cnt.values()),
           "heldout_gen_tokens": held_n, "coverage": cov, "order": order}, open(f"{OUT}/vocab_stats.json", "w"))
print(json.dumps({"files": files, "MB": nbytes >> 20, "distinct": len(cnt), "heldout_gen_tokens": held_n,
                  "coverage": cov}, indent=1))
