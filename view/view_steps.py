#!/usr/bin/env python3
"""Small, stdlib-only steps of build-view.sh (the tensor steps run in the serving image).

  view_steps.py relink DIR          replace symlinks to files with hardlinks to their targets (the serving container
                                    mounts only the view, so a symlink to another directory would dangle there)
  view_steps.py clone SRC DST       new view: hardlink every file, copy config.json and the index (they get edited)
  view_steps.py dh4 VIEW            quantized_layers["dh4"] = W4A16_NVFP4 (the draft head's quant prefix in the
                                    vllm-draft overlay; it avoids the checkpoint's "mtp*" / "lm_head" ignore rules)
  view_steps.py yarn512k VIEW       524,288-token context: YaRN x2 over the native 262,144 in text_config.rope_parameters,
                                    plus a top-level rope_parameters of type "default" (Qwen4ExpConfig would otherwise
                                    copy YaRN to the top level, where transformers' standardisation fails for lack of
                                    max_position_embeddings). In the view rather than --hf-overrides, so the MTP
                                    drafter gets the same RoPE.
  view_steps.py vocab IDS OUT       wrap a draft-vocab-ids.json as the {"order": [...]} stats file draft_head_sub.py reads
"""
import json, os, shutil, sys


def edit_config(view, fn):
    p = f"{view}/config.json"
    assert os.stat(p).st_nlink == 1, f"{p} is hardlinked: clone the view first"
    c = json.load(open(p)); fn(c); json.dump(c, open(p, "w"), indent=2)


op = sys.argv[1]
if op == "relink":
    d = sys.argv[2]; n = 0
    for name in sorted(os.listdir(d)):
        p = f"{d}/{name}"
        if os.path.islink(p) and os.path.isfile(p):
            target = os.path.realpath(p); os.unlink(p); os.link(target, p); n += 1
    print(f"relinked {n} files in {d}")
elif op == "clone":
    s, d = sys.argv[2].rstrip("/"), sys.argv[3].rstrip("/")
    os.makedirs(d)
    for name in sorted(os.listdir(s)):
        src = f"{s}/{name}"
        if name in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(src, f"{d}/{name}")
        elif os.path.isdir(src):
            os.symlink(os.path.realpath(src), f"{d}/{name}")
        else:
            os.link(os.path.realpath(src), f"{d}/{name}")
    print(f"cloned {s} -> {d}")
elif op == "dh4":
    edit_config(sys.argv[2], lambda c: c["quantization_config"]["quantized_layers"].__setitem__(
        "dh4", {"quant_algo": "W4A16_NVFP4", "group_size": 16}))
    print("dh4 added")
elif op == "yarn512k":
    def yarn(c):
        tc = c["text_config"]; base = dict(tc["rope_parameters"])
        rp = dict(base); rp.update({"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144})
        tc["rope_parameters"] = rp
        c["rope_parameters"] = {**base, "rope_type": "default"}
    edit_config(sys.argv[2], yarn)
    print("yarn x2 (524,288 tokens)")
elif op == "vocab":
    ids = json.load(open(sys.argv[2]))["ids"]
    json.dump({"order": ids}, open(sys.argv[3], "w"))
    print(f"{len(ids)} ids")
else:
    sys.exit(__doc__)
