#!/usr/bin/env python3
"""Ask the running Nemotron (or any OpenAI-compatible vLLM endpoint) to critically
review a document — used here to have the model review this recipe's README.

Usage:
    ./eval_readme.py [path-to-doc]        # defaults to README.md next to this script

Environment overrides:
    VLLM_URL    default http://spark-0e81.local:8000/v1/chat/completions
    VLLM_MODEL  default nemotron-3-super
    MAX_TOKENS  default 4000   (reviews are verbose; Nemotron also uses a reasoning channel)

The model's analysis often lands in the `reasoning` channel (Nemotron reasons even
with thinking off), so both `content` and `reasoning` are printed.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("VLLM_URL", "http://spark-0e81.local:8000/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL", "nemotron-3-super")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4000"))

doc_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "README.md")
doc = open(doc_path).read()

instr = (
    "You are a senior ML-infrastructure engineer reviewing documentation. "
    "Critically evaluate the following document. Assess: (1) technical accuracy, "
    "(2) completeness, (3) clarity/structure, (4) any errors, risky advice, or "
    "missing gaps. Be specific and actionable. End with a one-line verdict and a "
    f"score out of 10.\n\n--- DOCUMENT START ---\n{doc}\n--- DOCUMENT END ---"
)
body = {
    "model": MODEL,
    "temperature": 0.3,
    "max_tokens": MAX_TOKENS,
    "chat_template_kwargs": {"thinking": False},
    "messages": [{"role": "user", "content": instr}],
}
req = urllib.request.Request(
    URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})

t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.loads(r.read())
dt = time.perf_counter() - t0

choice = d["choices"][0]
m = choice["message"]
u = d.get("usage", {})
print(f"[doc={os.path.basename(doc_path)} chars={len(doc)} "
      f"prompt_tok={u.get('prompt_tokens')} completion_tok={u.get('completion_tokens')} "
      f"finish={choice.get('finish_reason')} end_to_end={dt:.1f}s]\n")
print("=== REVIEW ===\n" + (m.get("content") or "(empty content channel)"))
if m.get("reasoning"):
    print("\n=== MODEL REASONING CHANNEL ===\n" + m["reasoning"])
