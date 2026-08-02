# Model Olympics — coding bake-off on the dual DGX Spark

A running, apples-to-apples comparison of the models we serve locally on the two DGX Sparks, on
identical coding tasks. Each task is sent to every model with the **same prompt** and (where the
model supports it) **reasoning/thinking disabled**, for a fair speed + quality comparison. Where a
task has objective correctness, it's **auto-graded** against a verified reference.

## Models under test

| Key | Model | Role | Engine | Node | Endpoint | Model name | Reasoning control |
|---|---|---|---|---|---|---|---|
| `qwen-coder` | **Qwen3-Coder-Next** (80B total / ~3B active MoE) | dedicated coder (non-thinking) | Ollama (Q4_K_M GGUF) | `spark-a` | `http://spark-a.local:11434/v1` | `qwen3-coder-next` | native non-thinking |
| `nemotron` | **NVIDIA Nemotron-3-Super-120B-A12B** | reasoning / heavy coding | vLLM (NVFP4, TP=1) | `spark-b` | `http://spark-b.local:8000/v1` | `nemotron-3-super` | `chat_template_kwargs={"thinking": false}` |
| `qwen3.6` | **Qwen3.6-35B-A3B** (qmx chat model) | general chat/coder | Ollama | `spark-a` | `http://spark-a.local:11434` | `qwen3.6:35b-a3b` | native `/api/chat` `"think": false` |

Common request params: `temperature=0.2`. Timing is wall-clock end-to-end from the calling
workstation (includes prompt prefill + generation; model-resident/warm state varies, so treat times
as indicative, not benchmark-grade).

## Results (scorecard)

| # | Task | `qwen-coder` | `nemotron` | `qwen3.6` |
|---|---|---|---|---|
| 1 | Thread-safe LRU cache + TTL | ✅ **Excellent** — clean, capacity-validated, `cleanup()` returns removed-count · 40 s | ✅ **Best-engineered** — fully typed + docstrings, `_is_expired` helper (single lookup), in-place update · 131 s | ✅ **Good** — correct + documented; minor smells: `expiry` set before lock, `__len__` mutates · 9 s |
| 2 | Word Ladder II (LeetCode 126, hard) | ✅ **18/18** — perfect (auto-graded) · 87 s | ✅ **18/18** — perfect (auto-graded) · 135 s | ⚠️ **17/18** — missed `red→tax` (a multi-solution case) · 34 s |

All Task-1 outputs were correct and thread-safe; differences are polish/efficiency. Task 2 was
auto-graded by comparing each model's *set* of shortest transformation sequences against a verified
reference over fixed LeetCode examples + random small-alphabet graphs (degenerate `beginWord ==
endWord` inputs excluded — LC 126 disallows them).

**Running read:** `qwen-coder` and `nemotron` are tied on correctness (both flawless so far), with
`qwen-coder` reaching it ~1.5× faster; `qwen3.6` is fastest but took its first correctness ding on
the harder graph task.

## Prompts (exact)

### Task 1 — Thread-safe LRU cache + TTL
```
Write production-quality Python for a Thread-Safe LRU Cache with TTL (time-based expiration). Requirements:
- Fixed capacity with LRU eviction when full.
- Optional default TTL and optional per-item TTL (seconds); items expire after their TTL.
- Thread-safe for concurrent readers and writers.
- API: get(key), put(key, value, ttl=None), __len__ (non-expired count), clear(), and cleanup() to purge expired entries.
- Lazy expiration on access, plus the explicit cleanup().
Return ONLY a single complete Python class with all imports. No explanation.
```

### Task 2 — Word Ladder II (LeetCode 126)
```
Solve LeetCode 126 'Word Ladder II'. Given beginWord, endWord, and wordList, return ALL shortest transformation sequences from beginWord to endWord. Each step changes exactly one letter; every intermediate word (and endWord) must be in wordList. Return [] if there is no sequence. Implement EXACTLY this signature:

def findLadders(beginWord: str, endWord: str, wordList: list[str]) -> list[list[str]]:

Return ONLY the complete Python (imports + function). No explanation.
```

---

*Adding tasks over time. Each entry keeps: the exact prompt, per-model rating/score, and wall-clock
time. Auto-graded where a task has objective correctness.*
