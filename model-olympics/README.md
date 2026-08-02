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
| 3 | Thread-safe **O(1) LFU** cache (LeetCode 460, hard + concurrency + complexity) | ✅ **full pass** — `OrderedDict` buckets + `RLock`; concise, fastest ops (0.08 s/400k) · 78 s | ✅ **full pass** — textbook hand-rolled doubly-linked-list O(1) + `RLock`; most rigorous · 113 s | ✅ **full pass** — `OrderedDict` + plain `Lock` (non-reentrant, less defensive); concise · 38 s |
| 4 | **Async** token-bucket rate limiter (`asyncio`) | ✅ **full pass** — `asyncio.Lock` + computed-sleep (no busy-wait); 41 lines · 56 s | ✅ **full pass** — clean while-loop, sleeps outside the lock; 33 lines · 56 s | ✅ **full pass** — most concise (28 lines), same correct pattern · 31 s |

*(Task 3 "full pass" = all three auto-graded checks passed: functional correctness vs a verified O(1) reference over random op sequences, a 16-thread concurrency stress test with no crash/deadlock, and a complexity check of 400k ops completing in <0.25 s — i.e. genuinely O(1), no scanning.)*

*(Task 4 "full pass" = all three async timing checks passed: refill rate (~0.5 s to refill 5 tokens @10/s), concurrent burst+sustained (25 concurrent acquires ≈ 2.0 s), and a rate-cap check (~14 grants in 1 s — a broken limiter would grant thousands). All three used `asyncio.Lock` + a computed `asyncio.sleep` outside the lock, i.e. no busy-wait.)*

All Task-1 outputs were correct and thread-safe; differences are polish/efficiency. Task 2 was
auto-graded by comparing each model's *set* of shortest transformation sequences against a verified
reference over fixed LeetCode examples + random small-alphabet graphs (degenerate `beginWord ==
endWord` inputs excluded — LC 126 disallows them).

**Running read:** `qwen-coder` and `nemotron` are tied on correctness (both flawless across all 3
tasks), with `qwen-coder` reaching it ~1.5× faster. `qwen3.6` is consistently fastest but has been
slightly less reliable (one wrong answer on Word Ladder II; a non-reentrant `Lock` choice on LFU).
On the hard concurrency+complexity task all three produced a correct thread-safe O(1) LFU — Nemotron
with the most rigorous (real doubly-linked-list) implementation, `qwen-coder` the best balance of
concision and speed. On the async task (Task 4) all three again produced a correct token bucket
(`asyncio.Lock` + computed sleep, no busy-wait) — no correctness separation; `qwen3.6` was most
concise and fastest to generate. **Net so far:** correctness is nearly saturated (only `qwen3.6`'s
single Word-Ladder-II miss); the real spread is **speed** (`qwen3.6` > `qwen-coder` > `nemotron`) and
**rigor/polish** (`nemotron` ≥ `qwen-coder` > `qwen3.6`).

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

### Task 3 — Thread-safe O(1) LFU cache (LeetCode 460 + concurrency + complexity)
```
Design a THREAD-SAFE LFU (Least Frequently Used) cache with O(1) average time for both operations. Implement EXACTLY this class:

class LFUCache:
    def __init__(self, capacity: int): ...
    def get(self, key: int) -> int:            # value, or -1 if absent; a hit counts as a use
    def put(self, key: int, value: int) -> None:  # insert/update; on overflow evict the least-frequently-used key, breaking ties by least-recently-used

Requirements:
- get and put must be O(1) average time (use frequency buckets of doubly-linked lists / ordered maps + a min-frequency pointer; do NOT scan all keys).
- Thread-safe for concurrent get/put from many threads.
- capacity may be 0 (then nothing is ever stored; get always returns -1).
Return ONLY the complete Python (imports + class). No explanation.
```

### Task 4 — Async token-bucket rate limiter (asyncio)
```
Implement an ASYNC token-bucket rate limiter using asyncio. Implement EXACTLY this class:

class AsyncRateLimiter:
    def __init__(self, rate: float, capacity: float):
        # rate = tokens refilled per second; capacity = max tokens (burst size)
    async def acquire(self, tokens: float = 1.0) -> None:
        # wait until `tokens` are available, consume them, then return

Requirements:
- Token bucket: refills continuously at `rate` tokens/sec up to `capacity`; starts FULL.
- acquire(n) awaits until n tokens are available, then consumes them.
- Safe for many coroutines awaiting concurrently; NO busy-wait polling loop — use asyncio primitives and sleep for the computed wait time.
- Use a monotonic clock (time.monotonic or loop.time()).
Return ONLY the complete Python (imports + class). No explanation.
```

---

*Adding tasks over time. Each entry keeps: the exact prompt, per-model rating/score, and wall-clock
time. Auto-graded where a task has objective correctness.*
