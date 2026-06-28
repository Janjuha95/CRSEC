# PATCH_LOG — profiler wiring + two correctness regressions

Branch: `norm_deflection`. Instrumentation + targeted fixes only; no
architectural changes. All work verified with `python -m pytest tests/`
(114 passed) from `reverie/backend_server/`.

---

## 2026-06-28 — profiler observability, poignancy envelope, OpenAI removal

Three independent fixes. The conflict parser (`_final_output_decision`) was
already correct and was left untouched. No validated success-path return value
was changed.

### Part A — profiler observability (instrumentation only)

Already landed in `eb29f9c` (verified this pass): `set_dump_path` resolves the
path to absolute, prints `[profiler] writing to <abspath>`, and writes an
initial `dump()` so the file exists from step 0; `_FLUSH_EVERY` lowered
`200 → 25`; `reverie.start_server` dumps every step right after the movement
JSON; SIGINT/SIGTERM handlers dump then re-raise (SIGKILL can't be caught);
the committed `backend_server/profile.json` dummy was removed and
`profile.json` + `**/profile.json` added to `.gitignore`. Test:
`TestProfiler::test_set_dump_path_writes_file_with_counts`.

### Part B — poignancy fail-safe on every call

Root cause: `ChatGPT_safe_generate_response`
(`persona/prompt_template/gpt_structure.py`) did
`json.loads(curr[:rfind('}')+1])["output"]`, which throws on any qwen3 output
that isn't a bare `{"output": ...}` envelope (bare int, `5/10`, prose before
the `{`). The `except: pass` then burned all retries → `False` → `[FAIL_SAFE]`,
so the poignancy `func_clean_up` int-extractor was never reached.

- `ChatGPT_safe_generate_response` is now envelope-tolerant: it slices to the
  first `{` (so `text\n{...}` parses), wraps the `["output"]` extraction in its
  own `try`, and on failure falls back to the **raw stripped response** passed
  to `func_validate` / `func_clean_up`. Well-formed envelopes parse to exactly
  the same value as before, so working callers are unaffected.
- Removed the stray `print("…DEBUG 7")` and the trailing `True` (verbose) arg
  at the `run_gpt_prompt_event_poignancy` call site.
- Fixed the latent inconsistency in all three poignancy functions
  (event/thought/chat) where `__chat_func_validate` validated via
  `__func_clean_up` instead of `__chat_func_clean_up` — validation now matches
  the value actually returned. (Left the other ~6 unrelated functions that share
  the pattern untouched.)
- Test: `TestPoignancyEnvelopeTolerance` feeds `"5"`, `"I'd rate it 5/10."`,
  `text\n{"output":"7"}`, `{"output":"3"}` through the real poignancy path and
  asserts ints `5/5/7/3` with no fail-safe.

### Part C — port SpecificNormUtility off OpenAI (#8), crash-safe

Root cause: `SpecificNormUtility.specific_norm_utility`
(`norm/run_gpt_prompt_norm.py`) called `openai.ChatCompletion.create(...)`, but
`openai` is never imported → `NameError` → `except: return False`. The consumer
`norm_evaluate.py:565` then did `if len(utility) != 2` — `len(False)` is a
`TypeError` that crashed any long run reaching `norms_evaluate`.

- `specific_norm_utility` now routes through `from llm_router import llm_call`
  with `call_type="norm_evaluation"`, joining `self.msg` contents exactly like
  `norm/creation.py` `Creation.creation`. The `OUTPUT: <int>. <reason>` parse is
  kept, but the fail-safe is shape-matched: any inference/parse failure returns
  `[4, "fail_safe"]` (length 2) — never bare `False`/`[False]`. Removed the
  `print(self.msg)` / `print(gpt_ret)` debug lines.
- Defense in depth at `norm_evaluate.py:565`:
  `if not isinstance(utility, (list, tuple)) or len(utility) != 2: return False, new_norm`.
- The defector mirror `defection_engine.get_defector_norm_utility` now returns
  the same length-2 `[4, "fail_safe"]` on failure (was `[False]`); docstring
  updated.
- `model="gpt-3.5-turbo-16k"` on `SpecificNormUtility`/`Creation` is dead config
  (everything routes through `llm_call`) — left as-is, harmless.
- Tests: `TestSpecificNormUtilityPort` (parse → `[7, "because X"]`; raise →
  length-2 fail-safe; `generate_normal_norm_utility` consumer does not raise)
  and `TestDefectorNormUtilityShape` (parse + length-2 fail-safe).

---

## Follow-up — profiler output reliability

The profiler records correctly (at `llm_router.llm_call`), but `profile.json`
looked empty in practice: real dumps land in
`environment/frontend_server/storage/<sim>/profile.json` while the stale
committed `backend_server/profile.json` was being read, and the file only
flushed every 200 calls or on `fin`, so short / hard-killed runs never wrote
it. These changes make the output reliable. Instrumentation only — no sim
behavior or success-path returns changed. Recording was **not** added to
`gpt_structure.py` (every call already funnels through `llm_router`;
double-recording would inflate counts).

- **Unambiguous active path** (`call_profiler.set_dump_path`): the path is now
  resolved with `os.path.abspath`, announced once as
  `[profiler] writing to <abspath>`, and an initial `dump()` is written
  immediately so the file exists from step 0. `set_dump_path(None)` is
  tolerated (turns auto-dumps back off; used by tests).
- **Tighter flush cadence** (`call_profiler._FLUSH_EVERY`): `200 → 25`.
  Primary cadence is now the per-step dump below; this is the safety net for
  steps that issue many calls.
- **Per-step dump** (`reverie.start_server`): right after the per-step movement
  JSON is written, `call_profiler.dump(f"{sim_folder}/profile.json")` runs, so
  the profile updates every step regardless of call volume.
- **Survive hard exits** (`reverie.py`): added `import signal`; in
  `ReverieServer.__init__` (right after `set_dump_path`) SIGINT/SIGTERM
  handlers dump the profile, restore the default disposition, and re-raise so
  the process still terminates normally. Handler registration is guarded
  (`signal.signal` only works in the main thread). SIGKILL / Windows
  TerminateProcess cannot be caught — noted in the code comment.
- **Removed the misleading fixture**: deleted the committed
  `reverie/backend_server/profile.json` (the `run_gpt_prompt_x` /
  `some_counter` dummy) via `git rm`, and added `profile.json` +
  `**/profile.json` to `.gitignore` (it is always a generated artifact).

### Test

`tests/test_optimizations.py::TestProfiler::test_set_dump_path_writes_file_with_counts`
— `set_dump_path` to a temp file (asserts the file exists from step 0), record
N=7 calls, `dump()`, then assert `total_llm_calls == 7` and a non-empty
`per_function` with the right per-function count. Cleanup resets the profiler
and the global dump path.

---

## Part A — Profiler (the numbers we need)

### A1 — instrument the real choke point

The 9-step run produced `profile.json` with `total_llm_calls: 0` / `counters: {}`
because nothing on the hot path recorded into `call_profiler`.

The **real** single choke point is **`llm_router.llm_call`**, not
`gpt_structure.py`. Every Ollama *generate* call funnels through it:

```
ChatGPT_safe_generate_response / *_OLD / *_OLD_t0 / *_OLD_t1 / safe_generate_response
        -> ChatGPT_request / GPT4_request / GPT_request / ...
        -> openai_compat_call
        -> llm_router.llm_call   <-- recording happens here
```

This also captures the norm module's `safe_generate_response` path, which
`gpt_structure` wrappers do **not** sit in front of. `llm_router.llm_call`
already contained `call_profiler.record_call(prompt_fn, elapsed)` on both the
success and exception branches (it records each actual Ollama hit, including
retries, which is what we want for a load profile). Attribution uses
`_caller_prompt_fn()` — the nearest `run_gpt_*` frame on the stack.

**Decision:** recording stays at `llm_router.llm_call` and was **not** added to
the `gpt_structure` wrappers. Adding it in both places would double-count every
call, since the wrappers merely delegate downstream. (`get_embedding` calls
`ollama.embeddings` directly and is intentionally excluded — embeddings are a
different cost class and would pollute `total_llm_calls`.)

### A2 — incremental flush (`call_profiler.py`, `reverie.py`)

`profile.json` previously was only written on a clean `fin` (which never
happened) plus an `atexit` fallback that does not fire when the process is
hard-killed.

- `call_profiler.record_call` now flushes to disk **every `_FLUSH_EVERY` (200)
  recorded calls**, so a run cut off mid-way still leaves a partial profile.
  The flush happens outside the lock (dump → snapshot re-acquires the lock) to
  avoid re-entrant deadlock.
- `call_profiler.set_dump_path(path)` configures the target; `reverie.py`
  `ReverieServer.__init__` points it at `<sim_folder>/profile.json` so the
  incremental flush, the existing `save()` dump (sim save / `fin`), and the
  `atexit` fallback all land in the same place.
- The incremental flush **and** the `atexit` fallback only fire once a dump
  path has been set (i.e. a real run). This stops unit tests — which call
  `record_call` but never set a path — from clobbering a checked-in
  `profile.json` via `atexit`. (Confirmed: running the suite no longer modifies
  `reverie/backend_server/profile.json`.)

### A3 — verification counters

Counters now emitted into `profile.json["counters"]`:

| counter | site | meaning |
|---|---|---|
| `violation_check_cache_hit` | `run_gpt_prompt_norm.run_gpt_prompt_violation_check` | memoized violation-check served from cache |
| `violation_check_cache_miss` | same (added) | memoization on, but cache miss → real LLM call |
| `prefilter_skip_benign` | `violation_detection.detect_violations` (renamed from `prefilter_skipped_benign`) | event is benign (idle/sleeping/chatting), all active norms skipped |
| `prefilter_skip_no_overlap` | same (renamed from `prefilter_skipped_no_overlap`) | no event↔norm stem overlap, LLM skipped |
| `prefilter_llm_checked` | same (unchanged name) | reached the LLM after the pre-filter |
| `small_model_call` | `llm_router.llm_call` (added) | routed to `SMALL_MODEL` (qwen3:4b) |
| `primary_model_call` | `llm_router.llm_call` (added) | routed to primary/reasoning model |
| `pronunciatio_stub` | `run_gpt_prompt.run_gpt_prompt_pronunciatio` (renamed from `pronunciatio_headless_stub`) | headless emoji stub, LLM skipped |

The prefilter and pronunciatio counters were **renamed** to match the keys in
the spec (no test asserted the old names). `small_model_call` /
`primary_model_call` are incremented once per logical call, before the retry
loop.

### profile.json keys now emitted

```json
{
  "total_llm_calls": <int>,
  "total_llm_seconds": <float>,
  "per_function": { "<run_gpt_*>": { "count": <int>, "seconds": <float> }, ... },
  "counters": {
    "violation_check_cache_hit": <int>,
    "violation_check_cache_miss": <int>,
    "prefilter_skip_benign": <int>,
    "prefilter_skip_no_overlap": <int>,
    "prefilter_llm_checked": <int>,
    "small_model_call": <int>,
    "primary_model_call": <int>,
    "pronunciatio_stub": <int>
  }
}
```

(Each counter only appears once it has been incremented at least once.)

---

## Part B — poignancy regression

### Root cause

`run_gpt_prompt_event_poignancy`, `run_gpt_prompt_thought_poignancy`, and
`run_gpt_prompt_chat_poignancy` all go through **`ChatGPT_safe_generate_response`**,
which wraps the prompt asking for a strict JSON envelope and parses the reply
with:

```python
end_index = curr_gpt_response.rfind('}') + 1
curr_gpt_response = json.loads(curr_gpt_response[:end_index])["output"]
```

The optimization pass added these three functions to
`llm_router.SMALL_ROUTE_PROMPT_FNS`, routing them to **`qwen3:4b`**. The small
model does not reliably emit the `{"output": "<int>"}` envelope — it tends to
return a bare integer / prose. When it returns e.g. `5`, `json.loads("5")` is
the int `5` and `5["output"]` raises `TypeError`; the wrapper's bare `except`
swallows it, all `repeat` attempts fail, and the function returns `False` →
the `[FAIL_SAFE] ... ChatGPT path returned False` branch on **nearly every
call**. It was rare before the pass because the larger primary model satisfies
the envelope. (The `num_ctx` cap is not implicated: poignancy prompts/outputs
are tiny and well under 16k.)

### Fix

Per the spec ("if qwen3:4b output doesn't parse for poignancy, EXCLUDE
poignancy from small-model routing"), the three poignancy functions were
**removed from `SMALL_ROUTE_PROMPT_FNS`** in `llm_router.py`. They now run on
the primary/reasoning model and parse correctly again. `pronunciatio` and
`event_triple` remain small-routed (they don't use the strict JSON envelope).

The success-path returns of the poignancy functions were **not** touched.

### Tests

- `TestTieredRouting.test_poignancy_uses_primary_model` (new) — all three
  poignancy fns route to the primary model with tiered routing ON.
- `test_small_model_for_cosmetic_fns` / `test_tiered_routing_disabled` updated
  to use `run_gpt_prompt_pronunciatio` (still a small-routed fn).

---

## Part C — conflict-parser inversion

### Root cause

In `run_gpt_prompt_decide_if_norm_conflict` (`check_conflict_decide_talk_v5`),
the conflict flag was parsed by splitting the model output on the brittle
literal prompt substring:

```python
gpt_response.split('whether there is a conflict?\nAnswer in "yes" or "no" and provide a reason: ')[-1]
```

Qwen3 rarely echoes that exact substring (it interleaves `Reasoning:` etc.), so
the split returned the **whole response** and `_yn` then took the *first*
yes/no token anywhere — frequently a `yes` buried in the reasoning or in the
echoed `Answer in "yes" or "no"` instruction. Result: a genuine
`FINAL OUTPUT: No.` was logged as `conflict=yes` (no→yes inversion). When even
that failed to validate it fell back to `['ERROR']`.

### Fix

Added two module-level helpers in `run_gpt_prompt_norm.py`:

- `_yn_token(seg)` — first standalone yes/no, after stripping markdown
  emphasis / brackets / backticks (`**No**`, `[Yes]`).
- `_final_output_decision(gpt_response)` — authoritatively reads the
  **FINAL OUTPUT** line: uses the *last* `final output` marker (tolerates
  reasoning preamble and template echoes), handles a missing space after the
  colon / answer on the next line, and falls back to the last yes/no-bearing
  line. Returns `yes` / `no` / `None`.

The nested `__func_validate` / `__func_clean_up` now use
`_final_output_decision`. Per the template (FINAL OUTPUT is forced to `No`
whenever there is no conflict) the FINAL OUTPUT line is authoritative for both
returned fields: `No → no-conflict`, `Yes → conflict`. `output[0]` (the
conversation decision consumed by `norm_retrieve.generate_decide_if_norm_conflict`)
is unchanged in shape and still drives `x[0] == "yes"`.

`get_fail_safe` still returns `['ERROR']` (length 1). The caller checks
`output[0] == "yes"`, so `"ERROR"` cleanly means "no conflict, skip" without
crashing; the `if len(output) == 3` guard already skips the debug print for the
fail-safe.

### Tests

- `TestConflictParserUnit` — `_final_output_decision` on: reasoning-says-no +
  `FINAL OUTPUT: No`, `FINAL OUTPUT: Yes`, markdown/brackets, answer on next
  line, and undecidable → `None`.
- `TestConflictParserIntegration` (end-to-end through
  `run_gpt_prompt_decide_if_norm_conflict`):
  - **the required sample** — reasoning contains a stray `yes` but concludes
    `No` with `FINAL OUTPUT: No` → asserts `conflict=False` (`output[2] == "no"`,
    `output[0] == "no"`). This is exactly the case the old parser inverted.
  - `FINAL OUTPUT: Yes` → conflict.
  - unparseable response → `['ERROR']`, and `output[0] != "yes"` (clean skip).

---

## Constraints honored

- No validated success-path return values were altered (poignancy returns,
  violation-check cache shape, conflict `output[0]` shape).
- Instrumentation is behavior-neutral: counters/flush never change control flow;
  `dump()` never raises.
- A test accompanies each fix; full suite is green (108 passed).

## Files changed

- `reverie/backend_server/call_profiler.py` — settable dump path, incremental
  flush, atexit guard.
- `reverie/backend_server/reverie.py` — `set_dump_path` in `__init__`.
- `reverie/backend_server/llm_router.py` — small/primary counters; poignancy
  removed from small-model routing.
- `reverie/backend_server/norm/run_gpt_prompt_norm.py` — `violation_check_cache_miss`
  counter; FINAL-OUTPUT-authoritative conflict parser + helpers.
- `reverie/backend_server/norm/violation_detection.py` — prefilter counter renames.
- `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py` —
  `pronunciatio_stub` counter rename.
- `reverie/backend_server/tests/test_optimizations.py` — Part B/C tests + routing
  test updates.
