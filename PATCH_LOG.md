# PATCH_LOG — profiler wiring + two correctness regressions

---

## 2026-07-08 — num_predict caps, per-convo relationship cache, summarize routing, retry instrumentation

Branch `norm_deflection`, on top of `5b581e1`. Perf pass: generation-token
caps at the router choke point, hoisting a per-utterance 32b call out of the
conversation loop, moving two summarize fns off the 32b, and making
retry burn in the safe_generate wrappers visible. Every behavior change has an
env kill-switch defaulting to the new behavior. All touched files pass
`python -m py_compile`; 3 canary `llm_call()`s (int-score, yes/no,
dialogue-JSON) parsed cleanly with `done_reason != "length"` (run on
`qwen3:32b` via `CRSEC_TIERED_ROUTING=0` — this machine does not have
`qwen3:30b-a3b`/`qwen3:4b` pulled).

### Change 1 — Output-token caps (`llm_router.py`)

`llm_call()` previously set only `temperature`/`num_ctx`; classifier-style
calls could ramble unboundedly. Added `PROMPT_FN_NUM_PREDICT` (60 fns, keyed
by the same names `_caller_prompt_fn()` returns) wired into
`kwargs["options"]["num_predict"]`. `llm_logs/calls.jsonl` had no `prompt_fn`
field (114 smoke-test rows on `qwen3:8b`), so caps come from static tiers
cross-checked against each parser's contract + legacy `gpt_param` max_tokens:
32 for bare int/yes-no/option (poignancy ×3, decide_to_talk/react,
wake_up_hour, safety_score, duplicate_check, pronunciatio), 64 triples/names,
128 one-liners, 256 summaries/structured norm checks, 512 dialogue JSON,
768 norm creation/format/synthesis + `decide_if_norm_conflict`, 1024
insight_and_guidance, 2048 planning/decomp.

Deviations from the prompt's static tiers, from parser evidence:
`violation_check` 32→128 (4-key JSON), `seeds_content_check`/
`seeds_type_check(_v2)`/`fact_consistency_check` 32→256 (multi-segment
answers), pronunciatio 16→32 (floor), `decide_if_norm_conflict` →768: its
template demands 3× step-by-step reasoning before FINAL OUTPUT and
`_final_output_decision`'s bottom-scan fallback means truncation could
silently invert the decision rather than fail.

Truncation tripwire: `done_reason == "length"` →
`call_profiler.incr(f"truncated_{prompt_fn}")` + `_log_call` error field. A
too-tight cap causes parse-fail → repeat-loop re-fire (slower, not faster);
nonzero `truncated_*` counters in calib_006 are the signal to raise a cap.

Flags: `CRSEC_NUM_PREDICT=0` disables all caps; `CRSEC_NUM_PREDICT_DEFAULT`
(default 1024) for unlisted fns and `prompt_fn=="unknown"` direct callers
(defection_engine, creation.py, SpecificNormUtility).

### Change 2 — Per-conversation relationship cache (`converse.py`)

`agent_chat_v2` recomputed `generate_summarize_agent_relationship` (a 32b call
until Change 3, now PRIMARY) plus a 50-node `new_retrieve` on **every**
utterance — up to 10× per conversation for a quantity that cannot change
mid-conversation. Both directions are now computed once before the loop and
reused; the per-turn 15-node retrieve with `last_chat` is untouched.
`relationship_cache_hit` increments per in-loop use (hits ≈ utterances;
LLM calls saved ≈ hits − 2). Note: a conversation ending after 1 utterance now
costs 2 relationship calls instead of 1; break-even at 2, win from 3 on.
Flag: `CRSEC_RELATIONSHIP_CACHE=0` restores per-utterance recomputation.

### Change 3 — Summarize fns to PRIMARY (`llm_router.py`)

`run_gpt_prompt_agent_chat_summarize_ideas` and
`run_gpt_prompt_agent_chat_summarize_relationship` removed from
`REASONING_ROUTE_PROMPT_FNS` → fall through to `qwen3:30b-a3b`. They are
recall aids, not decisions. Spoken-line generators,
`run_gpt_prompt_decide_if_norm_conflict`, and all `run_gpt_prompt_norm*` stay
on the 32b. Tier comments updated.
Flag: `CRSEC_SUMMARIZE_ON_REASONING=1` re-adds both to the reasoning set.

### Change 4 — Retry-burn instrumentation (`gpt_structure.py`)

The `_OLD`/`_t0`/`_t1` wrapper variants turned out to be the live workhorses
(every norm fn + several persona fns call them), so all seven repeat loops are
instrumented: `retry_{fn}` on each failed iteration, `fail_safe_{fn}` on
exhaustion, with `fn` resolved lazily via `llm_router._caller_prompt_fn()`
(stack walk only on the failure path; helpers `_count_retry`/
`_count_fail_safe`). `compress_sim_storage_norm.py`'s private wrapper copy is
an offline tool and was left alone. **The retry audit + parser fixes (4b/4c)
were skipped: `calib_005.log` and `calib_005/profile.json` are absent on this
machine.** The calib_006 `retry_*`/`fail_safe_*` ranking is the input for that
follow-up.

### What to watch in calib_006

- `sec/step` vs calib_005 (expect a drop from caps + cache + routing).
- `truncated_*` counters ≈ 0; any nonzero one names the fn whose cap to raise.
- `retry_*` ranking → the 4c parser-fix shortlist.
- `relationship_cache_hit` ≈ total utterances across conversations.

### Files changed

- `reverie/backend_server/llm_router.py` — cap table + wiring, tripwire,
  summarize fns off reasoning set.
- `reverie/backend_server/persona/cognitive_modules/converse.py` — hoisted
  relationship summaries in `agent_chat_v2`.
- `reverie/backend_server/persona/prompt_template/gpt_structure.py` —
  retry/fail-safe counters in all seven safe_generate wrappers.

---

## 2026-06-29 — three-tier routing, event_triple to PRIMARY, stepper banner

Branch `norm_deflection`. Routing/instrumentation changes only; no validated
success-path return value was altered. Verified with `python -m pytest tests/`
(117 passed) from `reverie/backend_server/`.

**Root cause (throughput):** every wrapper in `gpt_structure.py` calls
`openai_compat_call(prompt, call_type="conversation")`. `"conversation"` was in
`REASONING_CALL_TYPES`, so **all** persona work (planning, actions, poignancy,
reflection, dialogue) routed to `qwen3:32b` (24 GB dense, slow). The intended
fast primary `qwen3:30b-a3b` (MoE, ~3 B active params) was never loaded. Net:
bulk ran on the slowest model — ~5–8× the per-token cost — driving ~40 s/step.

### Part 1 — Three-tier name-based routing (`llm_router.py`)

Replaced the two-tier call_type-only routing with a three-tier precedence keyed
primarily on the calling `run_gpt_*` function name (same stack walk as the
profiler, so no changes to gpt_structure wrappers):

| Tier | Condition | Target |
|------|-----------|--------|
| 1 | `prompt_fn in SMALL_ROUTE_PROMPT_FNS` | `SMALL_MODEL` (qwen3:4b) |
| 2 | `prompt_fn in REASONING_ROUTE_PROMPT_FNS` or `prompt_fn.startswith("run_gpt_prompt_norm")` | `REASONING_MODEL` (qwen3:32b) |
| 3 | `call_type in REASONING_CALL_TYPES` | `REASONING_MODEL` |
| 4 | else | `PRIMARY_MODEL` (qwen3:30b-a3b) |

`SMALL_ROUTE_PROMPT_FNS = {"run_gpt_prompt_pronunciatio"}` (event_triple
removed — see Part 2).

`REASONING_ROUTE_PROMPT_FNS` = the seven conversation/norm-decision functions
(`run_gpt_generate_iterative_chat_utt`, `run_gpt_prompt_agent_chat`,
`run_gpt_prompt_create_conversation`, `run_gpt_prompt_generate_next_convo_line`,
`run_gpt_prompt_agent_chat_summarize_ideas`,
`run_gpt_prompt_agent_chat_summarize_relationship`,
`run_gpt_prompt_decide_if_norm_conflict`). All `run_gpt_prompt_norm*` functions
match via `startswith`.

`REASONING_CALL_TYPES = {"norm_creation", "norm_evaluation",
"conflict_detection", "defection_assessment"}`. Removed `"conversation"`,
`"default"`, `"violation_check"` — these are high-volume bulk and belong on
PRIMARY.

`PRIMARY_CALL_TYPES` removed (dead code — no caller passed those types).
`_get_model` fallback now returns `PRIMARY_MODEL` (not REASONING) and logs at
`info` level only for genuinely unrecognized call_types (not for known bulk
types like `"conversation"`).

`CRSEC_TIERED_ROUTING=0` escape hatch preserved: routes everything to
`REASONING_MODEL` to match the pre-patch baseline cleanly for A/B comparison.

Added `reasoning_model_call` profiler counter so the three-way split
(small / reasoning / primary) is auditable in `profile.json`.

Live check: after this patch, `ollama ps` should show `qwen3:30b-a3b` loaded
and serving the bulk. `profile.json → counters.primary_model_call` should
dominate; `reasoning_model_call` only on convo/norm steps.

VRAM note: 30b-a3b (~18 GB) + 32b (24 GB) + 4b (5.4 GB) + nomic (0.6 GB)
≈ 48 GB — fits the A100-80GB with `OLLAMA_KEEP_ALIVE=24h`, no model-swap thrash.

### Part 2 — `event_triple` to PRIMARY + fail-safe counter

`run_gpt_prompt_event_triple` was in `SMALL_ROUTE_PROMPT_FNS` (qwen3:4b). The
small model mis-parsed `(s, p, o)`, fail-safing to `('X', 'is', 'idle')` and
flattening real café events in associative memory (hurts RQ2). It now hits
PRIMARY (tier 4).

- Added `call_profiler.incr("event_triple_fail_safe")` on the fail-safe branch
  so mis-parse rate on 30b-a3b is measurable.
- Hardened `__func_clean_up` to strip outer quotes from the whole response
  and from each split element, and to drop trailing prose after ` //` or ` --`
  separators. No other function's parser was touched.

### Part 3 — Stepper startup banner (`headless_stepper.py`)

Added `print(f"[stepper] max_step={max_step}", flush=True)` immediately after
`max_step` is resolved. The default of 100000 was already in place (not changed).
A mismatch between `headless_stepper.py max_step` and the step count passed to
reverie was previously silent; it is now visible in the stepper log.

### Tests (`tests/test_optimizations.py`)

- `_call_via` now accepts an optional `call_type` parameter (default
  `"conversation"`) so tier-3 routing can be exercised from a generic frame.
- `test_reasoning_model_for_other_fns` → renamed
  `test_primary_model_for_bulk_fns`; assertion updated from `REASONING_MODEL`
  to `PRIMARY_MODEL` (`run_gpt_prompt_violation_check` is now tier-4 default).
- `test_poignancy_uses_primary_model` → assertion updated from `REASONING_MODEL`
  to `PRIMARY_MODEL` (poignancy is not in any special routing set; the
  `call_type="conversation"` falls through to tier 4).
- `test_three_tier_mapping` (new): asserts all five required mappings —
  `run_gpt_prompt_daily_plan`→PRIMARY, `run_gpt_generate_iterative_chat_utt`→
  REASONING, `run_gpt_prompt_pronunciatio`→SMALL, `call_type="norm_evaluation"`→
  REASONING, `call_type="violation_check"`→PRIMARY.
- `test_norm_reflect_fn_routes_to_reasoning` (new): verifies the `startswith`
  rule for norm module callers.
- `test_event_triple_routes_to_primary` (new): confirms event_triple left SMALL.

### Files changed

- `reverie/backend_server/llm_router.py` — three-tier routing, REASONING_ROUTE_PROMPT_FNS,
  trimmed REASONING_CALL_TYPES, PRIMARY_CALL_TYPES removed, reasoning_model_call counter.
- `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py` —
  event_triple_fail_safe counter + hardened __func_clean_up.
- `headless_stepper.py` — max_step startup banner.
- `reverie/backend_server/tests/test_optimizations.py` — updated + new routing tests.

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

---

## 2026-07-08 — Pass 2 on calib_008 data: parsers, norm routing, caps v2, metrics wiring

Branch `norm_deflection`, on top of `462a1a6`. Driven by the calib_008
artifacts (200 steps, non-thinking models, 2,249 captured calls in
`calib_artifacts/calib_008/calls.jsonl`). Calls were attributed to prompt fns
by fingerprinting each fn's template's static text; attribution matches
profile.json call counts exactly for all 29 fns with calls.

### Change A — top-5 failing parsers (`run_gpt_prompt.py`, `run_gpt_prompt_norm.py`)

calib_008 ground truth: retry counters EQUAL call counts for four fns — they
failed on 100% of attempts and always returned fail_safe.

| fn | failure mode (from captured responses) | fix (parser-side only) |
|---|---|---|
| event_triple (396 calls, 132 fail_safe) | Qwen3 re-emits the full `(subject, predicate, object)` although the template primes `"Output: (SUBJ,"`; parser demanded exactly 2 elements → every event flattened to `('X','is','idle')` | `_parse_event_triple_completion`: drop the echoed subject (matched against the primed subject read back from the prompt), fold extra commas into the object slot |
| act_obj_event_triple (170, 34) | same (shared template; subject = game object) | same helper |
| norm_format (75, 25) | responses are VALID JSON; `_strip_scaffolding`'s "drop leading `{` with no `}` on line 1" heuristic beheads multi-line JSON, making the first-{-to-last-} slice unbalanced → 100% failure → **every norm seed of the run silently dropped** (this, not metrics wiring, is why RQ1/RQ3 were empty) | `_parse_norm_format_json`: no scaffolding strip; `raw_decode` of the first complete JSON value; brace-balancing recovery for truncated output; normalizes `{"norm_1": ...}` / bare-object / wrapper shapes; missing required keys now fail in clean_up so the repeat loop retries instead of `generate_format_norm` silently returning None |
| new_decomp_schedule (90, 18) | markdown restatement (`**09:00 ~ 09:30** —` unicode dashes, prose headers, annotation lines) vs required literal `" -- "` split | `_parse_new_decomp_schedule`: regex line scan + greedy chain-walk that only accepts a contiguous tiling of the scheduling window (subsumes the old sum-of-durations validator); handles full-restatement, mid-line completion, and standalone-restart response shapes |
| focal_pt (72, 18) | model returns a perfect `{"output": [...]}`; the envelope extraction hands the parser a decoded **list**, which both the str-only tolerant parser and `ast.literal_eval` rejected | `_parse_focal_pt_candidate`: accepts list / str-encoded list / numbered lines; used by both chat and fallback paths |

Replay acceptance (script: `reverie/backend_server/tests/replay_calib_parsers.py`,
run against all captured calib_008 responses through the shipped source parsers):

| fn | old pass rate | new pass rate |
|---|---|---|
| event_triple | 0% | **396/396 = 100%** |
| act_obj_event_triple | 0% | **170/170 = 100%** |
| norm_format | 0% | **75/75 = 100%** |
| new_decomp_schedule | 0% | **85/90 = 94.4%** (residual = ONE degenerate response × its 5 retries: model emitted 18 min of a 120-min window then em-space padding — nothing to parse; fail_safe reconstruction is correct. 85/85 responses containing a full schedule parse.) |
| focal_pt | 25% | **72/72 = 100%** |

No prompt template was changed. `_strip_scaffolding` itself was NOT modified
(shared by validated parsers); norm_format simply stopped using it.
Full test suite: 116 passed, 1 deselected (`test_options_pin_temperature_and_num_ctx`
— pre-existing failure introduced by `462a1a6`'s think-block change, fails on a
clean checkout too; not touched here as the think block is validated behavior).

### Change B — dominant norm fns to PRIMARY behind `CRSEC_NORM_ON_PRIMARY` (`llm_router.py`)

calib_008: `decide_if_norm_conflict` 264x/2,297s + `norm_reflect_from_thoughts`
71x/1,548s = 3,845s of 7,661s total LLM time (50.2%), both on the dense 32b.
New flag `CRSEC_NORM_ON_PRIMARY` (default "1", read at call time like
CRSEC_TIERED_ROUTING): routes exactly these two fns to PRIMARY_MODEL via
`NORM_PRIMARY_OVERRIDE_FNS`, checked AFTER the SMALL tier but BEFORE both the
tier-2 REASONING set (which contains decide_if_norm_conflict) and the
`startswith("run_gpt_prompt_norm")` prefix rule (which catches
norm_reflect_from_thoughts). All other norm_* fns stay on the 32b. Flag "0"
restores the previous routing exactly (covered by unit tests:
`test_norm_on_primary_default`, `test_norm_on_primary_disabled_restores_reasoning`;
the old `test_norm_reflect_fn_routes_to_reasoning` now checks the prefix rule
via norm_format instead).

Live-parse sanity DEFERRED to the cluster: this dev machine has only
qwen3:8b/qwen3:32b pulled (no `qwen3:30b-instruct`), so the "run one
representative prompt of each through PRIMARY and eyeball the parse" check
cannot run here — same limitation as the previous pass's canaries. Watch
`retry_run_gpt_prompt_decide_if_norm_conflict` / `retry_..._norm_reflect_from_thoughts`
in calib_009's profile.json: if the 30b's output format breaks
`_final_output_decision` (bottom-scan, unchanged) or the strict
"- Norm: ... Related thought: ..." reflect parser, retries will show it
immediately, and CRSEC_NORM_ON_PRIMARY=0 is the rollback.

### Change C — num_predict caps v2, derived from calib_008 (`llm_router.py`)

The v1 table came from thinking-contaminated logs / static tiers, truncated
live planning calls, and was disabled (CRSEC_NUM_PREDICT=0). v2 replaces the
whole PROMPT_FN_NUM_PREDICT table using calib_008's 2,249 calls (attributed
per fn by template fingerprint, exact vs profile.json): per-fn p99 response
chars, tokens ≈ chars/3.5, cap = max(64, ceil(2 × p99_tokens)).

Tight, data-derived caps ONLY for parsers verified to read from the top of
the output (a post-payload cut cannot corrupt the parse):

| fn | p99 tok | cap | parser evidence |
|---|---|---|---|
| event_poignancy | 4 | 64 | first-int (strict int() then `_extract_first_int`) |
| chat_poignancy | 4 | 64 | first-int |
| thought_poignancy | no calls | 64 | same parser family as the other poignancy fns |
| wake_up_hour | 307 | 614 | first-int; sampled responses lead with the hour |
| action_sector | 7 | 64 | option token, first line |
| action_arena | 2 | 64 | option token |
| action_game_object | 48 | 96 | option token |
| event_triple | 265 | 530 | first tuple; all 6 long/rambling calib_008 responses lead with it |
| act_obj_event_triple | 52 | 105 | first tuple |
| violation_check | 21 | 64 | one-line 4-key JSON |
| pronunciatio | stubbed | 64 | emoji; floor |

Everything else — bottom-scanning yes/no deciders (decide_to_talk p99 732,
decide_to_react 983, decide_if_norm_conflict 679, conflict_chat_reflect 638),
whole-string consumers (summaries, act_obj_desc), JSON whose truncation
silently drops data (norm_format), plans/schedules (new_decomp_schedule p99
3,407 — the largest measured), dialogue, and the entire unmeasured
norm_evaluate family (zero calib_008 calls because the seed pipeline was
stalled pre-Change-A) — gets a blanket 4096: pure runaway protection above
every measured p99. Unlisted fns keep CRSEC_NUM_PREDICT_DEFAULT (1024).

Unchanged: the CRSEC_NUM_PREDICT=0 kill-switch and the done_reason=="length"
truncated_* tripwire. calib_009 must run with caps ENABLED (leave
CRSEC_NUM_PREDICT unset) and watch truncated_* ≈ 0.
Data basis: `calib_artifacts/calib_008/calls.jsonl` per-fn length stats
(p50/p95/p99/max) computed by the attribution script; see the analysis table
in the Pass 2 session notes / length_stats.json.

### Phase 0 audit + Change D — metrics wiring (`norm/norm_evaluate.py`)

Phase 0 verdicts for calib_008's all-zero summary.json (static call sites +
dynamic evidence from the artifacts):

| metric | wired? | calib_008 ground truth | verdict |
|---|---|---|---|
| defection_* | yes (`calculate_defection_utility`; callers pass `persona.metrics`) | 0 defectors in the population | genuinely zero |
| violations_detected | yes (`process_violations` ← `getattr(persona,'metrics',None)`, non-None: set at reverie.py:169) | 25 checks (5 LLM + 20 cached), all verdicts `violation: False`; the 5 log hits of `"violation": true` are prompt echoes | genuinely zero |
| confront/gossip/ignore | yes (same site) | no violations → never ran | genuinely zero |
| norms_adopted/rejected | **NO — `log_norm_adoption` had zero call sites** | zero seeds created in 200 steps: all 15 NormSeedNode/ActNormNode prints are load-time bootstrap (log lines 5–67); saved norm_count/act_norm_count identical to base_ville_n10_with_norm for all 10 personas. Root cause: norm_format parser failed 100% (Change A), killing every seed pre-adoption | wiring gap AND genuinely zero |
| compliance_timeline | populated (state reader) | constant 1.0 / 15 active norms | see note below |

Compliance note: `_snapshot_compliance` reads `activation_state`/`validity_state`.
`validity_state` is set True at adoption and never updated anywhere — and it
FEEDS BACK into the sim (`normDatabase.retrieve_norms` filters on it), so
"wiring a compliance update" there would alter behavior, i.e. it is NOT
eligible for an observation-only hook. Defector comply/defect decisions are
already captured (wired defection_log → population_stats
defect/comply_decisions_this_step). Citizens have no explicit compliance
decision site — compliance is implicit in plan generation. No change made,
by design.

Change D therefore wires exactly the one missing hook: `_log_norm_adoption`
(guarded getattr/hasattr + try/except, observation-only) called immediately
after `norm_evaluate_check` at BOTH adoption decision sites in
norm_evaluate.py — the immediate path in `norms_evaluate` and the long-term
synthesis path in `run_long_term_norm_evaluate`. It logs accepted AND
rejected decisions; `norm.poignancy` at that point carries the utility score
on accept or the rejection code (-2 fact / -3 duplicate / -4 name) on
reject. Receiver-side adoption via conversation spreading flows through the
same `norms_evaluate` path (chat reflect → add_norm_seed → next-step
evaluation), so it is covered by the same hook. `init_evaluate` has no
callers (dead code) and load-time bootstrap adoption does not pass through
the instrumented sites — bootstrap loads are not adoption events.

Tests (4 new, in test_modifications.py): accepted logged, rejected logged,
raising collector cannot break adoption, absent `persona.metrics` is a no-op.
Suite: 122 passed, 1 deselected (pre-existing 462a1a6 failure).

Expected in a calib_009-like run (10 personas, no defectors, Change A live):
- `total_norms_adopted` / `total_norms_rejected` — nonzero as soon as any
  reflection survives norm_format (calib_008 had 71 reflections + 10 chat
  reflections that all died at the parser).
- `norm_adoption_log.json` — one entry per evaluation decision.
- Legitimately still zero in a defector-free baseline: defection_*,
  and violations/enforcement unless an adopted norm actually gets violated
  (violation checks fired only 25x after prefilter in calib_008).

---

## 2026-07-08 — Pass 3 on calib_009 data: unblock adoption, non-destructive evaluation, transactional synthesis

Branch `norm_deflection`, on top of `e61bb40`. Driven by calib_009 (200
steps, 24.3 s/step, 2,986 calls). Phase 0 verdicts that shaped the changes:

- Adoption chain: −1×188 in the adoption log == fail_safe_run_gpt_seeds_type_check_v2
  (564 calls = 188 loops × 3, retry counter == call count → 100% attempt
  failure). −2×11 (fact-consistency) and −3×11 (duplicate) are GENUINE parsed
  rejections (their 295/199 calls parsed with zero retries). norm_utility and
  the conflict check never executed — the chain died at the type check.
  seeds_type_check(v1), seeds_content_check and norm_utility(v2) are
  unreachable dead code: their prompt template files do not exist on disk.
- Active-norm "collapse 5→0 for 7 personas": REFUTED. The 7 citizens loaded
  with NO norm databases ("personal_norm_database.json could not find") and 0
  actives; actives were constant at 15 (3 entrepreneurs × 5) at steps 100 and
  200; specific_norm_deactive never executed (long-term synthesis never
  triggered — its trigger only decrements on successful adoption, and there
  were 0). The "collapse" misread the saved SEED database, where
  created-but-never-adopted seeds carry activation_state=false.
- 4 truncations on seeds_type_check_v2: the intended 4096 blanket cap doing
  its job — the model occasionally rambles (max response 18,710 chars ≈ 5.3k
  tokens); markers sit at the top, so truncated rambles still parse.
- violation_log norm_content=None: REFUTED — all 42 violation_log entries,
  42 enforcement entries and all persona observed_violations carry the real
  norm content. The pass-2 call site works; no change made (planned Change D
  dropped as a verified no-op).

### Change A (pass 3) — seeds_type_check_v2 parser (`norm/run_gpt_prompt_norm.py`)

Root cause: the template's final instruction line demands <"..."> wrapping
but its own EXAMPLES show the type in plain quotes; Qwen3 follows the
examples ('The type classification for INPUT is "descriptive"'), so the
parser's literal '<"' marker split failed on 564/564 calls. New module-level
`_parse_seeds_type_check`: three tolerant regexes (STEP 1 yes/no, STEP 2
correct/incorrect, type classification descriptive/injunctive) accepting
<">, plain quotes, bold or bare tokens; last occurrence wins (mirrors the
old split()[-1] semantics). No template change.

Siblings checked against calib_009 data: fact_consistency (295 calls,
199 yes/96 no, 0 no-parse) and duplicate_check (199 calls, 188 no/11 yes,
0 no-parse) parse cleanly — no changes. recognize_conflict_check and
specific_norm_utility have zero calib_009 calls (never reached); their
failure handling is covered by pass-3 Change B instead of speculative
parser edits.

Replay (tests/replay_calib_parsers.py, now covering the seed-evaluation
chain, run against BOTH bundles):
- calib_009: seeds_type_check_v2 564/564 (was 0), duplicate 199/199,
  fact_consistency 295/295; pass-2 fns 100% each (126/126, 38/38, 240/240,
  16/16, 24/24) — the pass-2 fixes held in the wild.
- calib_008 regression: unchanged (100/100/100/94.4/100; the 94.4 residual
  is the known single degenerate response × 5 retries).
Suite: 122 passed, 1 deselected (pre-existing 462a1a6 failure).

### Change B (pass 3) — failure ≠ rejection in the evaluation chain (`norm/norm_evaluate.py`)

norm_evaluate_check now returns save_tag ∈ {True, False, None}: None means a
stage produced NO parsed verdict (fail_safe / unparsed response), in which
case the seed is left pending (poignancy stays -1, re-evaluated at the next
trigger), `call_profiler.incr("eval_deferred_<stage>")` fires, and NO
adoption event is logged (_log_norm_adoption returns early on None).
Stage-level discrimination, all previously conflated with verdicts:

| stage | previous failure behavior | now |
|---|---|---|
| fact_consistency fail_safe | branded -2 (permanent rejection) | defer (`eval_deferred_fact_consistency`) |
| format rewrite → None | reject-this-round | defer (`eval_deferred_format_rewrite`) |
| duplicate fail_safe | treated as "not duplicate", chain continued | defer (`eval_deferred_duplicate_check`) |
| type_check fail_safe | rejected + logged with sentinel -1 (the 188) | defer (`eval_deferred_type_check`) |
| conflict fail_safe | treated as "conflict" → rejected | defer (`eval_deferred_conflict_check`) |
| utility [4,"fail_safe"] | silently ADOPTED with fabricated utility 4 | defer (`eval_deferred_utility`) |
| long-term utility False/misshapen | reject-this-round | defer (`eval_deferred_long_term_utility`) |

Parsed verdicts unchanged: name -4, fact-'no'-exhausted -2, duplicate-'yes'
-3, type STEP-1 'no' → reject with poignancy left -1 (pre-existing
semantics), conflict-'yes' → reject, parsed utility → adopt. NOTE for log
readers: from this pass on, a -1-scored rejection in norm_adoption_log is a
PARSED verdict (STEP-1 'no' / conflict 'yes'), never a parser failure —
failures no longer produce events at all.

The utility fns' [4,"fail_safe"] return contracts are untouched (tests pin
the length-2 failure shape); discrimination happens at the consumer via the
reason marker. 8 new tests (all-parsed identical behavior; per-stage
failure→defer; per-stage verdict→reject; deferral emits no metrics event).
Suite: 130 passed, 1 deselected.

### Change C (pass 3) — transactional long-term synthesis (`norm/norm_evaluate.py`)

Phase 0-B showed no erosion actually occurred in calib_009 (the long-term
path never even triggered), but the ordering hazard was real: on a
successful check the code deactivated the replaced specifics BEFORE adding
the replacement. run_long_term_norm_evaluate now:
- adds the verified replacement to the database first, then deactivates the
  specifics it replaces (deactivate-last); end state on the all-parsed path
  is byte-identical to before (both operations are independent state writes);
- increments `synthesis_aborted` on every failure path — classification
  fail, synthesis-check "stoped" (False / length mismatch), format→None per
  item, and a deferred (None) or rejected (False) replacement per item — and
  in all those cases the original active norms stay untouched.
3 new tests: success ordering (add_act before deactivate, trigger
decremented, no abort counter), rejected and deferred replacements leave
originals active + count aborts. Suite: 133 passed, 1 deselected.

### Change D (pass 3) — violation_log norm content: VERIFIED NO-OP, no commit

The premise ("norm_content: None") is refuted by the calib_009 data: all 42
violation_log entries, all 42 enforcement_log entries, and every
persona-side observed_violations record (Bob x30, Francisco x12) carry the
real norm content string and norm_id. The single log_violation call site
(violation_detection.py, `norm_content=getattr(norm_obj, "content", None)`)
resolves `.content` on live NormNode objects. Nothing to fix; no commit made
(fabricating a change to satisfy the commit list would be worse than
reporting the truth).

---

## 2026-07-09 — Pass 4: CRSEC-faithful base seeding, loud loader, no sentinel adoption events

Branch `norm_deflection`, on top of `1fc445a`. calib_010 artifacts are NOT
present on this clone — Change C (SpecificNormUtility attribution + parser,
needs the 87 captured norm_evaluation calls) is BLOCKED and deferred; A/B/D/E
land. Note: the committed base in THIS repo is already internally consistent
(3 entrepreneurs 5/5 with files, 7 citizens 0/0 without); the "every scratch
claims 5/5" inconsistency described for calib_010 lives in the cluster's
uncommitted working copy — one more reason the base must be regenerated by
script and committed from the cluster.

### Change A (pass 4) — `tools/seed_base_norms.py`

Standalone repair script making the base CRSEC-faithful: norm entrepreneurs
(Isabella Rodriguez, Tom Gomez per §4.1) get --norms-per personal norms via
the EXISTING creation flow (sys_prompt.txt + usr_prompt_v6.txt composed with
the persona's bootstrap scratch.json, llm_call(call_type="norm_creation"),
norm.creation._slice_json reused via import); every other persona has its
norm files removed and counts zeroed (the 3 accidental holders demoted).
- Validation before any write: exactly norm_1..norm_N keys; required fields
  per norm_save's schema (ID/type/content/subject/predicate/object/utility/
  activation_state/validity_state; related_desc + poi_reason defaulted to ""
  since the creation prompt doesn't emit them); type ∈ {descriptive,
  injunctive}; utility int 1..100; IDs renumbered; activation/validity forced
  true. Up to 3 generation attempts, then a loud abort — never writes garbage.
- Writes BOTH files (identical content), sets scratch norm_count /
  act_norm_count, and (default on, --no-update-identity to skip) syncs
  scratch identity: entrepreneurs → "entrepreneur", demoted accidents →
  "citizen" (defectors untouched) so population_stats stays truthful.
- `--dry-run` prints the plan + current-state verification table.
- `--from-file <json>` seeds deterministically from a hand-authored
  {persona: {norm_1: {...}}} file validated by the same schema — the
  reproducible option for the methods chapter.
- Ends with a verification table (files present / entry counts / scratch
  counts / identity) and a non-zero exit on any mismatch.

Generation mode REQUIRES a running Ollama with the primary model pulled —
run on the CLUSTER (LLM imports are lazy; --dry-run/--from-file work
anywhere). Validated locally: --dry-run against the real base (read-only, all
rows [ok]); full --from-file run against a temp copy (entrepreneurs seeded
5/5 both files, accidents stripped+demoted, table all-[ok], exit 0); loader
round-trip on the seeded output (NormDatabase loads 5 seeds + 5 actives, all
activation_state true).

### Change B (pass 4) — loader never silently zeroes (`norm/normDatabase.py`)

Three mismatch cases now shout a stderr banner (persona name, file path,
expected vs found counts, pointer to tools/seed_base_norms.py), increment
`norm_load_mismatch`, and raise RuntimeError under CRSEC_STRICT_NORM_LOAD=1:
1. seed file missing while scratch norm_count > 0;
2. validity file missing while scratch act_norm_count > 0;
3. a file with FEWER norm_i entries than scratch claims — previously this
   was not merely silent, it CRASHED with an unhandled KeyError; now it loads
   what exists, then warns.
Default behavior otherwise unchanged: consistent bases load exactly as
before, zero-count personas with no files stay silent. 5 new tests (missing
files warn ×2 counters, short file partially loads, consistent base silent,
zero-expected silent, strict mode raises). Suite: 138 passed, 1 deselected.

### Change D (pass 4) — violation content: premise NOT reproducible; regression test added

Traced end-to-end: there is exactly ONE violations producer
(detect_violations builds {"norm": <NormNode>} from the observer's act_norm
values; process_violations reads getattr(norm, "content", None); NormNode
carries .content; the prefilter only reads attributes). The calib_009 bundle
on this clone has ALL 42 violation_log entries, all 42 enforcement entries
and every observed_violations record populated with real content — the
"norm_content: None across calib_009/010" claim does not reproduce here
(calib_010 absent; if its 18 entries really are None on the cluster, diff
that clone's violation_detection.py / metrics.py against ce39502+, because
this code demonstrably populates content).

Added TestViolationContentLivePath: a REAL NormNode driven through the live
detect_violations → process_violations path (patched violation-check LLM)
must land its content string in metrics.log_violation, log_enforcement AND
scratch.observed_violations. Passes against current code; any refactor that
drops the attribute now fails loudly. No production change made.

### Change E (pass 4) — adoption events carry parsed verdicts with real utilities only

Code-level trace at current HEAD (calib_010 bundle absent locally, so the
"19 sentinel events" could not be replayed): after pass-3 Change B, NO
failure path can emit an adoption event (all defer, unit-proven). The only
remaining sentinel-scored events were GENUINE parsed rejections logged with
the norm object's bookkeeping codes: -4 name_check, -2 fact-'no'-exhausted,
-3 duplicate-'yes', -1 type-STEP1-'no' or conflict-'yes'. Those codes stay
on the norm object (they gate re-evaluation) but no longer leak into the
metrics stream:
- norm_evaluate_check marks the verdict stage on the seed
  (norm.reject_stage ∈ name_check / fact_consistency / duplicate_check /
  type_check_not_norm / conflict_check; cleared to None on acceptance);
- _log_norm_adoption sends utility_score = real parsed utility for accepted
  events and utility_score=None + reject_stage for rejections;
- MetricsCollector.log_norm_adoption gains the reject_stage field (default
  None; event dict now includes it).
Observation-only: sim state and control flow untouched; the poignancy
sentinels on the norm objects behave exactly as before.
Tests: parsed rejections carry the right stage (4 stages asserted), accepted
path clears the mark, rejected events log null utility + stage. Suite: 140
passed, 1 deselected.

### Change C (pass 4) — BLOCKED: calib_010 bundle not on this clone

The SpecificNormUtility attribution (+ explicit prompt_fn= override on
llm_call) and the parser fix both require the 87 captured
call_type="norm_evaluation" responses from calib_010/calls.jsonl. Evidence-
based rule: no parser gets rewritten against guessed output shapes. Deferred
until the bundle is synced; pass-3's eval_deferred_utility guard means the
current failure mode is a benign defer-and-retry, not corruption.

---

## 2026-07-09 — Pass 4.5: SpecificNormUtility attribution + tolerant parser + routing/cap

Branch `norm_deflection`, on top of `ec617a9` (cluster base-seeding commit,
pulled as a clean fast-forward). Evidence: calib_011 bundle; its calls.jsonl
holds 341 call_type="norm_evaluation" responses (254 from calib_011 + the 87
calib_010 carryover in the appended log), all on qwen3:32b.

Diagnosis: the utility template's examples show "- OUTPUT: <score>. Because
<reason>." and Qwen3 answers in markdown — "- **OUTPUT**: **30**. Because
..." (sometimes "- **Because**:" on its own line). The literal
`split("OUTPUT: ")` + `int()` parse failed on 328/341 captured calls (13
parsed = the plain-format minority), every failure deferring the seed →
adoption ran at ~3% of its rate and the 254 calls (8.5s avg, 2,148s total)
profiled as prompt_fn "unknown".

1. Attribution — `llm_call(..., prompt_fn=None)`: explicit override of the
   stack-walk for direct callers with no run_gpt_* frame. Labeled:
   SpecificNormUtility → "run_gpt_specific_norm_utility"; defection_engine
   → "run_gpt_defection_assessment" / "run_gpt_defector_norm_utility";
   creation.Creation → "run_gpt_norm_creation". Profiler + num_predict +
   routing now see real names.
2. Parser — module-level `_parse_norm_utility_response`: last OUTPUT marker
   (bold-tolerant) → first int in the segment → reason with markdown/bullet/
   "Because:" peeling (keeps the leading "Because", as the old parser did);
   fallback when no marker: a "<int>. <text>" score-LINE scan (deliberately
   never a bare first-int grab — the INPUT echo line contains times).
   Refusals ("**N/A**. Because...") raise → the pass-3 defer path handles
   them. json_schema was considered and rejected: it changes the RESPONSE
   distribution, which cannot be validated against captured data offline.
   Replay: 340/341 = 99.7% (the 1 residual is that genuine N/A refusal);
   calib_008/009 regression unchanged; the accumulated calib_011 log also
   re-validated every earlier parser in the wild (type_check_v2 928/935 —
   the 7 misses match the run's own 7 retries — all others 100%).
3. Routing + cap — "run_gpt_specific_norm_utility" joins
   NORM_PRIMARY_OVERRIDE_FNS (norm-reasoning tier consistency with
   decide_if_norm_conflict / norm_reflect_from_thoughts, both validated on
   the 30b in calib_011); CRSEC_NORM_ON_PRIMARY=0 restores the 32b exactly
   via the tier-3 call_type fallback. Cap 2712 (= 2 × p99 of 1,356 tok;
   measured p50 894 / p95 4,011 / p99 4,745 / max 4,927 chars): the score
   leads the response (top-reader), and the old "unknown" default of 1024
   truncated 4 responses. get_defector_norm_utility: attribution only —
   zero captured samples, parser and 32b routing untouched.

Tests: +3 (markdown-shape parser unit test incl. refusal raise; prompt_fn
override routes to PRIMARY with cap 2712; flag=0 restores 32b). Suite: 143
passed, 1 deselected (pre-existing 462a1a6 failure).

---

## 2026-07-11 — day-1 defection gating + seeded defector norms (branch mindwell_defectors)

calib_013_defector came back all-green EXCEPT defection_log = 0: the engine
was only reachable through the "New day" planning branch (second sim
morning), none of the behavior-shaping act-norm injections were gated, and
defectors started with zero norms anyway.

1. `decide_defection_cached` (norm/defection_engine.py): one LLM-backed
   decision per (defector, norm, sim-day), keyed on scratch.name /
   norm.content / curr_time.date(); cache hits skip the LLM and duplicate
   metrics logging; non-defectors short-circuit without touching the cache;
   curr_time=None falls back uncached. Defector-ness via scratch.is_defector()
   (agent_type fallback) — same source of truth as the gate sites.
2. Gated ALL behavior-shaping act-norm injection sites with the
   norm_compliance pattern (defect → norm excluded from that persona's
   prompt): plan.py generate_first_daily_plan / generate_hourly_schedule /
   generate_task_decomp / generate_task_decomp_v2; run_gpt_prompt.py
   run_gpt_prompt_summarize_ideas + run_gpt_prompt_generate_next_convo_line
   (both independently read act_norm to build prompts — gated identically).
   Retrofitted the 3 existing norm_compliance.py sites to the cached helper.
   Evaluation/observation paths (norm_evaluate, violation_detection, metrics)
   left ungated by design: defectors still know/evaluate/spread/observe.
3. `--seed-norms-from` in create_defector_personas.py FLIP mode + wired into
   tools/build_exp1_condition_bases.sh: flipped defectors get the union of
   Isabella's + Tom's norms (argument order, re-keyed norm_1..norm_10 with
   ID 1..10, all other fields verbatim, both norm files, scratch counts set
   to 10 → CRSEC_STRICT_NORM_LOAD passes, verified by live strict load).
   Build verification extended: defectors 10/10 files+scratch, entrepreneurs
   5/5, citizens 0/0, n=10 — all three bases rebuilt and asserted.

Verified: baseline base_ville_n10_with_norm byte-identical; flipped-persona
scratch diff is exactly 13 keys (the 11 from the flip + norm_count +
act_norm_count); merged files: Isabella 1-5 then Tom 6-10 verbatim, all
active+valid; every gate sits inside is_defector() (grep-confirmed) so
citizen/entrepreneur behavior in baseline runs is bit-identical; 3 new cache
unit tests (same-day hit, day rollover, per-norm keying, non-defector
no-cache, no-clock uncached). Suite: 146 passed, 1 deselected; replay green
vs calib_008/009/011. Expected day-1 cost in cond C: 5 defectors x 10 norms
= ~50 cached decisions.

---

## 2026-07-12 — overnight hardening: embedding guard + text-bloat caps (branch mindwell_defectors)

exp1_base_r1_c2 crashed at its first sim-midnight: _long_term_planning
joined all daily_req items into a plan-thought, get_embedding(thought)
exceeded the embed model's context window, ollama raised ResponseError
"context length" unhandled → run aborted. Three guards, all no-ops for
well-sized text (identical inputs → identical outputs under the caps):

1. get_embedding (gpt_structure.py, the single funnel for all 9 embedding
   call sites): clips input to CRSEC_EMBED_MAX_CHARS (default 8000) with a
   loud "[EMBED CLIP] <orig> -> <new> chars" line; on a context-length error
   halves and retries (max 3, loud each time) then re-raises — visible
   failure over silent junk. Newline-strip / temp_sleep / "this is blank"
   behavior unchanged.
2. daily_plan_v2 cleanup (run_gpt_prompt_norm.py): _cap_plan_items on BOTH
   parse paths — at most 20 items, each ≤200 chars, "[PLAN CAP]" on trigger.
   The line-fallback could turn every line of a rambling response into a
   plan item; the joined items are embedded and injected into all 24
   hourly-schedule prompts. Worst case now ≤4000 chars, under the embed cap.
3. revise-identity cleanups: _cap_identity_text (1000 chars, cut at the last
   sentence boundary past the halfway point, "[IDENTITY CAP]" on trigger) on
   run_gpt_revise_identity_currently and _daily_plan_req — their tails become
   scratch.currently / scratch.daily_plan_req and inflate every subsequent
   prompt via the identity stable set. Parsing logic untouched.

Verification: all embedding calls funnel through get_embedding (perceive ×2,
converse ×2, reflect ×3, retrieve ×1, plan ×1 — the plan.py:607 site is the
crash site; zero direct ollama.embeddings uses elsewhere).
ChatGPT_single_request's 4 call sites all live in plan.py revise_identity,
which is DEAD CODE (only caller commented out at plan.py:580 in favor of
revise_identity_v2) — no bare production path; left as-is per scope.
Tests: 1 new comprehensive test (clip, halving retry with call-length trace,
unrelated-error re-raise, 50-line ramble → 20×≤200 through the real wrapper
+ cleanup, 5000-char identity tail → ≤1000 at a sentence boundary,
under-cap passthroughs). Suite 147 passed, 1 deselected; replay green vs
calib_008/009/011. PROMPT_FN_NUM_PREDICT, routing, and all protected files
untouched (r1 comparability preserved).

---

## 2026-07-12 — tolerant parser for the defector norm-utility path (branch mindwell_defectors)

calib_014_defector passed all 7 defection gates but eval_deferred_utility
hit 93 (gate ≈0): condition sims route defector norm-utility evaluations to
get_defector_norm_utility — exercised for the first time in calib_014 — and
its parser demanded the literal 'OUTPUT: <int>.' scaffold. Same failure
class as the old SpecificNormUtility bug (247/254). Net effect unfixed:
defectors could never adopt new norms, biasing adoption rates by condition.

New module-level `_parse_defector_utility` (defection_engine.py), acceptance
in priority order: (1) OUTPUT marker, bold/backtick-tolerant, followed by an
int; (2) an int at the start of a line or after a Score:/Rating: label;
(3) first int in the LAST non-empty line, then anywhere. Every stage accepts
only 0..100. Reason = text after the score's sentence delimiter, else from
"Because" anywhere in the response, clipped to 300 chars, "no reason given"
when absent. Nothing parseable → the caller's [4, "fail_safe"] survives, so
real LLM failures still defer via norm_evaluate_check's marker check.
Prompt template unchanged; calculate_defection_utility's decision parsing
(51/51 real reasoning in calib_014) untouched; responses the old parser
handled produce identical scores/reasons (covered by the pre-existing
shape test, which still passes).

7 new tests: exact template shape, bare score line, bold+em-dash, prose
preamble with Score: label, inline "95 out of 100" sentence, reason
clip/never-empty, garbage/empty/non-numeric → fail_safe through the caller.
Suite 154 passed, 1 deselected; replay green vs calib_008/009/011.
Expected effect: defector utility evals parse → eval_deferred_utility back
to ≈0 in calib_015.
