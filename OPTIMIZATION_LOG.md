# Optimization Log — LLM-call reduction pass (norm_deflection)

Goal: make a 17,280-step, 10-agent run feasible (~55–85 s/step → ~5 s/step
budget) by eliminating redundant LLM calls. Simulation behavior unchanged on
the default flags except where noted (Steps 2–3, flag-gated).

## Step 0 — Instrumentation (no behavior change)

- **`reverie/backend_server/call_profiler.py`** (new): global profiler.
  Every Ollama call in `llm_router.llm_call` is timed and attributed to the
  nearest `run_gpt_*` function on the call stack. Named counters track cache
  hits and pre-filter skips.
- Dumped to **`<sim_folder>/profile.json`** on every `save`/`fin`
  (`reverie.py:ReverieServer.save`), and to `./profile.json` at interpreter
  exit as a fallback.
- Schema: `total_llm_calls`, `total_llm_seconds`,
  `per_function: {fn: {count, seconds}}`, `counters: {...}`.

## ⚠ Temperature audit finding (read this first)

The premise "temp=0 everywhere" was **not true in code**. All
`gpt_param = {"temperature": ...}` dicts in `run_gpt_prompt*.py` are
**vestigial** — `llm_router.llm_call` never received any options, so Ollama
applied the **model default** (qwen3 ships ~0.6) on *every* call.

Fix: `llm_router.py` now pins per-request options at the single choke point:

| Env var | Default | Meaning |
|---|---|---|
| `CRSEC_TEMPERATURE` | `0` | sampling temperature for all calls |
| `CRSEC_NUM_CTX` | `16384` | per-request context window cap (Step 1d) |

This is the one change that can alter outputs vs. previous runs — but it
makes runs deterministic, which is what the memoization (and the paper's
reproducibility) relies on. Set `CRSEC_TEMPERATURE` to restore sampling.

Nominal temp>0 `gpt_param`s exist (norm_reflect_from_thoughts 0.5,
norm_format 0.5, conflict_chat_reflect 0.5, chat_norms_summarize 0.5,
active_norms_classfication_v2 1, daily_plan_v2 1, and several in
run_gpt_prompt.py); **none of them are memoized** — only temp-0-nominal
functions are (see Step 1).

## Step 1 — Lossless optimizations (default ON)

All memoization gated by `CRSEC_MEMOIZE` (default `1`; set `0` to disable).
Caches are in-process dicts, max ~50k entries (FIFO-ish eviction), values
deep-copied on put/get; fail-safe results are never cached.

- **1a** `run_gpt_prompt_violation_check` memoized on
  `(observer_name, event_desc, norm.content)`
  (`norm/run_gpt_prompt_norm.py`). Dominant win: actions persist 30–360
  ticks, so identical checks repeated hundreds of times.
- **1b** Memoized in `persona/prompt_template/run_gpt_prompt.py`:
  - event/thought/chat poignancy, keyed
    `(name, get_str_iss(), description)` — the ISS is in the prompt and can
    change (revised "currently"), so it is part of the key to stay lossless.
  - `run_gpt_prompt_event_triple`, keyed `(persona.name, description)`.
  - `run_gpt_prompt_pronunciatio`, keyed `(description)` (when not stubbed).
- **1c** Headless emoji stub: `CRSEC_HEADLESS=1` →
  `run_gpt_prompt_pronunciatio` returns `"💬"` with no LLM call
  (default: **off**; pronunciatio is frontend-only cosmetics).
- **1d** `num_ctx` capped per request (see table above).
- **1e** Thinking-mode audit: **all simulation-path chat calls** route through
  `llm_router.llm_call`, which sends both `/no_think` (system msg) and
  `think=False`. ✔ suppressed everywhere. Exceptions (not thinking-mode
  issues, listed for completeness):
  - `gpt_structure.get_embedding` → `ollama.embeddings` (no thinking mode).
  - `SpecificNormUtility.specific_norm_utility`
    (`norm/run_gpt_prompt_norm.py:889`, called from `norm_evaluate.py:469`)
    still calls legacy `openai.ChatCompletion` — `openai` isn't imported, so
    it NameErrors into its `except` and returns `False`. Pre-existing,
    already flagged in PORTING_FLAGGED.md §8; NOT touched by this pass.
  - `compress_sim_storage_norm.py`, `test.py`: offline tools, legacy openai.

## Step 2 — Tiered model routing (flag-gated, default ON)

| Env var | Default |
|---|---|
| `CRSEC_TIERED_ROUTING` | `1` (set `0` to disable) |
| `CRSEC_SMALL_MODEL` | `qwen3:4b` |

Routing is keyed on the calling `run_gpt_*` function name (same stack walk
as the profiler). Only these cosmetic/noise-tolerant calls go small:
`run_gpt_prompt_pronunciatio`, `run_gpt_prompt_event_triple`,
`run_gpt_prompt_event_poignancy`, `run_gpt_prompt_thought_poignancy`,
`run_gpt_prompt_chat_poignancy`. Everything else keeps the existing
qwen3:30b-a3b / qwen3:32b routing.

Note: memoization keys do not include the model, so caches are per-process —
flip `CRSEC_TIERED_ROUTING` between runs, not mid-run.

## Step 3 — Violation-check pre-filter (flag-gated, default ON)

`CRSEC_VIOLATION_PREFILTER` (default `1`; set `0` to disable), in
`norm/violation_detection.py:detect_violations`, before any LLM call:

1. **Benign-pattern skip**: event descriptions containing
   `is chat with`, `chatting with`, `conversing about`, `is idle`,
   `sleeping`, `<waiting` are skipped for all norms.
2. **Stem-overlap gate**: remaining events need ≥1 shared content-word stem
   with the norm (stems built from the norm's subject/predicate/object,
   falling back to `norm.content`; cached on the norm object). Norms with no
   extractable stems **fail open** (always LLM-checked).

Audit counters in `profile.json`: `prefilter_skipped_benign`,
`prefilter_skipped_no_overlap`, `prefilter_llm_checked` (plus
`violation_check_cache_hit`, `prompt_cache_hit`,
`pronunciatio_headless_stub`).

This step is behavior-affecting by design (events the filter skips would
previously have been LLM-judged — overwhelmingly "no violation").

## Flag summary

| Flag | Default | Effect |
|---|---|---|
| `CRSEC_MEMOIZE` | 1 | Step 1a/1b caches |
| `CRSEC_HEADLESS` | off | pronunciatio stub "💬" |
| `CRSEC_NUM_CTX` | 16384 | Ollama num_ctx |
| `CRSEC_TEMPERATURE` | 0 | Ollama temperature (pinned!) |
| `CRSEC_TIERED_ROUTING` | 1 | small-model routing |
| `CRSEC_SMALL_MODEL` | qwen3:4b | small model id |
| `CRSEC_VIOLATION_PREFILTER` | 1 | violation pre-filter |

## How to A/B

Baseline: `CRSEC_MEMOIZE=0 CRSEC_TIERED_ROUTING=0 CRSEC_VIOLATION_PREFILTER=0`
(and optionally `CRSEC_TEMPERATURE=0.6` to mimic the old unpinned default).
Run 50 steps, `fin`, compare `<sim_folder>/profile.json` `total_llm_calls` /
`total_llm_seconds` and `per_function` against an all-defaults run. Target:
≥5× fewer total LLM calls.

## Expected impact

- Violation checks dominated cost (perceived_events × active_norms per agent
  per tick). Pre-filter + memoization should remove the vast majority; only
  novel non-benign event/norm pairs with stem overlap pay an LLM call.
- Poignancy/triple memoization removes repeats over an action's 30–360-tick
  lifetime; small-model routing cuts the cost of the remaining misses.
- Pronunciatio: zero calls in headless runs.

## Tests

`reverie/backend_server/tests/test_optimizations.py` (20 tests): cache hits
return identical object shapes and a single underlying call; fail-safes are
not cached; mutation can't poison the cache; ISS change misses the cache;
pre-filter never raises on malformed events and respects its flag; stub
returns `str`; routing/options verified. Full suite: **99 passed**.
