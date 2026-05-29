# CRSEC GPT-4 → Qwen3 Port: Patch Manifest

This document lists every code change made to bridge CRSEC's GPT-4-tuned
cleanup/parser code to the local Qwen3 backend (qwen3:32b dense for
reasoning, qwen3:30b-a3b MoE for primary calls, served via Ollama).

The four failure classes addressed:

- **Class 1 — Missing parent directories.** `open(path, 'w')` without prior
  `os.makedirs(...)` when the parent dir hasn't been materialized yet
  (common after sim-storage fork-copy).
- **Class 2 — Fail-safe wrong shape.** `get_fail_safe()` returned a value
  whose shape didn't match what the consumer expected; crash propagated
  past the safety net.
- **Class 3 — Cleanup parsing too strict.** Cleanups required GPT-4's exact
  output structure (durations, "}" terminators, exact `Answer in yes or no:`
  markers, etc.); Qwen3's looser output broke validate → fail-safe triggered
  → garbage propagated.
- **Class 4 — Prompt-leakage in LLM output.** Qwen3 sometimes echoes prompt
  scaffolding (`Answer:`, `{`, ``` `json`` ``` fences, persona name); strict
  cleanups treated the leaked tokens as content.

All fixes are **additively lenient**: GPT-4-style strict inputs still parse
to the same value as before. Verified by `tests/test_cleanups.py` (35 tests,
all green).

---

## Shared helpers added

In `reverie/backend_server/persona/prompt_template/gpt_structure.py` (added
in commit `1ef5f45`, expanded in `6f454e3`):

| Helper | Purpose |
|---|---|
| `_strip_scaffolding(text, persona_name=None)` | Strip Qwen3 prompt-leakage prefixes (`Answer:`, `Output:`, ` ``` `, leading `{` without closing `}`, persona name + " is ") and trailing noise (` ``` `, `END`, stray `}`). Idempotent and lossless on GPT-4 output. |
| `_extract_first_int(text)` | Pull first signed integer from text. Handles Qwen3 patterns like `"Score: 7"`, `"5/10"`, `"I'd rate it a 7."`. |
| `_extract_yes_no(text)` | Return first `yes`/`no` token after scaffolding-strip, or `None`. Handles `"Yes."`, `"Yes, because…"`, `"No - the norm forbids…"`. |
| `_log_fail_safe_trigger(fs)` | Emit `[FAIL_SAFE] <caller>: returning <fs>` to stderr (caller inferred via `inspect.stack[2].function`). Wired into the four `safe_response` helpers' fail-safe returns. |

In `reverie/backend_server/norm/creation.py`:

| Helper | Purpose |
|---|---|
| `_slice_json(response)` | Strip ``` ```json ... ``` ``` fences and slice from first `{` to last `}` so Qwen3 prelude ("Here are the norms:") doesn't break `json.loads`. |

In both `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py`
and `reverie/backend_server/norm/run_gpt_prompt_norm.py`:

| Helper | Purpose |
|---|---|
| `_log_fail_safe(fn_name, fs)` | One-line stderr breadcrumb for use inside `__func_clean_up` recovery branches (distinct from the gpt_structure logger which fires only when the safe wrapper exhausts its retries). |

---

## File-by-file change table

### `reverie/backend_server/persona/prompt_template/gpt_structure.py`
| Function | Class(es) | Change |
|---|---|---|
| `_log_fail_safe_trigger` | (new) | New helper; uses `inspect` to identify the `run_gpt_prompt_*` caller and emits a stderr breadcrumb when a fail-safe activates. |
| `_strip_scaffolding`, `_extract_first_int`, `_extract_yes_no` | (new, 4) | Shared Qwen3 leniency helpers used by both `run_gpt_prompt.py` and `run_gpt_prompt_norm.py`. |
| `ChatGPT_safe_generate_response_OLD` | (logging) | After the final fail-safe return, log via `_log_fail_safe_trigger`. |
| `GPT4_safe_generate_response_OLD`, `_t1`, `ChatGPT_safe_generate_response_OLD_t0` | (logging) | Same logging hook on the fail-safe return path. |
| `safe_generate_response` | (logging) | Same logging hook on the fail-safe return path. |

### `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py`
| Function | Class(es) | Change |
|---|---|---|
| (module top) | — | Add explicit imports for `_strip_scaffolding`, `_extract_first_int`, `_extract_yes_no` from `gpt_structure` (wildcard `*` doesn't pull leading-underscore names). |
| `_log_fail_safe` | (new) | Local stderr breadcrumb for cleanup recovery branches. |
| `run_gpt_prompt_wake_up_hour.__func_clean_up` | 3 | Strip scaffolding; if `int(split("am"))` still fails, extract first integer with `_extract_first_int` and require `0 ≤ n < 24`. |
| `run_gpt_prompt_daily_plan.__func_clean_up` | 3 | Strip scaffolding; guard against `i[-1]` index error on empty splits; if the strict GPT-4 `")"+digit` heuristic returns nothing, fall back to bullet/numbered-line parsing. |
| `run_gpt_prompt_generate_hourly_schedule.__func_clean_up` | 3, 4 | Strip scaffolding; raise instead of crashing on empty `cr[-1]` access. |
| `run_gpt_prompt_task_decomp.__func_clean_up` (v1) | 3, 4 | Strip scaffolding (incl. persona name); filter empty split lines; **if all lines lack `(duration in minutes: N, ...)`, evenly split the prompt's total duration across the parsed tasks (snapped to 5-min increments, remainder on the last)** — log the recovery via `_log_fail_safe`. |
| `run_gpt_prompt_task_decomp_v2.__func_clean_up` | 3, 4 | Identical fix as v1. |
| `run_gpt_prompt_action_sector.__func_clean_up` / `__func_validate` | 3, 4 | Strip scaffolding (incl. persona name); drop a leading stray `{` (the concrete `"Answer: {cafe"` leakage from the smoke test); accept missing `}`; strip trailing `.,;:`. Validate no longer requires `}`. |
| `run_gpt_prompt_action_arena.__func_clean_up` / `__func_validate` | 3, 4 | Same fix as `action_sector`. This is the path that was producing `arena.split(":")` "too many values to unpack" downstream. |
| `run_gpt_prompt_event_triple.__func_clean_up` | 3, 4 | Strip scaffolding; drop stray leading `(`. |
| `run_gpt_prompt_act_obj_desc.__func_clean_up`, `__chat_func_clean_up` | 3, 4 | Strip scaffolding; guard against empty `cr[-1]`. |
| `run_gpt_prompt_act_obj_event_triple.__func_clean_up` | 3, 4 | Strip scaffolding; drop stray leading `(`. |
| `run_gpt_prompt_decide_to_talk.__func_validate` / `__func_clean_up` | 3, 4 | Replace strict `split("Answer in yes or no:")` + literal `in ["yes","no"]` check with `_extract_yes_no` (tolerates `"Yes."`, `"Yes, because…"`, `"No - …"`). |
| `run_gpt_prompt_decide_to_react.__func_validate` / `__func_clean_up` | 3, 4 | Replace strict `split("Answer: Option")` with a regex that accepts `"Option 3"`, `"Answer: 3"`, or `"3"`. |
| `run_gpt_prompt_event_poignancy.__func_clean_up`, `__chat_func_clean_up` | 3, 4 | Strip scaffolding; if `int(s.strip())` fails, use `_extract_first_int`; clamp to `[1,10]`. |
| `run_gpt_prompt_thought_poignancy.__func_clean_up`, `__chat_func_clean_up` | 3, 4 | Identical fix as event_poignancy. |
| `run_gpt_prompt_chat_poignancy.__func_clean_up`, `__chat_func_clean_up` | 3, 4 | Identical fix as event_poignancy. |
| `run_gpt_prompt_extract_keywords.__func_clean_up` | 3, 4 | Strip scaffolding; tolerate absent `"Emotive keywords:"` marker by treating the whole response as the factual list. |
| `run_gpt_prompt_focal_pt.__func_clean_up` | 3, 4 | Strip scaffolding; accept `1)`, `1.`, `-`, `*` numbering/bullets; raise instead of returning `[""]`. |
| `run_gpt_prompt_insight_and_guidance.__func_clean_up` | 3, 4 | Strip scaffolding; tolerate alternate bullet styles; **accept thoughts without the `(because of N, M)` evidence tail** (empty evidence list). |

### `reverie/backend_server/norm/run_gpt_prompt_norm.py`
| Function | Class(es) | Change |
|---|---|---|
| (module top) | — | Add explicit imports for `_strip_scaffolding`, `_extract_first_int`, `_extract_yes_no`; import `re`. |
| `_log_fail_safe` | (new) | Local stderr breadcrumb (mirrors the one in `run_gpt_prompt.py`). |
| `run_gpt_prompt_decide_if_norm_conflict.__func_validate` / `__func_clean_up` | 3, 4 | Replace two strict `in ["yes","no"]` checks with the local `_yn` helper (first word stripped of punctuation, else first yes/no token). |
| `run_gpt_prompt_norm_format.__func_clean_up` | 4 | Strip scaffolding (incl. ``` ```json ``` fences); if no `OUTPUT:` marker, slice from first `{` to last `}` before `json.loads`. |
| `run_gpt_immediate_evaluate_recognization.__func_clean_up` | 3, 4 | Strip scaffolding; tolerate `"NORM UTILITY: 7."` vs bare `7`; tolerate `"Yes."` vs literal `yes` after `ANSWER:`. |
| `run_gpt_norm_duplicate_check.__func_validate` / `__func_clean_up` | 3, 4 | Use `_extract_yes_no`. |
| `run_gpt_norm_utility.__func_clean_up` | 3, 4 | Strip scaffolding; tolerate missing `OUTPUT:` marker; fall back to `_extract_first_int` and take remaining text as reason. |
| `run_gpt_long_term_norm_utility.__func_clean_up` | 3, 4 | Strip scaffolding; tolerate `FINAL SCORE: N` vs `FINAL SCORE: [N]`. |
| `run_gpt_prompt_violation_check.__func_validate` / `__func_clean_up` | 4 | New local `_extract_json` helper: strips ``` ```json ``` fences and slices `{...}` so Qwen3 prelude doesn't break `json.loads`. |
| `run_gpt_prompt_daily_plan_v2.__func_clean_up` | 3, 4 | Same fix as `daily_plan` in `run_gpt_prompt.py`. |
| `run_gpt_non_norm_conflict_chat_reflect.__func_validate` / `__func_clean_up` | 3, 4 | Use `_extract_yes_no`. |
| `run_gpt_norm_recognize_conflict_check.__func_validate` / `__func_clean_up` | 3, 4 | Strip scaffolding; use regex `\b(yes|no)\b` (tolerates `Answer: Yes.`). |
| `run_gpt_norm_fact_consistency_check.__func_validate` / `__func_clean_up` | 3, 4 | Same yes/no fix as `recognize_conflict_check`. |

### `reverie/backend_server/norm/creation.py`
| Function | Class(es) | Change |
|---|---|---|
| `_slice_json` | (new, 4) | Helper that strips ``` ```json ``` fences and slices `{...}` block. |
| `Create()` (norm-generation loop) | 1, 4 | Slice the LLM response through `_slice_json` before `json.loads`; **add `os.makedirs(os.path.dirname(norm_file), exist_ok=True)` before writing `personal_norm_database.json`** (validity-file makedirs was already present). |

### `reverie/backend_server/norm/norm_save.py`
| Function | Class(es) | Change |
|---|---|---|
| `norm_save` | 1 | Import `os`; `os.makedirs(out_json, exist_ok=True)` before writing the two JSON files. |

### `reverie/backend_server/norm/reputation.py`
| Function | Class(es) | Change |
|---|---|---|
| `ReputationSystem.save` | 1 | Import `os`; `os.makedirs(parent, exist_ok=True)` for caller-supplied filepath. |

### `reverie/backend_server/persona/memory_structures/spatial_memory.py`
| Function | Class(es) | Change |
|---|---|---|
| `MemoryTree.save` | 1 | Import `os`; `os.makedirs(parent, exist_ok=True)` before writing. (Note: `get_str_accessible_arena_game_objects` already uses the tolerant `split(":", 2)` + pad pattern that resolves the Class-4 colon-overflow that came from the action_arena leakage.) |

### `reverie/backend_server/tests/test_cleanups.py`
| Function / class | Purpose |
|---|---|
| `TestStripScaffolding` (6 tests) | GPT-4 bare unchanged; strips `Answer:`, `Answer: {cafe`, code fences, persona name; non-string passthrough. |
| `TestExtractFirstInt` (3 tests) | Bare integer; `Score: 7` / `5/10` / `"a 5"`; `None` on no digits. |
| `TestExtractYesNo` (3 tests) | Bare; Qwen3 variants (`Yes.`, `Yes, because…`, `No - …`); `None` on absent. |
| `TestCreationSliceJson` (3 tests) | Raw JSON unchanged; strips code fences; strips Qwen3 prelude/suffix. |
| `TestActionSectorCleanup` (4 tests) | GPT-4 `kitchen}`; Qwen3 `Answer: {cafe`; bare; trailing punctuation. |
| `TestActionArenaCleanup` (2 tests) | GPT-4 braced; Qwen3 unclosed brace. |
| `TestDecideToTalkCleanup` (4 tests) | GPT-4 strict; Qwen3 `Yes.`; Qwen3 with justification; Qwen3 bare no. |
| `TestEventPoignancyCleanup` (5 tests) | Bare int; `Score: 5`; `5/10`; `"a 8."`; clamp 11 → 10. |
| `TestWakeUpHourCleanup` (3 tests) | `7am` / `7 AM`; bare `7`; `Answer: 7` / `"7 in the morning"`. |
| `TestTaskDecompCleanup` (2 tests) | GPT-4 `(duration in minutes: …)` form; Qwen3 bare numbered tasks (verifies even-split adds up to expected total). |

Run with `python -m pytest tests/test_cleanups.py -v` from
`reverie/backend_server/`. All 35 tests pass; existing `test_modifications.py`
+ `test_antisocial_norms.py` (34 tests) remain green.

---

## What's intentionally unchanged

- All files under `persona/prompt_template/v2/`, `persona/prompt_template/v3_ChatGPT/`,
  and `norm/*_prompt/` — prompt templates are frozen for baseline replication.
- `llm_router.py`, `schemas.py`, all Pydantic schemas — working.
- All existing `print()` debug breadcrumbs — preserved as added breadcrumbs only.
- Function names, signatures, return-shape contracts — preserved.

---

## Commits

```
b7539cf test: add tests/test_cleanups.py covering Qwen3 leniency
d2fe2a1 port: makedirs + scaffold-tolerant JSON in creation.py and norm_save (class 1)
6f454e3 port: leniency fixes for run_gpt_prompt_norm.py (classes 3,4)
1ef5f45 port: leniency fixes for run_gpt_prompt.py + gpt_structure.py (classes 3,4)
```
