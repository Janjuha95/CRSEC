# CRSEC → Qwen3/Ollama Robustness Audit — FINDINGS

Branch: `norm_deflection`. Diagnosis only — **no code changed**. Each finding
gives `file:line`, the trigger, the caller that breaks, severity, and a
**minimal** proposed fix. Fixes are applied only after your approval.

All paths below are relative to `reverie/backend_server/`. Line numbers are as
of this audit.

## Severity summary

| # | Finding | Class | Severity | Reproduced? |
|---|---------|-------|----------|-------------|
| F1 | `generate_prompt` reads templates with the OS default codec (no `encoding=`) → `UnicodeDecodeError` | 3 | **CRASH** (whole step) | ✅ end-to-end |
| F2 | `SpecificNormUtility.specific_norm_utility` returns bare `False` → `len(False)` TypeError in `norm_evaluate_check` normal path | 1 | **CRASH** (whole step) | ✅ mechanism |
| F3 | Conflict parser: `conf` (logged "conflict?") inverts to `yes` on `no`-conflict output | 4 | Diagnostic corruption | ✅ |
| F4 | Conflict parser: `talk` (drives `norm_conflict`) flips under markdown/no-space `FINAL OUTPUT` | 4 | Silent behavior corruption | ✅ |
| F5 | `log_norm_adoption` has **zero callers** → `total_norms_adopted=0` | 5 | Metrics gap | n/a |
| F6 | `snapshot` gated on `step % 100 == 0` → `total_steps_logged=0` in <100-step runs | 5 | Metrics gap | n/a |
| F7 | `run_gpt_non_norm_conflict_chat_reflect` fail-safe is a truthy list, success shape is `bool` | 1 | Silent data pollution | — |
| F8 | `_snapshot_population_stats` counts roles via `scratch.identity` (free text) not `scratch.agent_type` | 5 | Metrics gap (secondary) | — |
| F9 | `NormDatabase` `act_norm[str(i)]` lookups assume contiguous 1..N keys; load uses saved `ID` | 2 | Conditional KeyError | — |

---

## Class 1 — Sentinel / invalid-return propagation

### Method: every `run_gpt_prompt_*` in `run_gpt_prompt_norm.py` was traced to its caller.

Result: **all of the `run_gpt_prompt_*` functions are shape-safe.** Each returns
its own fail-safe on exhaustion (the safe-wrappers return `fail_safe_response`,
not `False`), and every caller length-guards before indexing/unpacking. Trace
table:

| Function | Fail-safe | Caller | Caller guard | Verdict |
|---|---|---|---|---|
| `decide_if_norm_conflict` | `["ERROR"]` | `norm_retrieve.py:31` | `if x[0]=="yes"` else branch; `x[1]` only in yes-branch | safe |
| `norm_reflect_from_thoughts` | `["I am hungry"]` | `norm_reflect.py:64` | `len(ret)==0 or len(ret[0])!=2` | safe |
| `norm_format` | `False` | `norm_reflect.py:79` | `if ret==False`/`try ret["norm_1"]` | safe |
| `conflict_chat_reflect` | `["I am hungry"]` | `norm_reflect.py:128` | `if len(x)==2` | safe |
| `chat_norms_summarize` | `["I am hungry"]` | `norm_reflect.py:181` | iterates list | benign pollution only |
| `immediate_evaluate_recognization` | `["I am hungry"]` | `norm_evaluate.py:28` | `if len(ret)==2` | safe |
| `seeds_content_check` | `["I am hungry"]` | `norm_evaluate.py:82` | `if len(x)==3` | safe |
| `seeds_type_check` / `_v2` | `["I am hungry"]` | `norm_evaluate.py:92,347,373` | `if len(y) in (2,3)` | safe |
| `norm_recognize` | `["I am hungry"]` | `norm_evaluate.py:414` | `if len(x)!=4` | safe |
| `norm_duplicate_check` | `["I am hungry"]` | `norm_evaluate.py:303,324` | `if len(x)==2` | safe (but see **F1** re: its template) |
| `norm_fact_consistency_check` | `["I am hungry"]` | `norm_evaluate.py:273` | `if len(x)==2` | safe |
| `norm_utility` | `["I am hungry"]` | `norm_evaluate.py:392` | `if len(x)==2` | safe |
| `norm_long_term_synthesis` | `False` | `norm_evaluate.py:226` | `if act_norm==False or len(...)!=...` | safe |
| `active_norms_classfication_v2` | `False` | `norm_evaluate.py:163` | `if x==False` | safe |
| `norm_recognize_conflict_check` | `False` | `norm_evaluate.py:448` | `if x==False`/`=='yes'` | safe |
| `long_term_norm_utility` | `["I am hungry"]` | `norm_evaluate.py:552` | `try: len(utility)!=2 except` | safe |
| `revise_identity_*` | `False` | `norm_compliance.py:75-93` | `if note==False: return` | safe |
| `daily_plan_v2` | 7-item list | `norm_compliance.py:115` | list `+` list | safe |
| `violation_check` | `{...}` dict | `violation_detection.py:60` | `isinstance(result, dict)` + `.get` | safe |

I also re-scanned `run_gpt_prompt.py` (84 functions): the remaining non-`None`
sentinels (`get_keywords` → `[]`/`set`, `keyword_to_thoughts`/`convo_to_thoughts`
→ `""`) are **shape-consistent** with their success returns. No new class-1
issue there. (The prior None-return audit + `tests/test_cleanups.py` already
cover that file.)

### ⛔ F2 (CRASH) — `SpecificNormUtility.specific_norm_utility` returns bare `False`; caller does `len(False)`

- **Where:** `norm/run_gpt_prompt_norm.py:875-909`. The method calls
  `openai.ChatCompletion.create(...)` at **line 896**, but `openai` is **not
  imported in this module** (verified: `'openai' in dir(module)` is `False`),
  and the installed `openai==2.30.0` removed `ChatCompletion.create` anyway. So
  line 896 raises `NameError`, the bare `except:` at **line 900** swallows it,
  and the method returns bare **`False`** (line 901) — *not* `[False]`.
- **Trigger:** any non-defector (`citizen`/`entrepreneur`) norm evaluation that
  reaches the utility step. `generate_normal_norm_utility`
  (`norm/norm_evaluate.py:469-470`) returns that bare `False`.
- **Caller that breaks:** `norm_evaluate_check`, **normal-norm path**,
  `norm/norm_evaluate.py:565-567`:
  ```python
  utility = generate_normal_norm_utility(new_norm, persona)
  if len(utility) != 2:        # len(False) -> TypeError
  ```
  The **long-term path** (`:552-557`) is wrapped in `try/except`, so it's safe —
  it's only the normal path that is unguarded. Reproduced:
  `len(False)` → `TypeError: object of type 'bool' has no len()`.
- **Propagation:** `norms_evaluate` → `persona.move` (`persona.py:236`) →
  reverie main loop (`reverie.py:405`). **None** of these wrap the call, so the
  whole simulation step dies.
- **Functional gap (separate from the crash):** because line 896 *always*
  fails, citizens/entrepreneurs can **never** get a specific-norm utility score
  — the entire non-defector utility path is dead. (Prior audit flagged the
  OpenAI call in `PORTING_FLAGGED.md #8` but treated the `False`/`[False]`
  return as "by design / safe"; it missed that the bare-`False` branch crashes
  the unguarded `len()` caller.)
- **Severity:** CRASH + silent functional dead-path.
- **Proposed minimal fix (two parts):**
  1. Make the failure return **shape-correct**: `except: return [False]` (line
     901), matching `get_defector_norm_utility`'s documented contract
     (`[score, reason]` | `[False]`). This alone removes the crash.
  2. Restore function: replace the dead `openai.ChatCompletion.create` block
     with `llm_call(prompt, call_type="norm_evaluation")` (mirroring
     `get_defector_norm_utility` in `defection_engine.py:102-103`) and parse via
     the existing `OUTPUT:` logic. This is the real port fix for `#8`.
  - Log `[FAIL_SAFE] specific_norm_utility: ...` to stderr on the failure path.

### F7 (minor) — `run_gpt_non_norm_conflict_chat_reflect` fail-safe shape mismatch

- **Where:** `norm/run_gpt_prompt_norm.py:239-240` returns `["I am hungry"]`,
  but `__func_clean_up` (line 231) returns a **bool**.
- **Caller:** `norm_reflect.py:179` `reflect_tag = generate_non_norm_conflict_chat_reflect(...)`
  then `if reflect_tag:`. On fail-safe, `reflect_tag` is a **non-empty list →
  truthy**, so a failed LLM call is treated as "yes, reflect," spuriously
  entering the `generate_chat_norms_summarize` branch (`:181`).
- **Severity:** silent data pollution (extra/garbage norm seeds on LLM failure),
  no crash.
- **Proposed fix:** `get_fail_safe()` → `return False` (matches the bool
  success shape).

---

## Class 2 — Dict lookups on runtime-built keys

- **execute.py (already fixed, do not touch):** `address_tiles[plan]` at
  `:85` and `:98` are both guarded by `if plan not in maze.address_tiles`
  fall-backs (`:81-83`, `:94-96`). Confirmed present.

### F9 (conditional) — `NormDatabase.act_norm[str(i)]` assumes contiguous keys

- **Where:** `norm/normDatabase.py:87-91` (`retrieve_relevant_norms` iterates
  `range(1, act_norm_count+1)` and indexes `act_norm[str(i)]`); same pattern in
  `norm/norm_save.py:42-45`.
- **Trigger:** in-memory builds are contiguous (`add_act_norm` assigns
  `id = act_norm_count`), **but on load** (`normDatabase.py:60-73`) the dict key
  is the *saved* `norm["ID"]`, while the loop bound is the count. If a persisted
  `personal_norm_database_validity.json` has non-contiguous / duplicated IDs
  (hand-authored bootstrap norm files, or a file edited between runs),
  `act_norm[str(i)]` raises **KeyError** during retrieval/save.
- **Caller that breaks:** `retrieve_relevant_norms` is called every step via
  `norm_retrieve` → unguarded → KeyError propagates to the main loop.
- **Severity:** crash, but **data-conditional** (depends on saved files). Lower
  priority than F1/F2.
- **Proposed minimal fix:** iterate `self.act_norm.items()` (or
  `.get(str(i))` with `continue` on `None`) instead of a `range`+index, with a
  `[FAIL_SAFE]` breadcrumb on a missing key. (Note: changing iteration order is
  behavior-visible for `random`-dependent downstream — recommend `.items()`
  which preserves insertion order.)

- **Other constructed-key lookups checked and found safe:**
  `norm_evaluate.py:205,210` (`content_to_*`), `:479` (`content_to_act_norm[s]`)
  and `specific_norm_deactive` are all already wrapped in `try/except`.

---

## Class 3 — File reads / encoding

### ⛔ F1 (CRASH, systemic) — `generate_prompt` opens templates with the OS default codec

- **Where:** `persona/prompt_template/gpt_structure.py:342-344`:
  ```python
  f = open(prompt_lib_file, "r")   # no encoding=, no fallback
  prompt = f.read()
  ```
- **Root cause:** The CONTEXT lists this function as *already fixed* with
  "utf-8 read with gb18030 fallback." **That fix is not in the current code** —
  `git log -S gb18030 -- gpt_structure.py` returns nothing; the function reads
  with `locale.getpreferredencoding()`, which on this machine is **cp1252**.
  Converting the templates to UTF-8 (commit `29512bc`) was necessary but **not
  sufficient** — the *reader* must also request UTF-8.
- **Reproduced end-to-end** (bare-`open` read, this box):
  - `norm/norm_evaluate_prompt/duplicate_check_v1.txt` →
    `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 in position 454`
  - `persona/prompt_template/v2/generate_pronunciatio_v1.txt` →
    `UnicodeDecodeError` (emoji, byte 0x81)
- **Callers that break:** *every* prompt-driven function (84 in
  `run_gpt_prompt.py`, ~30 in `run_gpt_prompt_norm.py`, plus
  `defection_engine.py`) since they all funnel through `generate_prompt`. The
  crash happens at the `prompt = generate_prompt(...)` line **before** any
  safe-wrapper, so fail-safes never engage. Concretely: `duplicate_check_v1.txt`
  feeds `run_gpt_norm_duplicate_check` → `norm_evaluate_check` → `norms_evaluate`
  → `persona.move` (unwrapped) → main loop crash. `generate_pronunciatio_v1.txt`
  crashes the action-pronunciatio path the same way.
- **Severity:** CRASH. Highest-impact finding — it's the systemic root behind
  the "occasional `UnicodeDecodeError`" symptom and it gates F-class-3 entirely.
- **Proposed minimal fix:** read UTF-8 with a gb18030 fallback (the upstream
  repo has GBK-origin templates):
  ```python
  try:
      with open(prompt_lib_file, "r", encoding="utf-8") as f:
          prompt = f.read()
  except UnicodeDecodeError:
      print(f"[FAIL_SAFE] generate_prompt: {prompt_lib_file} not utf-8, retrying gb18030", file=sys.stderr)
      with open(prompt_lib_file, "r", encoding="gb18030") as f:
          prompt = f.read()
  ```

### Template re-scan (class-3 deliverable)

- **Invalid UTF-8 bytes remaining:** **none.** All `*.txt` templates are valid
  UTF-8 (the earlier GBK→UTF-8 conversion holds).
- **Valid-UTF-8-but-non-ASCII templates (12)** — these are the ones that
  mojibake or crash when read under cp1252 (i.e. by F1). CJK punctuation /
  smart quotes / em-dash / emoji:

  | Template | Non-ASCII | Under cp1252 |
  |---|---|---|
  | `norm/norm_evaluate_prompt/duplicate_check_v1.txt` | U+FF01 `！` | **CRASH** |
  | `persona/prompt_template/v2/generate_pronunciatio_v1.txt` | emoji (U+1F35E…) | **CRASH** |
  | `norm/creation_prompt/usr_prompt_antisocial_v1.txt` | U+FF1A `：` | mojibake |
  | `norm/creation_prompt/usr_prompt_v6.txt` | U+FF1A | mojibake |
  | `norm/defection_prompt/gossip_convo_v1.txt` | U+2014 `—` | mojibake |
  | `norm/norm_evaluate_prompt/long_term_synthesis_v2.txt` | U+3002 `。` | mojibake |
  | `norm/norm_identify_prompt/conflict_chat_reflect_v1.txt` | U+3002 | mojibake |
  | `norm/norm_identify_prompt/identify_norm_save_v3.txt` | U+FF08/09/1A/1B | mojibake |
  | `persona/prompt_template/v2/task_decomp_v1.txt` | U+2019 `’` | mojibake |
  | `persona/prompt_template/v2/task_decomp_v2.txt` | U+2019 | mojibake |
  | `persona/prompt_template/v3_ChatGPT/task_decomp_v1.txt` | U+2019 | mojibake |
  | `persona/prompt_template/v3_ChatGPT/task_decomp_v2.txt` | U+2019 | mojibake |

  After F1's fix all 12 read correctly; no per-template byte changes are needed.

### Minor — JSON loads without encoding

- `norm/normDatabase.py:22,59` (`json.load(open(...))`) and `reverie.py` persona
  JSON reads don't pass `encoding=`. JSON files are ASCII-clean today, so this
  is low risk; recommend `encoding="utf-8"` for consistency only.

---

## Class 4 — Parser correctness for Qwen3 output (`decide_if_norm_conflict`)

Function: `run_gpt_prompt_decide_if_norm_conflict`
(`norm/run_gpt_prompt_norm.py:21-74`), template
`norm/norm_retrieve_prompt/check_conflict_decide_talk_v5.txt`. It produces three
values: `output = [talk, gpt_response, conf]`.

- `talk` (`output[0]`) = `FINAL OUTPUT` line → **this is what drives behavior**:
  `norm_retrieve.py:31` `if x[0]=="yes"` → `plan.py:761-766` sets
  `persona.scratch.norm_conflict = True` and forces a conflict conversation.
- `conf` (`output[2]`) = the Question-1 answer → **only used in the diagnostic
  `print(... "conflict?", output[2])` at line 73.**

### F3 — `conf` inversion (diagnostic corruption)

- **Where:** line 52. `conf` is extracted by splitting on the literal
  `'whether there is a conflict?\nAnswer in "yes" or "no" and provide a reason: '`.
- **Trigger:** that split key **does not match the template**, which has a
  `Reasoning: Let's think step by step.` line *between* the question and the
  "Answer in…" line. So `.split(key)[-1]` returns the **entire response**, and
  `_yn` then returns the *first* yes/no token anywhere in the reasoning.
- **Reproduced:** a response that says "...Maybe **yes** at first glance, but on
  reflection… no conflict. … FINAL OUTPUT: No" parses to **`conf = "yes"`** while
  `talk = "no"`. This is exactly the "reasoned no-conflict / FINAL OUTPUT: No but
  logged conflict=yes" case you observed.
- **Severity:** diagnostic only (it's printed, not consumed) — but it
  corrupts any analysis that greps stdout for "conflict?".

### F4 — `talk` flips under markdown / no-space `FINAL OUTPUT` (behavioral)

- **Where:** lines 44 & 51 split on the exact literal `"FINAL OUTPUT: "`
  (trailing space, no markdown).
- **Trigger:** Qwen3 commonly emits `**FINAL OUTPUT:** No` or `FINAL OUTPUT:No`.
  Either breaks the split, `[-1]` becomes the whole response, and `_yn` returns
  the first yes/no token in the **reasoning** instead of the final answer.
- **Reproduced:** response "...Maybe **yes** initially, but no. **FINAL
  OUTPUT:** No" parses to **`talk = "yes"`** → `norm_conflict=True` → a conflict
  conversation that should not happen.
- **Severity:** silent behavior corruption (spurious or suppressed norm-conflict
  conversations — directly distorts the norm-emergence dynamics under study).

### Proposed minimal fix (F3+F4 together) — robust extraction

Add to `__func_clean_up` / `__func_validate`:
1. Strip markdown (`**`, `` ` ``) and scaffolding before parsing (reuse
   `_strip_scaffolding`).
2. **`talk`:** regex on the final-answer marker, tolerant of bold/spacing:
   `re.search(r"final\s*output\s*:?\s*\**\s*(yes|no)", s, re.I)`; take the
   **last** such match (final answer wins over any earlier echo).
3. **`conf`:** isolate the Question-1 block (`text between "Question 1" and
   "Question 2"`), then take the yes/no after its `reason:`/`Answer` marker
   (fallback: first yes/no in that block only) — never the whole response.
4. Keep `get_fail_safe()` = `["ERROR"]` (caller already handles it).

The same `FINAL OUTPUT` / markdown-tolerance pattern should be spot-checked on
the other markered parsers (`immediate_evaluate_recognization` `ANSWER:`/`NORM
UTILITY:`, `seeds_type_check` `OUTPUT:`), but those already strip scaffolding and
fall back to `_extract_first_int`/`_extract_yes_no`, so they're lower risk.

---

## Class 5 — Metrics wiring gap

`MetricsCollector` (`norm/metrics.py`) is instantiated at `reverie.py:142`, and
`persona.metrics = self.metrics` is set at `reverie.py:145`. Event hooks that
*are* wired correctly: `log_defection_attempt` (via `calculate_defection_utility`
in `norm_compliance.py`, defector-gated) and `log_violation`/`log_enforcement`
(via `process_violations`, `persona.py:244` passes `metrics`). The two reported
zeros come from two specific gaps:

### F5 — `total_norms_adopted = 0`: `log_norm_adoption` has **zero callers**

- **Where the metric is computed:** `metrics.py:247`
  (`sum(... if n["accepted"])` over `norm_adoption_log`).
- **Where the log would be filled:** `log_norm_adoption` (`metrics.py:82-94`) —
  `grep` confirms **no call sites anywhere**.
- **Where adoption actually happens (candidate fire sites — for your decision):**
  - `norm/norm_evaluate.py:587` — `norms_evaluate`, on `save_tag` True →
    `add_act_norm(new_norm)` = **accepted**. The matching `save_tag` False branch
    = **rejected** (poignancy set to −2/−3/−4 in `norm_evaluate_check`).
  - `norm/norm_evaluate.py:242` — `run_long_term_norm_evaluate`, `add_act_norm`.
  - `norm/norm_evaluate.py:258` — `init_evaluate`, `add_act_norm`.
  - `norms_evaluate` does not currently receive `metrics`, but `persona.metrics`
    is available, so a hook can read `getattr(persona, 'metrics', None)`.
- **I am not inventing the metric semantics** (what counts as "utility_score",
  whether long-term synthesis counts as an adoption, whether rejects should be
  logged) — flagging the gap and candidate hooks for you to decide.

### F6 — `total_steps_logged = 0`: `snapshot` only fires every 100 steps

- **Where:** `reverie.py:438-442`:
  ```python
  if self.step % 100 == 0:
      self.metrics.snapshot(self.personas, getattr(self,'reputation_system',None), self.step)
  ```
- **Trigger:** a ~50-step run never satisfies `step % 100 == 0`, so
  `population_stats` stays empty and `metrics.py:239`
  (`population_stats[-1]["step"] if population_stats else 0`) → **0**. The same
  gate zeroes `norm_snapshots`, `trust_snapshots`, and `compliance_timeline`.
- **Candidate fixes (your call):** lower the interval (e.g. every 10 steps),
  make it configurable, and/or take a final `snapshot(...)` inside `save()`
  (`reverie.py:213`, right before `save_all()`), so short runs always log ≥1.

### F8 — secondary: population role counts use the wrong field

- **Where:** `metrics.py:181-187` (`_snapshot_population_stats`) does
  `identity = persona.scratch.identity; if identity == "defector"`. But
  `scratch.identity` is the free-text ISS identity string; the **role** lives in
  `scratch.agent_type` (`"defector"`/`"entrepreneur"`/`"citizen"`, with
  `is_defector()` at `scratch.py:482`). As written, `defector_count` /
  `entrepreneur_count` are ~always 0. `log_norm_adoption` would inherit the same
  issue (its `agent_identity` arg is fed `scratch.identity`).
- **Proposed (pending your confirmation of intent):** use `scratch.agent_type`
  for role counts.

---

## Proposed fix order (await approval before any edit)

1. **F1** — `generate_prompt` utf-8 + gb18030 fallback. *(systemic crash; unblocks everything)*
2. **F2** — `specific_norm_utility` → return `[False]` on failure **and** route through `llm_call`. *(crash + dead path)*
3. **F4/F3** — robust `FINAL OUTPUT` / conflict extraction (markdown-tolerant; isolate Q1 block). *(behavioral + diagnostic)*
4. **F5/F6** — metrics hooks (after you choose semantics/threshold). 
5. **F7, F9, F8** — minor shape/lookup/role-field fixes.

Per the task contract: no validated success-path return will be altered;
every fail-safe will keep the caller's expected shape; each fix gets a
`[FAIL_SAFE] <fn>: ...` stderr breadcrumb and a regression test; `pytest` is run
at the end.
