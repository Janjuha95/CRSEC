# CRSEC GPT-4 → Qwen3 Port: Flagged Items

Things I noticed during the audit that look suspicious but where I either
wasn't confident the right fix is or where touching them risked silently
changing baseline-replication behavior. Listed so the next pass can decide.

---

## 1. `run_gpt_prompt_task_decomp` / `task_decomp_v2`: post-cleanup `fin_output[-1]` IndexError

**File:** `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py`,
around line 500 and 720.

After `__func_clean_up` returns, the code does:

```python
fin_output = []
time_sum = 0
for i_task, i_duration in output:
    time_sum += i_duration
    if time_sum <= duration:
        fin_output += [[i_task, i_duration]]
    else:
        break
ftime_sum = 0
for fi_task, fi_duration in fin_output:
    ftime_sum += fi_duration
fin_output[-1][1] += (duration - ftime_sum)   # <-- IndexError if fin_output is empty
```

If the very first cleanup-task's duration already exceeds `duration`,
`fin_output` is empty and `fin_output[-1]` raises. The existing TODO comment
calls this out ("THERE WAS A BUG HERE..."). The Class-2 fail-safe shape fix
(`[["asleep", duration]]`) means the fail-safe path returns at least one
non-zero entry, but the *successful* cleanup path can still produce an empty
`fin_output` if the model returned only one oversized task.

**Why I didn't fix:** the right semantics aren't obvious — does the agent
fall back to the fail-safe, or take just the oversized task clamped to
`duration`? Worth a conscious decision rather than a guess.

---

## 2. `GPT4_safe_generate_response` and `ChatGPT_safe_generate_response`
   return `False` instead of `fail_safe_response`

**File:** `reverie/backend_server/persona/prompt_template/gpt_structure.py`,
lines ~90 and ~129.

```python
def GPT4_safe_generate_response(..., fail_safe_response="error", ...):
    for i in range(repeat):
        try:
            ...
            return func_clean_up(...)
        except:
            pass
    return False  # <-- not fail_safe_response
```

So `fail_safe_response` is **never used** in these two paths. Callers
typically check `if output != False: return output` and then fall through to
a different code path. The contract is weird and confusing, but
intentional-looking. I left both as-is and only logged in the
`_OLD`/`safe_generate_response` paths (which actually return the fail-safe).

**Why I didn't fix:** flipping these to return `fail_safe_response` would
change observable behavior in functions that *currently* fall through to a
non-ChatGPT-plugin branch on `False`. Not a port-class problem; flagging.

---

## 3. `run_gpt_prompt_summarize_ideas`: `print_run_prompts(... output)` before `output` is defined

**File:** `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py`,
around line 2670.

```python
output = ChatGPT_safe_generate_response(prompt, example_output, ...)
if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param,
                     prompt_input, prompt, output)

if output != False:
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
```

Looks fine on a re-read — `output` *is* defined before the `if` — but the
`if debug or verbose` is inside the loop with `output != False` after, and
the function has *no return* on the `output == False` path. The function
falls off the end and returns `None`. Most other `run_gpt_prompt_*` have an
explicit `return False` or similar.

**Why I didn't fix:** behavior matches the pre-existing pattern (function
returns implicit `None` on the failure path, caller is expected to handle).
Touching it risks changing callsite assumptions.

---

## 4. `run_gpt_prompt_new_decomp_schedule.__func_clean_up`: regex-strict on `HH:MM ~ HH:MM -- activity`

**File:** `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py`,
around line 1395.

```python
for time_str, action in ret_temp:
    start_time = time_str.split(" ~ ")[0].strip()
    end_time = time_str.split(" ~ ")[1].strip()
    delta = datetime.datetime.strptime(end_time, "%H:%M") - ...
```

Qwen3 may emit `09:00 - 10:00` or `09:00am – 10:00am` (en-dash) instead of
the literal ` ~ ` separator. Won't unpack, falls through to fail-safe.

**Why I didn't fix:** the cleanup is intricate, and the fail-safe
(`get_fail_safe(main_act_dur, truncated_act_dur)`) returns a schedule that
preserves the consumer contract. So fail-safe activation here is graceful.
Adding leniency would expand behavior we have no smoke-test signal on.
Worth fixing if smoke-test logs show this fail-safe firing.

---

## 5. `run_gpt_prompt_create_conversation.__func_clean_up`: `for time_str, action in ret_temp:` may unpack-fail

**File:** `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py`,
around line 1400 inside `__func_clean_up`.

Same family as #4: assumes every line has exactly one ` -- ` separator,
crashes if Qwen3 emits any line without it.

**Why I didn't fix:** same reasoning as #4.

---

## 6. `run_gpt_prompt_norm.py` `run_gpt_active_norms_classfication.__func_clean_up`:
   indexes `str.split('\n')[1]` without bounds check

**File:** `reverie/backend_server/norm/run_gpt_prompt_norm.py`, around
line 382.

```python
for str in gpt_response.split("ABSTRACT: ")[1:]:
    ret += [[str.split('\n')[0], str.split('\n')[1]]]
```

If a section after `ABSTRACT:` has only one line, `split('\n')[1]` → IndexError.

**Why I didn't fix:** the function's `__func_validate` calls `__func_clean_up`
in a try/except, so the IndexError just kicks the safe wrapper into retry +
fail-safe. Not a new failure mode, just always-already-fragile. Worth
hardening if smoke logs show this fail-safe firing frequently.

---

## 7. Stale references in `run_gpt_prompt.py` to non-existent `random` / `string`

**File:** `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py`,
function `get_random_alphanumeric()` (line ~21).

Uses `random.randint(...)` and `random.choices(string.ascii_letters + string.digits, ...)`
but the module never imports `random` or `string` — these come in via
`from global_methods import *` and `from persona.prompt_template.gpt_structure import *`.

**Why I didn't fix:** it works because of the wildcard imports. Touching it
is the kind of "while you're at it" refactor the brief forbids.

---

## 8. `norm/run_gpt_prompt_norm.py`: `SpecificNormUtility` still references
   `openai.ChatCompletion.create`

**File:** `reverie/backend_server/norm/run_gpt_prompt_norm.py`, line ~824.

```python
gpt_ret = openai.ChatCompletion.create(model=self.model, ...)
```

This class still calls the OpenAI SDK directly, bypassing the Ollama router.
If `openai` isn't installed in the venv, instantiating this class crashes;
if it *is* installed, calls go to OpenAI's servers (or fail with no API
key) instead of local Qwen3.

**Why I didn't fix:** routing this through `llm_call(...)` requires a
contract decision — does it use REASONING tier or PRIMARY? What schema?
Norm utility is critical so probably REASONING. But this isn't a port-class
fix, it's an unported piece of the OpenAI-removal task. Flag for separate
PR.

**Severity:** Will likely crash on first call during norm evaluation. Check
`generate_normal_norm_utility(norm, persona)` in `norm_evaluate.py` and
whether `SpecificNormUtility` is exercised in the 10-step smoke test.

---

## 9. `run_gpt_prompt.py:_strip_scaffolding` and persona name leakage:
   only strips `<name> is `, not `<name>: ` mid-response

**File:** `reverie/backend_server/persona/prompt_template/gpt_structure.py`,
the `_strip_scaffolding` helper.

It strips persona-name prefixes (`"Maeve is "`, `"Maeve: "`, `"Maeve -- "`)
only at the very start of the response. If Qwen3 echoes the name mid-line
(e.g., `"The task: Maeve is sleeping at 7"`), the leak survives.

**Why I didn't fix:** mid-line stripping risks false positives (legitimate
references to the persona inside the actual answer). The current behavior
matches what cleanups already handle via `.split(" is ")[-1]` patterns.
Flag for revisit if smoke logs show this pattern.

---

## 10. `generate_hourly_schedule` prompt path — note no leniency fix

The prompt template `norm/norm_compliance_prompt/generate_hourly_schedule_ours_v1.txt`
ends with `... Activity: {persona.scratch.get_str_firstname()} is`. The
prompt structure is "complete the sentence", so Qwen3 may emit just the
predicate. The current cleanup splits on `"Activity:"` and finds `" is "`,
which works for most outputs. But the empty-response guard I added now
raises instead of silently returning empty string. The safe wrapper will
re-try then fail-safe to `"asleep"`. That's the same behavior as before for
the empty case but loudly logged.

**Severity:** none, but worth knowing — `fail_safe = "asleep"` propagating
into hourly schedule entries silently inflates "asleep" minutes in metrics.

---

## 11. Tests run with `cwd=backend_server`, real cleanups need cwd=backend_server

`tests/test_cleanups.py` calls `setUpModule` to `os.chdir(BACKEND)` because
some run_gpt_prompt_* functions load prompt templates from disk via
relative paths. In real runs from `reverie/backend_server/reverie.py` this
is fine. If someone runs CRSEC from a different cwd, they'll hit
`FileNotFoundError` on prompt files — that's a long-standing fragility, not
a port-class problem.

**Why I didn't fix:** out of scope; the existing test files already pin cwd
the same way.
