"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import os
import random
import sys
import time
import inspect

from utils import *
import call_profiler
from llm_router import openai_compat_call, _caller_prompt_fn


def _count_retry(retry_fn):
    """Count one failed safe_generate iteration against the calling
    run_gpt_* function. Resolves the caller name lazily (stack walk only on
    the failure path) and returns it so subsequent iterations reuse it.
    Every still-broken parser multiplies its own LLM cost by its repeat
    budget; the retry_* / fail_safe_* counters make that burn visible in
    profile.json."""
    if retry_fn is None:
        retry_fn = _caller_prompt_fn()
    call_profiler.incr(f"retry_{retry_fn}")
    return retry_fn


def _count_fail_safe(retry_fn):
    """Count one exhausted safe_generate loop (all repeats failed)."""
    call_profiler.incr(f"fail_safe_{retry_fn or _caller_prompt_fn()}")


def _log_fail_safe_trigger(fs):
    """Emit a one-line breadcrumb naming the caller of the safe_* wrapper.

    Walks two frames up to find the run_gpt_prompt_* function that originally
    invoked one of the safe_response helpers below. Falls back to "?" if the
    caller can't be identified.
    """
    try:
        caller = "?"
        # frame 0 = this fn; 1 = the safe_* wrapper; 2 = the run_gpt_prompt_* caller
        outer = inspect.stack()
        if len(outer) >= 3:
            caller = outer[2].function
        print(f"[FAIL_SAFE] {caller}: returning {fs!r}", file=sys.stderr)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Qwen3 leniency helpers (added during port from GPT-4)
# Qwen3 is less disciplined than GPT-4 about prompt scaffolding: it tends to
# echo "Answer:", leading braces, code fences, or the persona's name back into
# its response. These helpers strip that scaffolding before strict parsers run.
# They are no-ops on properly-formatted GPT-4 output (lossless).
# ----------------------------------------------------------------------------

# Prefixes Qwen3 occasionally echoes at the start of a response.
_QWEN_LEADING_NOISE = (
    "answer:", "answer :", "output:", "output :",
    "response:", "response :", "result:", "result :",
    "final answer:", "final output:",
    "```json", "```",
)

# Suffixes Qwen3 occasionally appends.
_QWEN_TRAILING_NOISE = ("```", "---", "end", "END")


def _strip_scaffolding(text, persona_name=None):
    """Strip common Qwen3 prompt-leakage scaffolding from a model response.

    Applied BEFORE strict GPT-4-style parsers. Idempotent and lossless on
    well-formatted output.
    """
    if not isinstance(text, str):
        return text
    s = text.strip()

    # Drop fenced code blocks: ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()

    # Repeatedly peel leading noise tokens.
    changed = True
    while changed:
        changed = False
        low = s.lower()
        for token in _QWEN_LEADING_NOISE:
            if low.startswith(token):
                s = s[len(token):].lstrip()
                changed = True
                break
        if persona_name:
            for prefix in (f"{persona_name} is ", f"{persona_name}: ", f"{persona_name} -- "):
                if s.startswith(prefix):
                    s = s[len(prefix):]
                    changed = True
                    break

    # Drop a single leading "{" if there's no matching "}" on the first line.
    if s.startswith("{") and "}" not in s.split("\n", 1)[0]:
        s = s[1:].lstrip()

    for token in _QWEN_TRAILING_NOISE:
        if s.endswith(token):
            s = s[:-len(token)].rstrip()
    if s.endswith("}") and "{" not in s:
        s = s[:-1].rstrip()

    return s.strip()


def _extract_first_int(text):
    """Pull the first standalone integer from a string, or None if none found."""
    if not isinstance(text, str):
        return None
    import re as _re
    m = _re.search(r"-?\d+", text)
    if m is None:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _extract_yes_no(text):
    """Return 'yes' or 'no' (lower) from text, or None if neither found.

    Handles Qwen3 deviations like 'Yes.', 'Yes, because...', 'No - the norm
    is...'. The first standalone yes/no token wins.
    """
    if not isinstance(text, str):
        return None
    import re as _re
    s = _strip_scaffolding(text).lower()
    # Look for standalone yes/no.
    m = _re.search(r"\b(yes|no)\b", s)
    if m:
        return m.group(1)
    return None

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt):
  temp_sleep()

  return openai_compat_call(prompt, call_type="conversation")


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  try:
    return openai_compat_call(prompt, call_type="conversation")

  except:
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt):
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  try:
    return openai_compat_call(prompt, call_type="conversation")

  except:
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  retry_fn = None
  for i in range(repeat):

    try:
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass
    retry_fn = _count_retry(retry_fn)

  _count_fail_safe(retry_fn)
  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  retry_fn = None
  for i in range(repeat):

    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      # Envelope-tolerant parse: prefer the {"output": ...} envelope, but slice
      # to the first '{' so leading prose ("text\n{...}") doesn't break it. If
      # there is no parseable envelope (bare int, "5/10", plain text), fall
      # back to the raw stripped response and let func_validate / func_clean_up
      # decide. Well-formed envelopes still parse to exactly the same value as
      # before, so callers that already work are unaffected.
      end_index = curr_gpt_response.rfind('}') + 1
      start_index = curr_gpt_response.find('{')
      if start_index != -1 and end_index > start_index:
        envelope = curr_gpt_response[start_index:end_index]
      else:
        envelope = curr_gpt_response[:end_index]
      try:
        candidate = json.loads(envelope)["output"]
      except Exception:
        candidate = curr_gpt_response

      if func_validate(candidate, prompt=prompt):
        return func_clean_up(candidate, prompt=prompt)

      if verbose:
        print ("---- repeat count: \n", i, candidate)
        print (candidate)
        print ("~~~~")

    except:
      pass
    retry_fn = _count_retry(retry_fn)

  _count_fail_safe(retry_fn)
  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  retry_fn = None
  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose:
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass
    retry_fn = _count_retry(retry_fn)
  print ("FAIL SAFE TRIGGERED")
  _count_fail_safe(retry_fn)
  _log_fail_safe_trigger(fail_safe_response)
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  try:
    return openai_compat_call(prompt, call_type="conversation")
  except:
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  retry_fn = None
  for i in range(repeat):
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt):
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose:
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
    retry_fn = _count_retry(retry_fn)
  _count_fail_safe(retry_fn)
  _log_fail_safe_trigger(fail_safe_response)
  return fail_safe_response


def get_embedding(text, model="nomic-embed-text"):
  """All 9 embedding call sites funnel through here. Unguarded, a bloated
  thought (e.g. day-2 plan join) exceeds the embed model's context window and
  ollama raises ResponseError 'context length' — which killed exp1_base_r1_c2
  at its first sim-midnight. Clip up front, and on a context-length error
  halve and retry (max 3) before re-raising: visible failure > silent junk."""
  import ollama
  text = text.replace("\n", " ")
  temp_sleep()
  if not text:
    text = "this is blank"
  max_chars = int(os.environ.get("CRSEC_EMBED_MAX_CHARS", "8000"))
  if len(text) > max_chars:
    print(f"[EMBED CLIP] {len(text)} -> {max_chars} chars")
    text = text[:max_chars]
  for halving in range(4):  # initial attempt + up to 3 halvings
    try:
      return ollama.embeddings(model=model, prompt=text)["embedding"]
    except Exception as e:
      if "context length" in str(e).lower() and halving < 3:
        new_len = max(1, len(text) // 2)
        print(f"[EMBED CLIP] context-length error, halving "
              f"{len(text)} -> {new_len} chars (retry {halving + 1}/3)")
        text = text[:new_len]
        continue
      raise


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)


def GPT4_safe_generate_response_OLD(prompt,
                                    repeat=3,
                                    fail_safe_response="error",
                                    func_validate=None,
                                    func_clean_up=None,
                                    verbose=False):
    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    retry_fn = None
    for i in range(repeat):
        try:
            curr_gpt_response = GPT4_request(prompt)  # .strip()
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")

        except:
            pass
        retry_fn = _count_retry(retry_fn)
    print("FAIL SAFE TRIGGERED")
    _count_fail_safe(retry_fn)
    _log_fail_safe_trigger(fail_safe_response)
    return fail_safe_response

def GPT4_request_t1(prompt):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-3's response.
    """

    try:
        return openai_compat_call(prompt, call_type="conversation")

    except:
        print("ChatGPT ERROR")
        return "ChatGPT ERROR"

def GPT4_safe_generate_response_OLD_t1(prompt,
                                    repeat=3,
                                    fail_safe_response="error",
                                    func_validate=None,
                                    func_clean_up=None,
                                    verbose=False):
    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    retry_fn = None
    for i in range(repeat):
        try:
            curr_gpt_response = GPT4_request_t1(prompt)  # .strip()
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")

        except:
            pass
        retry_fn = _count_retry(retry_fn)
    print("FAIL SAFE TRIGGERED")
    _count_fail_safe(retry_fn)
    _log_fail_safe_trigger(fail_safe_response)
    return fail_safe_response

def ChatGPT_request_t0(prompt):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-3's response.
    """
    # temp_sleep()
    try:
        return openai_compat_call(prompt, call_type="conversation")

    except:
        print("ChatGPT ERROR")
        return "ChatGPT ERROR"

def ChatGPT_safe_generate_response_OLD_t0(prompt,
                                          repeat=3,
                                          fail_safe_response="error",
                                          func_validate=None,
                                          func_clean_up=None,
                                          verbose=False):
    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    retry_fn = None
    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request_t0(prompt)  # .strip()
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")

        except:
            pass
        retry_fn = _count_retry(retry_fn)
    print("FAIL SAFE TRIGGERED")
    _count_fail_safe(retry_fn)
    _log_fail_safe_trigger(fail_safe_response)
    return fail_safe_response


















