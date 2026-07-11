import sys
import re
import os
import copy

sys.path.append('../')
import call_profiler
from llm_router import llm_call
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.gpt_structure import (
    _strip_scaffolding, _extract_first_int, _extract_yes_no,
)
from persona.prompt_template.print_prompt import *
from norm.print_prompt_norm import *


def _log_fail_safe(fn_name, fs):
    """One-line breadcrumb when a cleanup itself falls back to fail-safe."""
    try:
        print(f"[FAIL_SAFE] {fn_name}: returning {fs!r}", file=sys.stderr)
    except Exception:
        pass


def _yn_token(seg):
    """First standalone yes/no token in `seg`, with markdown emphasis,
    brackets and backticks stripped first. Returns 'yes'/'no' or None."""
    if not isinstance(seg, str):
        return None
    s = re.sub(r"[*`\[\]<>]", " ", _strip_scaffolding(seg)).lower()
    m = re.search(r"\b(yes|no)\b", s)
    return m.group(1) if m else None


def _final_output_decision(gpt_response):
    """Authoritatively read the model's FINAL OUTPUT line.

    Handles the ways Qwen3 deviates from the GPT-4 template:
      * reasoning preamble before the answer (uses the LAST 'final output'
        marker, since the model often echoes the template earlier),
      * markdown emphasis (``**No**``) and brackets (``[No]``),
      * a missing space after the colon or the answer on the next line.

    Returns 'yes', 'no', or None when no decision can be read. The earlier
    parser split on a brittle literal prompt substring; when that substring
    was absent (the common case with Qwen3) the split returned the WHOLE
    response and the first yes/no token — frequently a 'yes' buried in the
    reasoning or in the echoed ``Answer in "yes" or "no"`` instruction — was
    taken, inverting a genuine 'No' final output into conflict=yes.
    """
    if not isinstance(gpt_response, str):
        return None
    s = _strip_scaffolding(gpt_response)
    low = s.lower()
    idx = low.rfind("final output")
    if idx != -1:
        seg = s[idx + len("final output"):].lstrip(": \t\r\n")
        first_line = seg.split("\n")[0] if seg else ""
        d = _yn_token(first_line)
        if d:
            return d
    # No usable FINAL OUTPUT marker: scan from the bottom for the last line
    # that actually carries a yes/no (the concluding answer).
    for line in reversed(s.splitlines()):
        d = _yn_token(line)
        if d:
            return d
    return None


def run_gpt_prompt_decide_if_norm_conflict(target_person_description, init_persona_norms, target_p, init_p_identity,
                                           init_p_innate, verbose=False):
    def create_prompt_input(target_person_description, init_persona_norms, target_p, init_p_identity,
                            init_p_innate):
        prompt_input = []
        prompt_input += [target_person_description]
        prompt_input += [init_persona_norms]
        prompt_input += [init_p_identity]
        prompt_input += [init_p_innate]
        prompt_input += [target_p]
        return prompt_input

    def __func_validate(gpt_response, prompt=""):
        # Accept the response as soon as a FINAL OUTPUT decision is readable;
        # otherwise retry (and ultimately fall back to the ['ERROR'] skip).
        try:
            return _final_output_decision(gpt_response) in ("yes", "no")
        except Exception:
            return False

    def __func_clean_up(gpt_response, prompt=""):
        # The FINAL OUTPUT line is authoritative for BOTH the conversation
        # decision (output[0], consumed by norm_retrieve) and the logged
        # conflict flag (output[2]): per check_conflict_decide_talk_v5 the
        # FINAL OUTPUT is forced to 'No' whenever there is no conflict, so
        # No -> no-conflict, Yes -> conflict.
        decision = _final_output_decision(gpt_response) or "no"
        return [decision, gpt_response, decision]

    def get_fail_safe():
        # ['ERROR'] (length 1) is the skip sentinel: the caller checks
        # output[0] == "yes", so "ERROR" cleanly means "no conflict, skip".
        fs = "ERROR"
        return [fs]

    gpt_param = {"engine": "gpt-4-1106-preview", "max_tokens": 20,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_retrieve_prompt/check_conflict_decide_talk_v5.txt"
    prompt_input = create_prompt_input(target_person_description, init_persona_norms, target_p, init_p_identity,
                                       init_p_innate)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    if len(output) == 3:
        print("check_conflict_decide_talk_vvvvvvvvv5: talk?", output[0],"; conflict?", output[2])
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_norm_reflect_from_thoughts(reletive_thoughts, verbose=False):
    '''
    Args:
        reletive_thoughts: str of thought with its filling thoughts: "1.xxxx\n2.xxxxx\n3.xxxxx\n"

    Returns:
    '''

    def create_prompt_input(reletive_thoughts):
        prompt_input = [reletive_thoughts]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = gpt_response.strip()
        ret = []
        for str in gpt_response.split('- Norm: ')[1:]:
            ret += [[str.split(" Related thought: ")[0].strip('\"'),
                     str.split(" Related thought: ")[1].strip('\n').strip('\"')]]
        print(gpt_response)
        print(ret)
        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-4-1106-preview", "max_tokens": 150,
                 "temperature": 0.5, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_identify_prompt/thought_reflect_v3.txt"
    prompt_input = create_prompt_input(reletive_thoughts)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def _parse_norm_format_json(gpt_response):
    """Parse a norm-format response into the {"norm_1": {...}} dict that
    generate_format_norm consumes.

    Deliberately does NOT use _strip_scaffolding: its "drop a leading '{'
    with no '}' on the first line" heuristic beheads multi-line JSON, which
    made the old first-{-to-last-} slice unbalanced and failed 100% of
    calib_008 calls (killing every norm seed). raw_decode reads the first
    complete JSON value; a brace-balancing retry recovers truncated output.
    Module-level so replay tests can import it.
    """
    s = gpt_response.strip() if isinstance(gpt_response, str) else ""
    # peel a code fence without touching braces
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    if "OUTPUT:" in s:
        s = s.split("OUTPUT:")[-1].strip()
    first = s.find("{")
    if first == -1:
        raise ValueError("norm_format: no JSON object in response")
    tail = s[first:]
    try:
        j, _ = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError:
        # truncated JSON: cut back to the last complete "}", close the
        # remaining open braces, drop a dangling comma
        j = None
        for cut in range(len(tail), 0, -1):
            if tail[cut - 1] == "}":
                cand = tail[:cut]
                opens = cand.count("{") - cand.count("}")
                if opens > 0:
                    cand = cand + "}" * opens
                cand = re.sub(r",\s*}", "}", cand)
                try:
                    j, _ = json.JSONDecoder().raw_decode(cand)
                    break
                except json.JSONDecodeError:
                    continue
        if j is None:
            raise ValueError("norm_format: unrecoverable JSON")
    if not isinstance(j, dict):
        raise ValueError("norm_format: not a dict")
    # normalize alternative top-level shapes to {"norm_1": {...}}
    if "norm_1" not in j:
        if "content" in j:                      # bare norm object
            j = {"norm_1": j}
        else:                                   # e.g. {"norm": {...}}
            for v in j.values():
                if isinstance(v, dict) and "content" in v:
                    j = {"norm_1": v}
                    break
            else:
                raise ValueError("norm_format: no norm object found")
    # generate_format_norm indexes these keys; missing ones must fail HERE so
    # the repeat loop retries instead of silently dropping the seed
    norm = j["norm_1"]
    for k in ("ID", "type", "content", "subject", "predicate", "object"):
        if k not in norm:
            raise ValueError(f"norm_format: missing key {k}")
    return j


def run_gpt_prompt_norm_format(norm_str, verbose=False):
    def create_prompt_input(norm_str):
        prompt_input = [norm_str]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        j = _parse_norm_format_json(gpt_response)
        print(type(j))
        print(j)
        return j

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0.5, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_identify_prompt/identify_norm_save_v3.txt"
    prompt_input = create_prompt_input(norm_str)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)
    print(norm_str, output)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_conflict_chat_reflect(all_utt, verbose=False):
    '''
    Returns:
        output=[norm_str, reflect_tag]
    '''

    def create_prompt_input(all_utt):
        prompt_input = [all_utt]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):

        print(gpt_response)
        res = gpt_response.split('Step 3: ')[-1]
        if res[:2].lower() == 'no' and res[:4].lower() != 'norm':
            return ["", False]
        norm_str = res.split(": ")[-1]
        return [norm_str, True]

    def __func_validate(gpt_response, prompt=""):
        try:
            if len(gpt_response.split('Step 3:')) > 1:
                return True
            return False
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0.5, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_identify_prompt/conflict_chat_reflect_v1.txt"
    prompt_input = create_prompt_input(all_utt)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_non_norm_conflict_chat_reflect(all_utt, verbose=False):
    '''
    Returns:
        output = ture or false
    '''

    def create_prompt_input(all_utt):
        prompt_input = [all_utt]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):

        print(gpt_response)
        return _extract_yes_no(gpt_response) == "yes"

    def __func_validate(gpt_response, prompt=""):
        try:
            return _extract_yes_no(gpt_response) in ("yes", "no")
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_identify_prompt/non_conflict_chat_reflect_v1.txt"
    prompt_input = create_prompt_input(all_utt)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_chat_norms_summarize(all_utt, verbose=False):
    def create_prompt_input(all_utt):
        prompt_input = [all_utt]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = "1. " + gpt_response.strip()
        ret = []
        for str in gpt_response.split("\n"):
            ret += [str.split(". ")[-1]]
        print(gpt_response)
        print(ret)
        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0.5, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_identify_prompt/chat_summarize_norms_v1.txt"
    prompt_input = create_prompt_input(all_utt)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_immediate_evaluate_recognization(curr_norm_seed, curr_active_norms, seed_related_desc, curr_person_name,
                                             verbose=False):
    '''

    Args:
        curr_norm_seed: str
        curr_active_norms:str
        seed_related_desc:str
        curr_person_name:str

    Returns:
        output = [poi, norm_tag]
    '''

    def create_prompt_input(curr_norm_seed, curr_active_norms, seed_related_desc, curr_person_name):
        prompt_input = [curr_norm_seed]
        prompt_input += [curr_active_norms]
        prompt_input += [seed_related_desc]
        prompt_input += [curr_person_name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        # Qwen3 may emit "NORM UTILITY: 7." or "norm utility is 7" or just
        # an integer; same for ANSWER (yes/no). Tolerate both.
        s = _strip_scaffolding(gpt_response)
        util_seg = s.split("NORM UTILITY: ")[-1].split("\n")[0]
        try:
            poi = int(util_seg.strip())
        except Exception:
            n = _extract_first_int(util_seg)
            poi = n if n is not None else 0
        tag_seg = s.split("ANSWER: ")[-1]
        m = re.search(r"\b(yes|no)\b", tag_seg.lower())
        if m and m.group(1) == 'yes':
            return [poi, True]
        return [poi, False]

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/immediate_evaluate_recognization_v1.txt"
    prompt_input = create_prompt_input(curr_norm_seed, curr_active_norms, seed_related_desc, curr_person_name)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_seeds_content_check(candidate_norm, curr_active_norms, seed_related_desc, verbose=False):
    def create_prompt_input(candidate_norm, curr_active_norms, seed_related_desc):
        prompt_input = [candidate_norm]
        prompt_input += [curr_active_norms]
        prompt_input += [seed_related_desc]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        x1 = gpt_response.split('Answer 1 (return only in "YES" or "NO")###:')[-1].strip().split('\n')[0].lower().split(
            '\"')[0]
        x2 = gpt_response.split('Answer 2###:')[-1].strip().split('\n')[0].lower().split('\"')[0]
        x3 = gpt_response.split('STAGE 1:')[-1].strip().split('\n')[0].lower().split('\"')[0]
        return [x1, x2, x3]

    def __func_validate(gpt_response, prompt=""):
        try:
            if \
                    gpt_response.split('Answer 1 (return only in "YES" or "NO")###:')[-1].strip().split('\n')[
                        0].lower().split(
                        '\"')[0] in ["yes", "no"] and \
                            gpt_response.split('Answer 2###:')[-1].strip().split('\n')[0].lower().split('\"')[0] in [
                        "yes",
                        "no"] and \
                            gpt_response.split('STAGE 1:')[-1].strip().split('\n')[0].lower().split('\"')[0] in ["yes",
                                                                                                                 "no"]:
                return True
            return False
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-4", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/seeds_content_check_v1.txt"
    prompt_input = create_prompt_input(candidate_norm, curr_active_norms, seed_related_desc)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_active_norms_classfication(curr_active_norms, verbose=False):
    def create_prompt_input(curr_active_norms):
        prompt_input = [curr_active_norms]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        ret = []
        for str in gpt_response.split("ABSTRACT: ")[1:]:
            ret += [[str.split('\n')[0], str.split('\n')[1]]]
        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/active_norms_classfication_v1.txt"
    prompt_input = create_prompt_input(curr_active_norms)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_seeds_type_check(candidate_norm, verbose=False):
    def create_prompt_input(candidate_norm):
        prompt_input = [candidate_norm]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        x1 = gpt_response.split('OUTPUT: <"')[-1].split('"')[0].lower()
        x2 = gpt_response.lower().split('the type classification for input is "')[-1].split('"')[0]
        return [x1, x2]

    def __func_validate(gpt_response, prompt=""):
        try:
            if gpt_response.split('OUTPUT: <"')[-1].split('"')[0].lower() in ['incorrect', 'correct'] and \
                    gpt_response.lower().split('the type classification for input is "')[-1].split('"')[0] in [
                'descriptive', 'prohibitive', 'prescriptive']:
                return True
            return False
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-4", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/seeds_type_check_v1.txt"
    prompt_input = create_prompt_input(candidate_norm)
    prompt = generate_prompt(prompt_input, prompt_template)

    print(prompt)
    fail_safe = get_fail_safe()
    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_norm_recognize(candidate_norm, curr_active_norms, candidate_norm_utility, persona_ISS, persona_name,
                           verbose=False):
    def create_prompt_input(candidate_norm, curr_active_norms, candidate_norm_utility, persona_ISS, persona_name):
        prompt_input = [candidate_norm]
        prompt_input += [curr_active_norms]
        prompt_input += [candidate_norm_utility]
        prompt_input += [persona_ISS]
        prompt_input += [persona_name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        x1 = gpt_response.split('Answer 1: ')[-1].split('\n')[0].lower()
        x2 = gpt_response.split('Answer 2: ')[-1].split('\n')[0].lower()
        x3 = gpt_response.split('Answer 3: ')[-1].split('\n')[0].lower()
        x4 = gpt_response.split('Answer 4: ')[-1].split('\n')[0].lower()
        return [x1, x2, x3, x4]

    def __func_validate(gpt_response, prompt=""):
        print(gpt_response)
        try:
            x1 = gpt_response.split('Answer 1: ')[-1].split('\n')[0].lower()
            x2 = gpt_response.split('Answer 2: ')[-1].split('\n')[0].lower()
            x3 = gpt_response.split('Answer 3: ')[-1].split('\n')[0].lower()
            x4 = gpt_response.split('Answer 4: ')[-1].split('\n')[0].lower()
            if x1 in ['yes', 'no'] and x2 in ['yes', 'no'] and x3 in ['yes', 'no'] and x4 in ['yes', 'no']:
                return True
            return False
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-4", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/recognization_v1.txt"
    prompt_input = create_prompt_input(candidate_norm, curr_active_norms, candidate_norm_utility, persona_ISS,
                                       persona_name)
    prompt = generate_prompt(prompt_input, prompt_template)

    print(prompt)
    fail_safe = get_fail_safe()
    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_norm_duplicate_check(candidate_norm, curr_active_norms, verbose=False):
    def create_prompt_input(candidate_norm, curr_active_norms):
        prompt_input = [candidate_norm]
        prompt_input += [curr_active_norms]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        yn = _extract_yes_no(gpt_response)
        if yn is None:
            raise ValueError(f"norm_duplicate_check: no yes/no in {gpt_response!r}")
        return [yn, True]

    def __func_validate(gpt_response, prompt=""):
        try:
            return _extract_yes_no(gpt_response) in ('yes', 'no')
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-4-1106-preview", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/duplicate_check_v1.txt"
    prompt_input = create_prompt_input(candidate_norm, curr_active_norms)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()

    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_norm_fact_consistency_check(candidate_norm, seed_related_desc, verbose=False):
    def create_prompt_input(candidate_norm, seed_related_desc):
        prompt_input = [candidate_norm]
        prompt_input += [seed_related_desc]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        # Same Qwen3 yes/no tolerance as recognize_conflict_check.
        s = _strip_scaffolding(gpt_response)
        seg = s.split('Answer: ')[-1] if 'Answer: ' in s else s
        m = re.search(r"\b(yes|no)\b", seg.lower())
        x1 = m.group(1) if m else None
        if x1 is None:
            raise ValueError(f"norm_fact_consistency_check: no yes/no in {gpt_response!r}")
        x2 = s.split('New norm: ')[-1]
        return [x1, x2]

    def __func_validate(gpt_response, prompt=""):
        try:
            s = _strip_scaffolding(gpt_response)
            seg = s.split('Answer: ')[-1] if 'Answer: ' in s else s
            return bool(re.search(r"\b(yes|no)\b", seg.lower()))
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/fact_consistency_check_v1.txt"
    prompt_input = create_prompt_input(candidate_norm, seed_related_desc)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_norm_utility(candidate_norm, verbose=False):
    def create_prompt_input(candidate_norm):
        prompt_input = [candidate_norm]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        # GPT-4: "OUTPUT: 3. Reason text". Qwen3 may emit the score with the
        # reason on a new line, or drop the "OUTPUT:" marker entirely.
        s = _strip_scaffolding(gpt_response)
        seg = s.split('OUTPUT: ')[-1] if 'OUTPUT: ' in s else s
        try:
            x1 = int(seg.split('.')[0].strip())
            x2 = seg.split('.', 1)[1].strip() if '.' in seg else ""
        except Exception:
            n = _extract_first_int(seg)
            if n is None:
                raise
            x1 = n
            # Take everything after the first integer as the reason.
            after = re.sub(r"^.*?-?\d+", "", seg, count=1).lstrip(".:- ").strip()
            x2 = after
        return [x1, x2]

    def __func_validate(gpt_response, prompt=""):
        print(gpt_response)
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/norm_utility_v2.txt"
    prompt_input = create_prompt_input(candidate_norm)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD_t0(prompt, 3, fail_safe,
                                                   __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_norm_long_term_synthesis(classified_act_norms, verbose=False):
    def create_prompt_input(classified_act_norms):
        prompt_input = [classified_act_norms]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        ret = []
        for str in gpt_response.split(": ")[1:]:
            ret += [str.split('\n')[0]]
        return ret

    def __func_validate(gpt_response, prompt=""):
        print(gpt_response)
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/long_term_synthesis_v2.txt"
    prompt_input = create_prompt_input(classified_act_norms)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_active_norms_classfication_v2(curr_active_norms, verbose=False):
    def create_prompt_input(curr_active_norms):
        prompt_input = [curr_active_norms]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        return gpt_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False

    gpt_param = {"engine": "gpt-4-1106-preview", "max_tokens": 150,
                 "temperature": 1, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/active_norms_classfication_v3.txt"
    prompt_input = create_prompt_input(curr_active_norms)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = GPT4_safe_generate_response_OLD_t1(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# Tolerant markers for seeds_type_check_v2 (calib_009 parser pass). The
# template's final instruction line demands <"..."> around every answer, but
# its own EXAMPLES show the type in plain quotes — Qwen3 follows the examples
# ('The type classification for INPUT is "descriptive"'), so the literal
# '<"' marker split failed on 564/564 calib_009 calls, killing every seed at
# the type check. Each regex accepts <">, plain quotes, markdown bold, or no
# wrapping at all; the LAST occurrence wins (mirrors the old split()[-1]).
_TYPE_STEP1_RE = re.compile(
    r'STEP\s*1\s*:?\s*[<\["*\'\s]*(yes|no)\b', re.IGNORECASE)
_TYPE_STEP2_RE = re.compile(
    r'STEP\s*2\s*:?\s*[<\["*\'\s]*(incorrect|correct)\b', re.IGNORECASE)
_TYPE_CLASS_RE = re.compile(
    r'type\s+classification\s+for\s+(?:the\s+)?INPUT\s+is'
    r'\s*[:<\["*\'\s]*(descriptive|injunctive)\b', re.IGNORECASE)


def _parse_seeds_type_check(gpt_response):
    """[step1_yes_no, step2_correctness, type] from a type-check response.
    Module-level so tests/replay_calib_parsers.py can replay it."""
    if not isinstance(gpt_response, str):
        raise ValueError("seeds_type_check: not a string")
    m1 = _TYPE_STEP1_RE.findall(gpt_response)
    m2 = _TYPE_STEP2_RE.findall(gpt_response)
    m3 = _TYPE_CLASS_RE.findall(gpt_response)
    if not (m1 and m2 and m3):
        raise ValueError(
            f"seeds_type_check: markers missing "
            f"(step1={bool(m1)} step2={bool(m2)} class={bool(m3)})")
    return [m1[-1].lower(), m2[-1].lower(), m3[-1].lower()]


def run_gpt_seeds_type_check_v2(candidate_norm, norm_type, verbose=False):
    def create_prompt_input(candidate_norm, norm_type):
        prompt_input = [norm_type]
        prompt_input += [candidate_norm]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        return _parse_seeds_type_check(gpt_response)

    def __func_validate(gpt_response, prompt=""):
        try:
            _parse_seeds_type_check(gpt_response)
            return True
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-4", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/seeds_type_check_v3.txt"
    prompt_input = create_prompt_input(candidate_norm, norm_type)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_norm_recognize_conflict_check(candidate_norm, curr_active_norms, verbose=False):
    def create_prompt_input(candidate_norm, curr_active_norms):
        prompt_input = [candidate_norm]
        prompt_input += [curr_active_norms]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        # GPT-4: "Answer: yes." → "yes". Qwen3 may emit "Answer: Yes,
        # because..." → take first yes/no token after the marker, else any.
        s = _strip_scaffolding(gpt_response)
        seg = s.split('Answer: ')[-1] if 'Answer: ' in s else s
        m = re.search(r"\b(yes|no)\b", seg.lower())
        x1 = m.group(1) if m else None
        if x1 is None:
            raise ValueError(f"norm_recognize_conflict_check: no yes/no in {gpt_response!r}")
        return x1

    def __func_validate(gpt_response, prompt=""):
        try:
            s = _strip_scaffolding(gpt_response)
            seg = s.split('Answer: ')[-1] if 'Answer: ' in s else s
            return bool(re.search(r"\b(yes|no)\b", seg.lower()))
        except:
            return False

    def get_fail_safe():
        return False

    gpt_param = {"engine": "gpt-4", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/recognize_conflict_check_v1.txt"
    prompt_input = create_prompt_input(candidate_norm, curr_active_norms)
    prompt = generate_prompt(prompt_input, prompt_template)

    print(prompt)
    fail_safe = get_fail_safe()
    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_long_term_norm_utility(candidate_norm, related_specific_norms, related_specific_utilities, verbose=False):
    def create_prompt_input(candidate_norm, related_specific_norms, related_specific_utilities):
        prompt_input = [candidate_norm]
        prompt_input += [related_specific_norms]
        prompt_input += [related_specific_utilities]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        # GPT-4: "FINAL SCORE: [N]". Qwen3 may emit "FINAL SCORE: N",
        # "Final score: N/10", or just an integer with surrounding text.
        s = _strip_scaffolding(gpt_response)
        x1 = s.split('>. ')[-1].split('\n')[0]
        score_seg = s.split('FINAL SCORE:')[-1] if 'FINAL SCORE:' in s else s
        try:
            x2 = int(float(score_seg.split('[')[-1].split(']')[0]))
        except Exception:
            n = _extract_first_int(score_seg)
            if n is None:
                raise
            x2 = n
        return [x2, x1]

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return ["I am hungry"]

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_evaluate_prompt/abstract_norm_utility_v1.txt"
    prompt_input = create_prompt_input(candidate_norm, related_specific_norms, related_specific_utilities)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD_t0(prompt, 3, fail_safe,
                                                   __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# The utility template's examples show "- OUTPUT: <score>. Because <reason>."
# Qwen3 wraps the marker and the score in markdown bold ("- **OUTPUT**:
# **30**. Because ..." / "- **Because**: ..."), which broke the literal
# 'OUTPUT: ' + int() parse on 328/341 captured calib_010+011 calls — every
# failure deferred the seed, throttling adoption to ~3% of its rate.
_UTILITY_OUTPUT_RE = re.compile(r"OUTPUT\**\s*:\s*")
_UTILITY_SCORE_LINE_RE = re.compile(r"^[\s>*-]*\**(\d{1,3})\**\s*[.:]\s*(.*)$")


def _parse_norm_utility_response(ret_str):
    """[score, reason] from a specific-norm-utility response.

    Tolerates markdown bold around the OUTPUT marker and the score, bullet
    prefixes, and a reason continued on following lines. The last OUTPUT
    marker wins (mirrors the old split()[-1]); without a marker, falls back
    to a "<int>. <text>" score-line scan — never a bare first-int grab, since
    the INPUT echo line usually contains times/numbers. Raises on no parse
    (callers keep their shape-matched fail_safe). Module-level so
    tests/replay_calib_parsers.py can replay it."""
    if not isinstance(ret_str, str):
        raise ValueError("norm_utility: not a string")
    s = _strip_scaffolding(ret_str)
    markers = list(_UTILITY_OUTPUT_RE.finditer(s))
    if markers:
        seg = s[markers[-1].end():]
        m = re.search(r"-?\d+", seg)
        if m is None:
            raise ValueError(f"norm_utility: no score after OUTPUT marker: {seg[:80]!r}")
        score = int(m.group(0))
        reason = seg[m.end():]
    else:
        for line in s.split("\n"):
            m = _UTILITY_SCORE_LINE_RE.match(line.strip())
            if m:
                score = int(m.group(1))
                reason = m.group(2)
                break
        else:
            raise ValueError(f"norm_utility: no OUTPUT marker or score line: {s[:80]!r}")
    # peel "**. Because ..." / ". **Because**: ..." down to the reason text,
    # keeping the leading "Because" as the old parser did
    reason = reason.lstrip("*").lstrip().lstrip(".").lstrip()
    reason = re.sub(r"^[-\s]*\**(Because)\**\s*:?\s*", r"\1 ", reason,
                    flags=re.IGNORECASE).strip()
    return [score, reason]


class SpecificNormUtility:
    def __init__(self, system_msg, model="gpt-3.5-turbo-16k", temprature=0, max_tokens=4096, top_p=1,
                 frequency_penalty=0, presence_penalty=0):
        # parameter
        self.msg = [{"role": "system", "content": system_msg}]
        self.model = model
        self.temp = temprature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    # input prompt, return explicit norms
    def specific_norm_utility(self, candidate_norm):
        prompt_template = "norm/norm_evaluate_prompt/specific_norm_utility_v2.txt"
        prompt = generate_prompt([candidate_norm], prompt_template)

        user_prompt = {"role": "user", "content": prompt}
        self.msg.append(user_prompt)

        # Route through the local Ollama router (mirrors norm/creation.py
        # Creation.creation) instead of the removed OpenAI client. The
        # fail-safe is shape-matched ([int, reason], length 2) so the consumer
        # in norm_evaluate.py (`if len(utility) != 2`) never sees a bare False.
        fail_safe = [4, "fail_safe"]
        composed = "\n\n".join(m["content"] for m in self.msg)
        try:
            ret_str = llm_call(composed, call_type="norm_evaluation",
                               prompt_fn="run_gpt_specific_norm_utility")
            return _parse_norm_utility_response(ret_str)
        except Exception:
            return fail_safe


def run_gpt_revise_identity_plan(statements, p_name, time, verbose=False):
    def create_prompt_input(statements, p_name, time):
        prompt_input = [statements]
        prompt_input += [p_name]
        prompt_input += [time]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        return gpt_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_compliance_prompt/revise_identity_plan_v1.txt"
    prompt_input = create_prompt_input(statements, p_name, time)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_revise_identity_thought(statements, p_name, verbose=False):
    def create_prompt_input(statements, p_name):
        prompt_input = [statements]
        prompt_input += [p_name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        return gpt_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_compliance_prompt/revise_identity_thought_v1.txt"
    prompt_input = create_prompt_input(statements, p_name)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# ── Text-bloat caps (overnight hardening, Jul 12) ────────────────────────────
# Two upstream cleanups can return unbounded text that then feeds every
# subsequent prompt and embedding: daily_plan_v2's line-fallback turns each
# line of a rambling response into a plan item (the joined items are embedded
# as a plan-thought and injected into all 24 hourly-schedule prompts), and
# the revise-identity tails become scratch.currently / scratch.daily_plan_req
# (identity-stable-set injection). Caps are no-ops for well-sized output —
# identical inputs produce identical results when nothing is oversized.

_PLAN_MAX_ITEMS = 20
_PLAN_MAX_ITEM_CHARS = 200
_IDENTITY_MAX_CHARS = 1000


def _cap_plan_items(items):
    """At most 20 daily-plan items, each clipped to 200 chars."""
    capped = items
    if len(capped) > _PLAN_MAX_ITEMS:
        print(f"[PLAN CAP] {len(capped)} -> {_PLAN_MAX_ITEMS} items")
        capped = capped[:_PLAN_MAX_ITEMS]
    long_items = sum(1 for i in capped if len(i) > _PLAN_MAX_ITEM_CHARS)
    if long_items:
        print(f"[PLAN CAP] clipped {long_items} item(s) to "
              f"{_PLAN_MAX_ITEM_CHARS} chars")
        capped = [i[:_PLAN_MAX_ITEM_CHARS] for i in capped]
    return capped


def _cap_identity_text(text, label):
    """Clip an identity field to 1000 chars, cutting at the last sentence
    boundary before the limit when one exists past the halfway point."""
    if not isinstance(text, str) or len(text) <= _IDENTITY_MAX_CHARS:
        return text
    clipped = text[:_IDENTITY_MAX_CHARS]
    cut = clipped.rfind(". ")
    if cut > _IDENTITY_MAX_CHARS // 2:
        clipped = clipped[:cut + 1]
    print(f"[IDENTITY CAP] {label}: {len(text)} -> {len(clipped)} chars")
    return clipped


def run_gpt_revise_identity_currently(persona, plan_note, thought_note, curr_active_norms_and_utility, verbose=False):
    def create_prompt_input(persona, plan_note, thought_note, curr_active_norms_and_utility):
        prompt_input = [persona.scratch.name]
        prompt_input += [(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')]
        prompt_input += [persona.scratch.currently]
        prompt_input += [(plan_note + thought_note).replace('\n', '')]
        prompt_input += [persona.scratch.curr_time.strftime('%A %B %d')]
        prompt_input += [curr_active_norms_and_utility]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        return _cap_identity_text(gpt_response.split("Status: ")[-1],
                                  "currently")

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_compliance_prompt/revise_identity_currently_v1.txt"
    prompt_input = create_prompt_input(persona, plan_note, thought_note, curr_active_norms_and_utility)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_revise_identity_daily_plan_req(persona, curr_act_norms, verbose=False):
    def create_prompt_input(persona, curr_act_norms):
        prompt_input = [persona.scratch.get_str_iss()]
        prompt_input += [persona.scratch.curr_time.strftime('%A %B %d')]
        prompt_input += [persona.scratch.name]
        prompt_input += [curr_act_norms]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        return _cap_identity_text(gpt_response, "daily_plan_req")

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_compliance_prompt/revise_identity_daily_plan_req_v1.txt"
    prompt_input = create_prompt_input(persona, curr_act_norms)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_daily_plan_v2(persona, wake_up_hour, curr_act_norm, test_input=None, verbose=False):
    def create_prompt_input(persona, wake_up_hour, curr_act_norm, test_input=None):
        if test_input: return test_input
        prompt_input = []
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [persona.scratch.get_str_lifestyle()]
        prompt_input += [persona.scratch.get_str_curr_date_str()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [f"{str(wake_up_hour)}:00 am"]
        prompt_input += [curr_act_norm]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # See daily_plan in run_gpt_prompt.py for rationale: tolerate
        # bullets/numbered lists when the ")"+digit heuristic finds nothing.
        s = _strip_scaffolding(gpt_response)
        cr = []
        _cr = s.split(")")
        for i in _cr:
            if not i:
                continue
            if i[-1].isdigit():
                i = i[:-1].strip()
                if i and i[-1] in (".", ","):
                    cr += [i[:-1].strip()]
        if cr:
            return _cap_plan_items(cr)
        for line in s.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\-\*•]\s*", "", line)
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if line and line[-1] in (".", ","):
                line = line[:-1].rstrip()
            if line:
                cr.append(line)
        if not cr:
            raise ValueError("daily_plan_v2: could not parse any items")
        return _cap_plan_items(cr)

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt="")
        except:
            return False
        return True

    def get_fail_safe():
        fs = ['wake up and complete the morning routine at 6:00 am',
              'eat breakfast at 7:00 am',
              'read a book from 8:00 am to 12:00 pm',
              'have lunch at 12:00 pm',
              'take a nap from 1:00 pm to 4:00 pm',
              'relax and watch TV from 7:00 pm to 8:00 pm',
              'go to bed at 11:00 pm']
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 500,
                 "temperature": 1, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/norm_compliance_prompt/daily_planning_compliance_v2.txt"
    prompt_input = create_prompt_input(persona, wake_up_hour, curr_act_norm, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)

    output = ([f"wake up and complete the morning routine at {wake_up_hour}:00 am"]
              + output)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# Memoization for violation checks (Step 1a of the perf pass). Agent actions
# persist 30-360 ticks, so identical (observer, event_desc, norm) triples
# repeat hundreds of times; with temperature pinned to 0 in llm_router the
# LLM output is deterministic, so caching is lossless. Disable with
# CRSEC_MEMOIZE=0. Fail-safe results (LLM error / parse failure) are NOT
# cached so transient errors don't get frozen in.
_VIOLATION_CACHE = {}
_VIOLATION_CACHE_MAX = 50000


def run_gpt_prompt_violation_check(event_desc, norm_content, observer_name, verbose=False):
    def create_prompt_input(event_desc, norm_content, observer_name):
        prompt_input = []
        prompt_input += [event_desc]
        prompt_input += [norm_content]
        prompt_input += [observer_name]
        return prompt_input

    def _extract_json(s):
        # Strip ```json fences; then slice from first { to last } so prompt
        # leakage (e.g., a leading "Here is my analysis:" prelude from Qwen3)
        # doesn't break parsing.
        s = _strip_scaffolding(s)
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            s = s[first:last + 1]
        return json.loads(s)

    def __func_validate(gpt_response, prompt=""):
        try:
            parsed = _extract_json(gpt_response)
            for key in ("violation", "severity", "certainty", "response"):
                if key not in parsed:
                    return False
            return True
        except:
            return False

    def __func_clean_up(gpt_response, prompt=""):
        return _extract_json(gpt_response)

    def get_fail_safe():
        return {"violation": False, "severity": 0, "certainty": 0, "response": "ignore"}

    memoize = os.environ.get("CRSEC_MEMOIZE", "1") != "0"
    cache_key = (observer_name, event_desc, norm_content)
    if memoize and cache_key in _VIOLATION_CACHE:
        call_profiler.incr("violation_check_cache_hit")
        return copy.deepcopy(_VIOLATION_CACHE[cache_key])
    if memoize:
        call_profiler.incr("violation_check_cache_miss")

    gpt_param = {"engine": "gpt-4-1106-preview", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "norm/defection_prompt/violation_check_v1.txt"
    prompt_input = create_prompt_input(event_desc, norm_content, observer_name)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)
    if debug or verbose:
        print_run_prompts_norm(prompt_template, gpt_param, prompt_input, prompt, output)

    ret = output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # Cache only validated successes (fail_safe is returned by identity on
    # LLM/parse failure, so `is` distinguishes it from a real no-violation).
    if memoize and output is not fail_safe:
        if len(_VIOLATION_CACHE) >= _VIOLATION_CACHE_MAX:
            _VIOLATION_CACHE.pop(next(iter(_VIOLATION_CACHE)))
        _VIOLATION_CACHE[cache_key] = copy.deepcopy(ret)
    return ret
