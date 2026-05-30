"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: run_gpt_prompt.py
Description: Defines all run gpt prompt functions. These functions directly
interface with the safe_generate_response function.
"""
import re
import datetime
import sys
import ast

sys.path.append('../../')

from global_methods import *
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.gpt_structure import (
    _strip_scaffolding, _extract_first_int, _extract_yes_no,
)
from persona.prompt_template.print_prompt import *


# Qwen3 leniency helpers (_strip_scaffolding, _extract_first_int) and the
# fail-safe logger are defined in gpt_structure and re-exported via the
# "from persona.prompt_template.gpt_structure import *" above.


def _log_fail_safe(fn_name, fs):
    """One-line breadcrumb when a cleanup itself falls back to fail-safe.

    Used inside __func_clean_up recovery branches (the safe_* wrappers in
    gpt_structure already log when validate keeps failing).
    """
    try:
        print(f"[FAIL_SAFE] {fn_name}: returning {fs!r}", file=sys.stderr)
    except Exception:
        pass


def get_random_alphanumeric(i=6, j=6):
    """
    Returns a random alpha numeric strength that has the length of somewhere
    between i and j.

    INPUT:
      i: min_range for the length
      j: max_range for the length
    OUTPUT:
      an alpha numeric str with the length of somewhere between i and j.
    """
    k = random.randint(i, j)
    x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return x


##############################################################################
# CHAPTER 1: Run GPT Prompt
##############################################################################

def run_gpt_prompt_wake_up_hour(persona, test_input=None, verbose=False):
    """
    Given the persona, returns an integer that indicates the hour when the
    persona wakes up.

    INPUT:
      persona: The Persona class instance
    OUTPUT:
      integer for the wake up hour.
    """

    def create_prompt_input(persona, test_input=None):
        if test_input: return test_input
        prompt_input = [persona.scratch.get_str_iss(),
                        persona.scratch.get_str_lifestyle(),
                        persona.scratch.get_str_firstname()]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # GPT-4 emits a bare "7am" / "7 am". Qwen3 may emit "7 AM",
        # "Answer: 7am", "I'd say 7", or just "7". Strip scaffolding and
        # extract the first integer in [0, 24).
        s = _strip_scaffolding(gpt_response)
        # Original strict path first (preserves GPT-4 baseline behavior).
        try:
            cr = int(s.strip().lower().split("am")[0])
            if 0 <= cr < 24:
                return cr
        except Exception:
            pass
        # Lenient path: first integer in the response.
        n = _extract_first_int(s)
        if n is not None and 0 <= n < 24:
            return n
        raise ValueError(f"wake_up_hour: could not parse hour from {gpt_response!r}")

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt="")
        except:
            return False
        return True

    def get_fail_safe():
        fs = 8
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 5,
                 "temperature": 0.8, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.txt"
    prompt_input = create_prompt_input(persona, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_daily_plan(persona,
                              wake_up_hour,
                              test_input=None,
                              verbose=False):
    """
    Basically the long term planning that spans a day. Returns a list of actions
    that the persona will take today. Usually comes in the following form:
    'wake up and complete the morning routine at 6:00 am',
    'eat breakfast at 7:00 am',..
    Note that the actions come without a period.

    INPUT:
      persona: The Persona class instance
    OUTPUT:
      a list of daily actions in broad strokes.
    """

    def create_prompt_input(persona, wake_up_hour, test_input=None):
        if test_input: return test_input
        prompt_input = []
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [persona.scratch.get_str_lifestyle()]
        prompt_input += [persona.scratch.get_str_curr_date_str()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [f"{str(wake_up_hour)}:00 am"]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # GPT-4 emits lines like "1) eat breakfast at 7:00 am" — split on
        # ")" and check trailing digit of the next item to find boundaries.
        # Qwen3 may use "1.", bullets, or no numbering; if the GPT-4 form
        # extracts nothing, fall back to bullet/numbered line parsing.
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
            return cr
        # Lenient fallback: numbered or bulleted lines.
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
            raise ValueError("daily_plan: could not parse any items")
        return cr

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

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 500,
                 "temperature": 1, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/daily_planning_v6.txt"
    prompt_input = create_prompt_input(persona, wake_up_hour, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    output = ([f"wake up and complete the morning routine at {wake_up_hour}:00 am"]
              + output)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_generate_hourly_schedule(persona,
                                            curr_hour_str,
                                            p_f_ds_hourly_org,
                                            hour_str,
                                            curr_act_norms,
                                            intermission2=None,
                                            test_input=None,
                                            verbose=False):
    def create_prompt_input(persona,
                            curr_hour_str,
                            p_f_ds_hourly_org,
                            hour_str,
                            curr_act_norms,
                            intermission2=None,
                            test_input=None):
        if test_input: return test_input
        schedule_format = ""
        for i in hour_str:
            schedule_format += f"[{persona.scratch.get_str_curr_date_str()} -- {i}]"
            schedule_format += f" Activity: [Fill in]\n"
        schedule_format = schedule_format[:-1]

        intermission_str = f"Here the originally intended hourly breakdown of"
        intermission_str += f" {persona.scratch.get_str_firstname()}'s schedule today: "
        for count, i in enumerate(persona.scratch.daily_req):
            intermission_str += f"{str(count + 1)}) {i}, "
        intermission_str = intermission_str[:-2]

        prior_schedule = ""
        if p_f_ds_hourly_org:
            prior_schedule = "\n"
            for count, i in enumerate(p_f_ds_hourly_org):
                prior_schedule += f"[(ID:{get_random_alphanumeric()})"
                prior_schedule += f" {persona.scratch.get_str_curr_date_str()} --"
                prior_schedule += f" {hour_str[count]}] Activity:"
                prior_schedule += f" {persona.scratch.get_str_firstname()}"
                prior_schedule += f" is {i}\n"

        prompt_ending = f"[(ID:{get_random_alphanumeric()})"
        prompt_ending += f" {persona.scratch.get_str_curr_date_str()}"
        prompt_ending += f" -- {curr_hour_str}] Activity:"
        prompt_ending += f" {persona.scratch.get_str_firstname()} is"

        if intermission2:
            intermission2 = f"\n{intermission2}"

        prompt_input = []
        prompt_input += [schedule_format]
        prompt_input += [persona.scratch.get_str_iss()]

        prompt_input += [prior_schedule + "\n"]
        prompt_input += [intermission_str]
        if intermission2:
            prompt_input += [intermission2]
        else:
            prompt_input += [""]
        prompt_input += [prompt_ending]
        prompt_input += [curr_act_norms]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # Qwen3 may echo "Answer:" or wrap in fences; strip first.
        cr = _strip_scaffolding(gpt_response).strip()
        if not cr:
            raise ValueError("generate_hourly_schedule: empty response")
        if cr[-1] == ".":
            cr = cr[:-1]
        cr = cr.split("Activity:")[-1]
        index = cr.find(" is ")
        if index >-1:
            cr=cr[index+4:]
        #cr = cr.split(" is ")[-1]
        return cr

    def __func_validate(gpt_response, prompt=""):
        print("generate_hourly_schedule_ours_vvvvvvvvvvvv1:", gpt_response)
        try:
            __func_clean_up(gpt_response, prompt="")
        except:
            return False
        return True

    def get_fail_safe():
        fs = "asleep"
        return fs

    # # ChatGPT Plugin ===========================================================
    # def __chat_func_clean_up(gpt_response, prompt=""): ############
    #   cr = gpt_response.strip()
    #   if cr[-1] == ".":
    #     cr = cr[:-1]
    #   return cr

    # def __chat_func_validate(gpt_response, prompt=""): ############
    #   try: __func_clean_up(gpt_response, prompt="")
    #   except: return False
    #   return True

    # print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 10") ########
    # gpt_param = {"engine": "", "max_tokens": 15,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v3_ChatGPT/generate_hourly_schedule_v2.txt" ########
    # prompt_input = create_prompt_input(persona,
    #                                    curr_hour_str,
    #                                    p_f_ds_hourly_org,
    #                                    hour_str,
    #                                    intermission2,
    #                                    test_input)  ########
    # prompt = generate_prompt(prompt_input, prompt_template)
    # example_output = "studying for her music classes" ########
    # special_instruction = "The output should ONLY include the part of the sentence that completes the last line in the schedule above." ########
    # fail_safe = get_fail_safe() ########
    # output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
    #                                         __chat_func_validate, __chat_func_clean_up, True)
    # if output != False:
    #   return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "gpt-4-1106-preview", "max_tokens": 50,
                 "temperature": 0.5, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    # prompt_template = "persona/prompt_template/v2/generate_hourly_schedule_v2.txt"
    prompt_template = "norm/norm_compliance_prompt/generate_hourly_schedule_ours_v1.txt"
    prompt_input = create_prompt_input(persona,
                                       curr_hour_str,
                                       p_f_ds_hourly_org,
                                       hour_str,
                                       curr_act_norms,
                                       intermission2,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                 __func_validate, __func_clean_up)

    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_task_decomp(persona,
                               task,
                               duration,
                               curr_act_norms,
                               test_input=None,
                               verbose=False):
    def create_prompt_input(persona, task, duration, curr_act_norms, test_input=None):

        """
        Today is Saturday June 25. From 00:00 ~ 06:00am, Maeve is
        planning on sleeping, 06:00 ~ 07:00am, Maeve is
        planning on waking up and doing her morning routine,
        and from 07:00am ~08:00am, Maeve is planning on having breakfast.
        """

        curr_f_org_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
        all_indices = []
        # if curr_f_org_index > 0:
        #   all_indices += [curr_f_org_index-1]
        all_indices += [curr_f_org_index]
        if curr_f_org_index + 1 <= len(persona.scratch.f_daily_schedule_hourly_org):
            all_indices += [curr_f_org_index + 1]
        if curr_f_org_index + 2 <= len(persona.scratch.f_daily_schedule_hourly_org):
            all_indices += [curr_f_org_index + 2]

        curr_time_range = ""

        print("DEBUG")
        print(persona.scratch.f_daily_schedule_hourly_org)
        print(all_indices)

        summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. '
        summ_str += f'From '
        for index in all_indices:
            print("index", index)
            if index < len(persona.scratch.f_daily_schedule_hourly_org):
                start_min = 0
                for i in range(index):
                    start_min += persona.scratch.f_daily_schedule_hourly_org[i][1]
                end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[index][1]
                start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                              + datetime.timedelta(minutes=start_min))
                end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                            + datetime.timedelta(minutes=end_min))
                start_time_str = start_time.strftime("%H:%M%p")
                end_time_str = end_time.strftime("%H:%M%p")
                summ_str += f"{start_time_str} ~ {end_time_str}, {persona.name} is planning on {persona.scratch.f_daily_schedule_hourly_org[index][0]}, "
                if curr_f_org_index + 1 == index:
                    curr_time_range = f'{start_time_str} ~ {end_time_str}'
        summ_str = summ_str[:-2] + "."

        prompt_input = []
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [summ_str]
        # prompt_input += [persona.scratch.get_str_curr_date_str()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [task]
        prompt_input += [curr_time_range]
        prompt_input += [duration]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [curr_act_norms]

        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print("TOODOOOOOO")
        print(gpt_response)
        print("-==- -==- -==- ")
        # Strip Qwen3 prompt-leakage (Answer:, code fences, etc.).
        gpt_response = _strip_scaffolding(gpt_response, persona.scratch.get_str_firstname())
        gpt_response=gpt_response.split('\n\n')[0]
        gpt_response=gpt_response.split("1. "+persona.scratch.get_str_firstname()+" is ")[-1]

        print("TOODOOOOOO")
        print(gpt_response)
        print("-==- -==- -==- ")

        # TODO SOMETHING HERE sometimes fails... See screenshot
        temp = [i.strip() for i in gpt_response.split("\n") if i.strip()]
        print("temppppppppppppppppppppppppppp: ",temp)
        _cr = []
        cr = []
        for count, i in enumerate(temp):
            if count != 0:
                _cr += [" ".join([j.strip() for j in i.split(" ")][3:])]
            else:
                _cr += [i]
        print("_crrrrrrrrrrrrrrr: ",_cr)

        total_expected_min = int(prompt.split("(total duration in minutes")[-1]
                                 .split("):")[0].strip())

        # GPT-4 path: every line carries "(duration in minutes: N, ...)".
        # Qwen3 path: lines may be bare numbered tasks with no duration tail.
        # If ANY line lacks the duration tail, fall back to evenly splitting
        # the expected total across the N tasks (remainder to last).
        has_durations = all("(duration in minutes:" in i for i in _cr if i)
        if has_durations:
            for count, i in enumerate(_cr):
                k = [j.strip() for j in i.split("(duration in minutes:")]
                task = k[0]
                if task and task[-1] == ".":
                    task = task[:-1]
                duration = int(k[1].split(",")[0].strip())
                cr += [[task, duration]]
        else:
            _log_fail_safe("task_decomp.__func_clean_up", "missing duration annotations; even-splitting")
            n_tasks = max(1, len(_cr))
            base = (total_expected_min // n_tasks)
            # snap to 5-min increments to match GPT-4 expectations downstream
            base = max(5, base - (base % 5))
            durations = [base] * n_tasks
            durations[-1] = total_expected_min - base * (n_tasks - 1)
            for count, i in enumerate(_cr):
                task = i.strip()
                # strip leading numbering like "2. " if still present
                task = re.sub(r"^\d+[\.\)]\s*", "", task).strip()
                if task and task[-1] in ".,":
                    task = task[:-1]
                cr += [[task, durations[count]]]

        print("crrrrrrrrrrrrrrr: ",cr)

        # TODO -- now, you need to make sure that this is the same as the sum of
        #         the current action sequence.
        curr_min_slot = [["dummy", -1], ]  # (task_name, task_index)
        for count, i in enumerate(cr):
            i_task = i[0]
            i_duration = i[1]

            i_duration -= (i_duration % 5)
            if i_duration > 0:
                for j in range(i_duration):
                    curr_min_slot += [(i_task, count)]
        curr_min_slot = curr_min_slot[1:]

        if len(curr_min_slot) > total_expected_min:
            last_task = curr_min_slot[60]
            for i in range(1, 6):
                curr_min_slot[-1 * i] = last_task
        elif len(curr_min_slot) < total_expected_min:
            last_task = curr_min_slot[-1]
            for i in range(total_expected_min - len(curr_min_slot)):
                curr_min_slot += [last_task]

        cr_ret = [["dummy", -1], ]
        for task, task_index in curr_min_slot:
            if task != cr_ret[-1][0]:
                cr_ret += [[task, 1]]
            else:
                cr_ret[-1][1] += 1
        cr = cr_ret[1:]

        return cr

    def __func_validate(gpt_response, prompt=""):
        # TODO -- this sometimes generates error
        try:
            __func_clean_up(gpt_response)
        except:
            pass
            # return False
        return gpt_response

    def get_fail_safe():
        fs = [["asleep", duration]]
        return fs

    gpt_param = {"engine": "gpt-4-1106-preview", "max_tokens": 1000,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/task_decomp_v3.txt"
    # prompt_template = "norm/plan_prompt/task_decomp_ours_v1.txt"
    prompt_template = "norm/norm_compliance_prompt/task_decomp_compliance_v2.txt"
    prompt_input = create_prompt_input(persona, task, duration, curr_act_norms)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    print("?????")
    print(prompt)
    #output = safe_generate_response(prompt, gpt_param, 5, get_fail_safe(),
    #                                __func_validate, __func_clean_up)
    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)

    # TODO THERE WAS A BUG HERE...
    # This is for preventing overflows...
    """
    File "/Users/joonsungpark/Desktop/Stanford/Projects/
    generative-personas/src_exploration/reverie_simulation/
    brain/get_next_action_v3.py", line 364, in run_gpt_prompt_task_decomp
    fin_output[-1][1] += (duration - ftime_sum)
    IndexError: list index out of range
    """

    print("IMPORTANT VVV DEBUG")

    # print (prompt_input)
    # print (prompt)
    print(output)

    fin_output = []
    time_sum = 0
    for i_task, i_duration in output:
        time_sum += i_duration
        # HM?????????
        # if time_sum < duration:
        if time_sum <= duration:
            fin_output += [[i_task, i_duration]]
        else:
            break
    ftime_sum = 0
    for fi_task, fi_duration in fin_output:
        ftime_sum += fi_duration

    # print ("for debugging... line 365", fin_output)
    fin_output[-1][1] += (duration - ftime_sum)
    output = fin_output

    task_decomp = output
    ret = []
    for decomp_task, duration in task_decomp:
        ret += [[f"{task} ({decomp_task})", duration]]
    output = ret

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

def run_gpt_prompt_task_decomp_v2(persona,
                               task,
                               duration,
                               curr_act_norms,
                               test_input=None,
                               verbose=False):
    def create_prompt_input(persona, task, duration, curr_act_norms, test_input=None):

        """
        Today is Saturday June 25. From 00:00 ~ 06:00am, Maeve is
        planning on sleeping, 06:00 ~ 07:00am, Maeve is
        planning on waking up and doing her morning routine,
        and from 07:00am ~08:00am, Maeve is planning on having breakfast.
        """

        curr_f_org_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
        all_indices = []
        # if curr_f_org_index > 0:
        #   all_indices += [curr_f_org_index-1]
        all_indices += [curr_f_org_index]
        if curr_f_org_index + 1 <= len(persona.scratch.f_daily_schedule_hourly_org):
            all_indices += [curr_f_org_index + 1]
        if curr_f_org_index + 2 <= len(persona.scratch.f_daily_schedule_hourly_org):
            all_indices += [curr_f_org_index + 2]

        curr_time_range = ""

        print("DEBUG")
        print(persona.scratch.f_daily_schedule_hourly_org)
        print(all_indices)

        summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. '
        summ_str += f'From '
        for index in all_indices:
            print("index", index)
            if index < len(persona.scratch.f_daily_schedule_hourly_org):
                start_min = 0
                for i in range(index):
                    start_min += persona.scratch.f_daily_schedule_hourly_org[i][1]
                end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[index][1]
                start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                              + datetime.timedelta(minutes=start_min))
                end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                            + datetime.timedelta(minutes=end_min))
                start_time_str = start_time.strftime("%H:%M%p")
                end_time_str = end_time.strftime("%H:%M%p")
                summ_str += f"{start_time_str} ~ {end_time_str}, {persona.name} is planning on {persona.scratch.f_daily_schedule_hourly_org[index][0]}, "
                if curr_f_org_index == index:
                    curr_time_range = f'{start_time_str} ~ {end_time_str}'
        summ_str = summ_str[:-2] + "."

        prompt_input = []
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [summ_str]
        # prompt_input += [persona.scratch.get_str_curr_date_str()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [task]
        prompt_input += [curr_time_range]
        prompt_input += [duration]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [curr_act_norms]

        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print("TOODOOOOOO")
        print(gpt_response)
        print("-==- -==- -==- ")
        # Strip Qwen3 prompt-leakage (Answer:, code fences, etc.).
        gpt_response = _strip_scaffolding(gpt_response, persona.scratch.get_str_firstname())
        gpt_response=gpt_response.split('\n\n')[0]
        gpt_response=gpt_response.split("1. "+persona.scratch.get_str_firstname()+" is ")[-1]

        print("TOODOOOOOO")
        print(gpt_response)
        print("-==- -==- -==- ")

        # TODO SOMETHING HERE sometimes fails... See screenshot
        temp = [i.strip() for i in gpt_response.split("\n") if i.strip()]
        print("temppppppppppppppppppppppppppp: ",temp)
        _cr = []
        cr = []
        for count, i in enumerate(temp):
            if count != 0:
                _cr += [" ".join([j.strip() for j in i.split(" ")][3:])]
            else:
                _cr += [i]
        print("_crrrrrrrrrrrrrrr: ",_cr)

        total_expected_min = int(prompt.split("(total duration in minutes")[-1]
                                 .split("):")[0].strip())

        # See task_decomp v1 for rationale: tolerate Qwen3 outputs that omit
        # "(duration in minutes: ...)" by evenly dividing the expected total.
        has_durations = all("(duration in minutes:" in i for i in _cr if i)
        if has_durations:
            for count, i in enumerate(_cr):
                k = [j.strip() for j in i.split("(duration in minutes:")]
                task = k[0]
                if task and task[-1] == ".":
                    task = task[:-1]
                duration = int(k[1].split(",")[0].strip())
                cr += [[task, duration]]
        else:
            _log_fail_safe("task_decomp_v2.__func_clean_up", "missing duration annotations; even-splitting")
            n_tasks = max(1, len(_cr))
            base = (total_expected_min // n_tasks)
            base = max(5, base - (base % 5))
            durations = [base] * n_tasks
            durations[-1] = total_expected_min - base * (n_tasks - 1)
            for count, i in enumerate(_cr):
                task = i.strip()
                task = re.sub(r"^\d+[\.\)]\s*", "", task).strip()
                if task and task[-1] in ".,":
                    task = task[:-1]
                cr += [[task, durations[count]]]
        print("crrrrrrrrrrrrrrr: ",_cr)

        # TODO -- now, you need to make sure that this is the same as the sum of
        #         the current action sequence.
        curr_min_slot = [["dummy", -1], ]  # (task_name, task_index)
        for count, i in enumerate(cr):
            i_task = i[0]
            i_duration = i[1]

            i_duration -= (i_duration % 5)
            if i_duration > 0:
                for j in range(i_duration):
                    curr_min_slot += [(i_task, count)]
        curr_min_slot = curr_min_slot[1:]

        if len(curr_min_slot) > total_expected_min:
            last_task = curr_min_slot[60]
            for i in range(1, 6):
                curr_min_slot[-1 * i] = last_task
        elif len(curr_min_slot) < total_expected_min:
            last_task = curr_min_slot[-1]
            for i in range(total_expected_min - len(curr_min_slot)):
                curr_min_slot += [last_task]

        cr_ret = [["dummy", -1], ]
        for task, task_index in curr_min_slot:
            if task != cr_ret[-1][0]:
                cr_ret += [[task, 1]]
            else:
                cr_ret[-1][1] += 1
        cr = cr_ret[1:]

        return cr

    def __func_validate(gpt_response, prompt=""):
        # TODO -- this sometimes generates error
        try:
            __func_clean_up(gpt_response)
        except:
            pass
            # return False
        return gpt_response

    def get_fail_safe():
        fs = [["asleep", duration]]
        return fs

    gpt_param = {"engine": "gpt-4-1106-preview", "max_tokens": 1000,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/task_decomp_v3.txt"
    # prompt_template = "norm/plan_prompt/task_decomp_ours_v1.txt"
    prompt_template = "norm/norm_compliance_prompt/task_decomp_compliance_v2.txt"
    prompt_input = create_prompt_input(persona, task, duration, curr_act_norms)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    print("?????")
    print(prompt)
    #output = safe_generate_response(prompt, gpt_param, 5, get_fail_safe(),
    #                                __func_validate, __func_clean_up)
    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)

    # TODO THERE WAS A BUG HERE...
    # This is for preventing overflows...
    """
    File "/Users/joonsungpark/Desktop/Stanford/Projects/
    generative-personas/src_exploration/reverie_simulation/
    brain/get_next_action_v3.py", line 364, in run_gpt_prompt_task_decomp
    fin_output[-1][1] += (duration - ftime_sum)
    IndexError: list index out of range
    """

    print("IMPORTANT VVV DEBUG")

    # print (prompt_input)
    # print (prompt)
    print(output)

    fin_output = []
    time_sum = 0
    for i_task, i_duration in output:
        time_sum += i_duration
        # HM?????????
        # if time_sum < duration:
        if time_sum <= duration:
            fin_output += [[i_task, i_duration]]
        else:
            break
    ftime_sum = 0
    for fi_task, fi_duration in fin_output:
        ftime_sum += fi_duration

    # print ("for debugging... line 365", fin_output)
    fin_output[-1][1] += (duration - ftime_sum)
    output = fin_output

    task_decomp = output
    ret = []
    for decomp_task, duration in task_decomp:
        ret += [[f"{task} ({decomp_task})", duration]]
    output = ret

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_action_sector(action_description,
                                 persona,
                                 maze,
                                 test_input=None,
                                 verbose=False):
    def create_prompt_input(action_description, persona, maze, test_input=None):
        act_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"

        prompt_input = []

        prompt_input += [persona.scratch.get_str_name()]
        prompt_input += [persona.scratch.living_area.split(":")[1]]
        x = f"{act_world}:{persona.scratch.living_area.split(':')[1]}"
        prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]

        prompt_input += [persona.scratch.get_str_name()]
        prompt_input += [f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"]
        x = f"{act_world}:{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]

        if persona.scratch.get_str_daily_plan_req() != "":
            prompt_input += [f"\n{persona.scratch.get_str_daily_plan_req()}"]
        else:
            prompt_input += [""]

        # MAR 11 TEMP
        accessible_sector_str = persona.s_mem.get_str_accessible_sectors(act_world)
        curr = accessible_sector_str.split(", ")
        fin_accessible_sectors = []
        for i in curr:
            if "'s house" in i:
                if persona.scratch.last_name in i:
                    fin_accessible_sectors += [i]
            else:
                fin_accessible_sectors += [i]
        accessible_sector_str = ", ".join(fin_accessible_sectors)
        # END MAR 11 TEMP

        prompt_input += [accessible_sector_str]

        action_description_1 = action_description
        action_description_2 = action_description
        if "(" in action_description:
            action_description_1 = action_description.split("(")[0].strip()
            action_description_2 = action_description.split("(")[-1][:-1]
        prompt_input += [persona.scratch.get_str_name()]
        prompt_input += [action_description_1]

        prompt_input += [action_description_2]
        prompt_input += [persona.scratch.get_str_name()]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # GPT-4 produces "{kitchen}". Qwen3 may produce "kitchen", "{kitchen",
        # "Answer: {kitchen}", or "Answer: {kitchen". Strip scaffolding first,
        # then take everything before "}" if present, else the whole thing.
        s = _strip_scaffolding(gpt_response, persona.scratch.get_str_name())
        # The "Answer: {" prefix may leave a stray "{" past _strip_scaffolding
        # if it co-occurs with content; drop it explicitly.
        if s.startswith("{"):
            s = s[1:]
        cleaned_response = s.split("}")[0].split("\n")[0].strip()
        # Drop trailing punctuation that Qwen3 sometimes appends.
        while cleaned_response and cleaned_response[-1] in ".,;:":
            cleaned_response = cleaned_response[:-1].rstrip()
        return cleaned_response

    def __func_validate(gpt_response, prompt=""):
        # Strict GPT-4 path required "}" and disallowed ",". Qwen3 often omits
        # the closing brace. Accept either form; reject commas (multi-answer).
        s = _strip_scaffolding(gpt_response)
        if len(s.strip()) < 1:
            return False
        if "," in s.split("}")[0]:
            return False
        return True

    def get_fail_safe():
        fs = ("kitchen")
        return fs

    # # ChatGPT Plugin ===========================================================
    # def __chat_func_clean_up(gpt_response, prompt=""): ############
    #   cr = gpt_response.strip()
    #   return cr

    # def __chat_func_validate(gpt_response, prompt=""): ############
    #   try:
    #     gpt_response = __func_clean_up(gpt_response, prompt="")
    #   except:
    #     return False
    #   return True

    # print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 20") ########
    # gpt_param = {"engine": "", "max_tokens": 15,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v3_ChatGPT/action_location_sector_v2.txt" ########
    # prompt_input = create_prompt_input(action_description, persona, maze)  ########
    # prompt = generate_prompt(prompt_input, prompt_template)
    # example_output = "Johnson Park" ########
    # special_instruction = "The value for the output must contain one of the area options above verbatim (including lower/upper case)." ########
    # fail_safe = get_fail_safe() ########
    # output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
    #                                         __chat_func_validate, __chat_func_clean_up, True)
    # if output != False:
    #   return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v1/action_location_sector_v1.txt"
    prompt_input = create_prompt_input(action_description, persona, maze)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    y = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
    x = [i.strip() for i in persona.s_mem.get_str_accessible_sectors(y).split(",")]
    if output not in x:
        # output = random.choice(x)
        output = persona.scratch.living_area.split(":")[1]

    print("DEBUG", random.choice(x), "------", output)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_action_arena(action_description,
                                persona,
                                maze, act_world, act_sector,
                                test_input=None,
                                verbose=False):
    def create_prompt_input(action_description, persona, maze, act_world, act_sector, test_input=None):
        prompt_input = []
        # prompt_input += [persona.scratch.get_str_name()]
        # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["arena"]]
        # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["sector"]]
        prompt_input += [persona.scratch.get_str_name()]
        x = f"{act_world}:{act_sector}"
        prompt_input += [act_sector]

        # MAR 11 TEMP
        accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(x)
        curr = accessible_arena_str.split(", ")
        fin_accessible_arenas = []
        for i in curr:
            if "'s room" in i:
                if persona.scratch.last_name in i:
                    fin_accessible_arenas += [i]
            else:
                fin_accessible_arenas += [i]
        accessible_arena_str = ", ".join(fin_accessible_arenas)
        # END MAR 11 TEMP

        prompt_input += [accessible_arena_str]

        action_description_1 = action_description
        action_description_2 = action_description
        if "(" in action_description:
            action_description_1 = action_description.split("(")[0].strip()
            action_description_2 = action_description.split("(")[-1][:-1]
        prompt_input += [persona.scratch.get_str_name()]
        prompt_input += [action_description_1]

        prompt_input += [action_description_2]
        prompt_input += [persona.scratch.get_str_name()]

        prompt_input += [act_sector]

        prompt_input += [accessible_arena_str]
        # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["arena"]]
        # x = f"{maze.access_tile(persona.scratch.curr_tile)['world']}:{maze.access_tile(persona.scratch.curr_tile)['sector']}:{maze.access_tile(persona.scratch.curr_tile)['arena']}"
        # prompt_input += [persona.s_mem.get_str_accessible_arena_game_objects(x)]

        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # Same Qwen3 leniency as action_sector: tolerate missing closing brace
        # and "Answer: {" leakage. Returned value flows into arena.split(":")
        # downstream, so stripping the stray "{" matters.
        s = _strip_scaffolding(gpt_response, persona.scratch.get_str_name())
        if s.startswith("{"):
            s = s[1:]
        cleaned_response = s.split("}")[0].split("\n")[0].strip()
        while cleaned_response and cleaned_response[-1] in ".,;:":
            cleaned_response = cleaned_response[:-1].rstrip()
        return cleaned_response

    def __func_validate(gpt_response, prompt=""):
        s = _strip_scaffolding(gpt_response)
        if len(s.strip()) < 1:
            return False
        if "," in s.split("}")[0]:
            return False
        return True

    def get_fail_safe():
        fs = ("kitchen")
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v1/action_location_object_vMar11.txt"
    prompt_input = create_prompt_input(action_description, persona, maze, act_world, act_sector)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    print(output)
    # y = f"{act_world}:{act_sector}"
    # x = [i.strip() for i in persona.s_mem.get_str_accessible_sector_arenas(y).split(",")]
    # if output not in x:
    #   output = random.choice(x)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_action_game_object(action_description,
                                      persona,
                                      maze,
                                      temp_address,
                                      test_input=None,
                                      verbose=False):
    def create_prompt_input(action_description,
                            persona,
                            temp_address,
                            test_input=None):
        prompt_input = []
        if "(" in action_description:
            action_description = action_description.split("(")[-1][:-1]

        prompt_input += [action_description]
        prompt_input += [persona
                             .s_mem.get_str_accessible_arena_game_objects(temp_address)]
        return prompt_input

    def __func_validate(gpt_response, prompt=""):
        if len(gpt_response.strip()) < 1:
            return False
        return True

    def __func_clean_up(gpt_response, prompt=""):
        cleaned_response = gpt_response.strip()
        return cleaned_response

    def get_fail_safe():
        fs = ("bed")
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v1/action_object_v2.txt"
    prompt_input = create_prompt_input(action_description,
                                       persona,
                                       temp_address,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    x = [i.strip() for i in persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(",")]
    if output not in x:
        output = random.choice(x)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_pronunciatio(action_description, persona, verbose=False):
    def create_prompt_input(action_description):
        if "(" in action_description:
            action_description = action_description.split("(")[-1].split(")")[0]
        prompt_input = [action_description]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = gpt_response.strip()
        if len(cr) > 3:
            cr = cr[:3]
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt="")
            if len(gpt_response) == 0:
                return False
        except:
            return False
        return True

    def get_fail_safe():
        fs = "😋"
        return fs

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        cr = gpt_response.strip()
        if len(cr) > 3:
            cr = cr[:3]
        return cr

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt="")
            if len(gpt_response) == 0:
                return False
        except:
            return False
        return True
        return True

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 4")  ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.txt"  ########
    prompt_input = create_prompt_input(action_description)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "🛁🧖‍♀️"  ########
    special_instruction = "The value for the output must ONLY contain the emojis."  ########
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "", "max_tokens": 15,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    # prompt_template = "persona/prompt_template/v2/generate_pronunciatio_v1.txt"
    # prompt_input = create_prompt_input(action_description)

    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_event_triple(action_description, persona, verbose=False):
    def create_prompt_input(action_description, persona):
        if "(" in action_description:
            action_description = action_description.split("(")[-1].split(")")[0]
        prompt_input = [persona.name,
                        action_description,
                        persona.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # GPT-4: "(predicate, object)". Qwen3 may omit parens or wrap in
        # "Answer: (..)". Strip scaffolding and stray leading "(" before
        # splitting.
        cr = _strip_scaffolding(gpt_response).strip()
        if cr.startswith("("):
            cr = cr[1:]
        cr = [i.strip() for i in cr.split(")")[0].split(",")]
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            gpt_response = __func_clean_up(gpt_response, prompt="")
            if len(gpt_response) != 2:
                return False
        except:
            return False
        return True

    def get_fail_safe(persona):
        fs = (persona.name, "is", "idle")
        return fs

    # ChatGPT Plugin ===========================================================
    # def __chat_func_clean_up(gpt_response, prompt=""): ############
    #   cr = gpt_response.strip()
    #   cr = [i.strip() for i in cr.split(")")[0].split(",")]
    #   return cr

    # def __chat_func_validate(gpt_response, prompt=""): ############
    #   try:
    #     gpt_response = __func_clean_up(gpt_response, prompt="")
    #     if len(gpt_response) != 2:
    #       return False
    #   except: return False
    #   return True

    # print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 5") ########
    # gpt_param = {"engine": "", "max_tokens": 15,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v3_ChatGPT/generate_event_triple_v1.txt" ########
    # prompt_input = create_prompt_input(action_description, persona)  ########
    # prompt = generate_prompt(prompt_input, prompt_template)
    # example_output = "(Jane Doe, cooking, breakfast)" ########
    # special_instruction = "The value for the output must ONLY contain the triple. If there is an incomplete element, just say 'None' but there needs to be three elements no matter what." ########
    # fail_safe = get_fail_safe(persona) ########
    # output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
    #                                         __chat_func_validate, __chat_func_clean_up, True)
    # if output != False:
    #   return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "gpt-4-1106-preview", "max_tokens": 30,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt"
    prompt_input = create_prompt_input(action_description, persona)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe(persona)  ########
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    # __func_validate, __func_clean_up)
    output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
                                             __func_validate, __func_clean_up)
    output = (persona.name, output[0], output[1])

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona, verbose=False):
    def create_prompt_input(act_game_object, act_desp, persona):
        prompt_input = [act_game_object,
                        persona.name,
                        act_desp,
                        act_game_object,
                        act_game_object]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = _strip_scaffolding(gpt_response).strip()
        if not cr:
            raise ValueError("act_obj_desc: empty response")
        if cr[-1] == ".": cr = cr[:-1]
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            gpt_response = __func_clean_up(gpt_response, prompt="")
        except:
            return False
        return True

    def get_fail_safe(act_game_object):
        fs = f"{act_game_object} is idle"
        return fs

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        cr = _strip_scaffolding(gpt_response).strip()
        if not cr:
            raise ValueError("act_obj_desc(chat): empty response")
        if cr[-1] == ".": cr = cr[:-1]
        return cr

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            gpt_response = __func_clean_up(gpt_response, prompt="")
        except:
            return False
        return True

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 6")  ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/generate_obj_event_v1.txt"  ########
    prompt_input = create_prompt_input(act_game_object, act_desp, persona)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "being fixed"  ########
    special_instruction = "The output should ONLY contain the phrase that should go in <fill in>."  ########
    fail_safe = get_fail_safe(act_game_object)  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    import sys; print("[FAIL_SAFE] run_gpt_prompt_act_obj_desc: ChatGPT path returned False, using fail_safe", file=sys.stderr)
    return fail_safe, [fail_safe, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "", "max_tokens": 30,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    # prompt_template = "persona/prompt_template/v2/generate_obj_event_v1.txt"
    # prompt_input = create_prompt_input(act_game_object, act_desp, persona)
    # prompt = generate_prompt(prompt_input, prompt_template)
    # fail_safe = get_fail_safe(act_game_object)
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona, verbose=False):
    def create_prompt_input(act_game_object, act_obj_desc):
        prompt_input = [act_game_object,
                        act_obj_desc,
                        act_game_object]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = _strip_scaffolding(gpt_response).strip()
        if cr.startswith("("):
            cr = cr[1:]
        cr = [i.strip() for i in cr.split(")")[0].split(",")]
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            gpt_response = __func_clean_up(gpt_response, prompt="")
            if len(gpt_response) != 2:
                return False
        except:
            return False
        return True

    def get_fail_safe(act_game_object):
        fs = (act_game_object, "is", "idle")
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 30,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt"
    prompt_input = create_prompt_input(act_game_object, act_obj_desc)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe(act_game_object)
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    output = (act_game_object, output[0], output[1])

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_new_decomp_schedule(persona,
                                       main_act_dur,
                                       truncated_act_dur,
                                       start_time_hour,
                                       end_time_hour,
                                       inserted_act,
                                       inserted_act_dur,
                                       test_input=None,
                                       verbose=False):
    def create_prompt_input(persona,
                            main_act_dur,
                            truncated_act_dur,
                            start_time_hour,
                            end_time_hour,
                            inserted_act,
                            inserted_act_dur,
                            test_input=None):
        persona_name = persona.name
        start_hour_str = start_time_hour.strftime("%H:%M %p")
        end_hour_str = end_time_hour.strftime("%H:%M %p")

        original_plan = ""
        for_time = start_time_hour
        for i in main_act_dur:
            original_plan += f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + \
                             i[0]
            original_plan += "\n"
            for_time += datetime.timedelta(minutes=int(i[1]))

        new_plan_init = ""
        for_time = start_time_hour
        for count, i in enumerate(truncated_act_dur):
            new_plan_init += f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + \
                             i[0]
            new_plan_init += "\n"
            if count < len(truncated_act_dur) - 1:
                for_time += datetime.timedelta(minutes=int(i[1]))

        new_plan_init += (for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M") + " ~"

        prompt_input = [persona_name,
                        start_hour_str,
                        end_hour_str,
                        original_plan,
                        persona_name,
                        inserted_act,
                        inserted_act_dur,
                        persona_name,
                        start_hour_str,
                        end_hour_str,
                        end_hour_str,
                        new_plan_init]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        new_schedule = prompt + " " + gpt_response.strip()
        new_schedule = new_schedule.split("The revised schedule:")[-1].strip()
        new_schedule = new_schedule.split("\n")

        ret_temp = []
        for i in new_schedule:
            ret_temp += [i.split(" -- ")]

        ret = []
        for time_str, action in ret_temp:
            start_time = time_str.split(" ~ ")[0].strip()
            end_time = time_str.split(" ~ ")[1].strip()
            delta = datetime.datetime.strptime(end_time, "%H:%M") - datetime.datetime.strptime(start_time, "%H:%M")
            delta_min = int(delta.total_seconds() / 60)
            if delta_min < 0: delta_min = 0
            ret += [[action, delta_min]]

        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            gpt_response = __func_clean_up(gpt_response, prompt)
            dur_sum = 0
            for act, dur in gpt_response:
                dur_sum += dur
                if str(type(act)) != "<class 'str'>":
                    return False
                if str(type(dur)) != "<class 'int'>":
                    return False
            x = prompt.split("\n")[0].split("originally planned schedule from")[-1].strip()[:-1]
            x = [datetime.datetime.strptime(i.strip(), "%H:%M %p") for i in x.split(" to ")]
            delta_min = int((x[1] - x[0]).total_seconds() / 60)

            if int(dur_sum) != int(delta_min):
                return False

        except:
            return False
        return True

    def get_fail_safe(main_act_dur, truncated_act_dur):
        dur_sum = 0
        for act, dur in main_act_dur: dur_sum += dur

        ret = truncated_act_dur[:]
        ret += main_act_dur[len(ret) - 1:]

        # If there are access, we need to trim...
        ret_dur_sum = 0
        count = 0
        over = None
        for act, dur in ret:
            ret_dur_sum += dur
            if ret_dur_sum == dur_sum:
                break
            if ret_dur_sum > dur_sum:
                over = ret_dur_sum - dur_sum
                break
            count += 1

        if over:
            ret = ret[:count + 1]
            ret[-1][1] -= over

        return ret

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 1000,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/new_decomp_schedule_v1.txt"
    prompt_input = create_prompt_input(persona,
                                       main_act_dur,
                                       truncated_act_dur,
                                       start_time_hour,
                                       end_time_hour,
                                       inserted_act,
                                       inserted_act_dur,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe(main_act_dur, truncated_act_dur)
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    # print ("* * * * output")
    # print (output)
    # print ('* * * * fail_safe')
    # print (fail_safe)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_decide_to_talk(persona, target_persona, retrieved, test_input=None,
                                  verbose=False):
    def create_prompt_input(init_persona, target_persona, retrieved,
                            test_input=None):
        last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
        last_chatted_time = ""
        last_chat_about = ""
        if last_chat:
            last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
            last_chat_about = last_chat.description

        context = ""
        for c_node in retrieved["events"]:
            curr_desc = c_node.description.split(" ")
            curr_desc[2:3] = ["was"]
            curr_desc = " ".join(curr_desc)
            context += f"{curr_desc}. "
        context += "\n"
        for c_node in retrieved["thoughts"]:
            context += f"{c_node.description}. "

        curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
        init_act_desc = init_persona.scratch.act_description
        if "(" in init_act_desc:
            init_act_desc = init_act_desc.split("(")[-1][:-1]

        if len(init_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc:
            init_p_desc = f"{init_persona.name} is already {init_act_desc}"
        elif "waiting" in init_act_desc:
            init_p_desc = f"{init_persona.name} is {init_act_desc}"
        else:
            init_p_desc = f"{init_persona.name} is on the way to {init_act_desc}"

        target_act_desc = target_persona.scratch.act_description
        if "(" in target_act_desc:
            target_act_desc = target_act_desc.split("(")[-1][:-1]

        if len(target_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc:
            target_p_desc = f"{target_persona.name} is already {target_act_desc}"
        elif "waiting" in init_act_desc:
            target_p_desc = f"{init_persona.name} is {init_act_desc}"
        else:
            target_p_desc = f"{target_persona.name} is on the way to {target_act_desc}"

        prompt_input = []
        prompt_input += [context]

        prompt_input += [curr_time]

        prompt_input += [init_persona.name]
        prompt_input += [target_persona.name]
        prompt_input += [last_chatted_time]
        prompt_input += [last_chat_about]

        prompt_input += [init_p_desc]
        prompt_input += [target_p_desc]
        prompt_input += [init_persona.name]
        prompt_input += [target_persona.name]
        return prompt_input

    def _extract_yes_no(s):
        # GPT-4 reliably emits a bare "yes"/"no" after "Answer in yes or no:".
        # Qwen3 emits "Yes.", "Yes, because...", or just "Yes" without the
        # marker. Try: (1) text after marker, (2) first word of cleaned text,
        # (3) any yes/no token anywhere.
        s = _strip_scaffolding(s)
        tail = s.split("Answer in yes or no:")[-1].strip().lower()
        head = re.split(r"[\s,.;:!?]", tail.strip(), maxsplit=1)[0]
        if head in ("yes", "no"):
            return head
        # Fall back to scanning the whole response for the first yes/no token.
        m = re.search(r"\b(yes|no)\b", s.lower())
        if m:
            return m.group(1)
        return None

    def __func_validate(gpt_response, prompt=""):
        try:
            return _extract_yes_no(gpt_response) is not None
        except:
            return False

    def __func_clean_up(gpt_response, prompt=""):
        ans = _extract_yes_no(gpt_response)
        if ans is None:
            raise ValueError(f"decide_to_talk: no yes/no in {gpt_response!r}")
        return ans

    def get_fail_safe():
        fs = "yes"
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 20,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/decide_to_talk_v2.txt"
    prompt_input = create_prompt_input(persona, target_persona, retrieved,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_decide_to_react(persona, target_persona, retrieved, test_input=None,
                                   verbose=False):
    def create_prompt_input(init_persona, target_persona, retrieved,
                            test_input=None):

        context = ""
        for c_node in retrieved["events"]:
            curr_desc = c_node.description.split(" ")
            curr_desc[2:3] = ["was"]
            curr_desc = " ".join(curr_desc)
            context += f"{curr_desc}. "
        context += "\n"
        for c_node in retrieved["thoughts"]:
            context += f"{c_node.description}. "

        curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
        init_act_desc = init_persona.scratch.act_description
        if "(" in init_act_desc:
            init_act_desc = init_act_desc.split("(")[-1][:-1]
        if len(init_persona.scratch.planned_path) == 0:
            loc = ""
            if ":" in init_persona.scratch.act_address:
                loc = init_persona.scratch.act_address.split(":")[-1] + " in " + \
                      init_persona.scratch.act_address.split(":")[-2]
            init_p_desc = f"{init_persona.name} is already {init_act_desc} at {loc}"
        else:
            loc = ""
            if ":" in init_persona.scratch.act_address:
                loc = init_persona.scratch.act_address.split(":")[-1] + " in " + \
                      init_persona.scratch.act_address.split(":")[-2]
            init_p_desc = f"{init_persona.name} is on the way to {init_act_desc} at {loc}"

        target_act_desc = target_persona.scratch.act_description
        if "(" in target_act_desc:
            target_act_desc = target_act_desc.split("(")[-1][:-1]
        if len(target_persona.scratch.planned_path) == 0:
            loc = ""
            if ":" in target_persona.scratch.act_address:
                loc = target_persona.scratch.act_address.split(":")[-1] + " in " + \
                      target_persona.scratch.act_address.split(":")[-2]
            target_p_desc = f"{target_persona.name} is already {target_act_desc} at {loc}"
        else:
            loc = ""
            if ":" in target_persona.scratch.act_address:
                loc = target_persona.scratch.act_address.split(":")[-1] + " in " + \
                      target_persona.scratch.act_address.split(":")[-2]
            target_p_desc = f"{target_persona.name} is on the way to {target_act_desc} at {loc}"

        prompt_input = []
        prompt_input += [context]
        prompt_input += [curr_time]
        prompt_input += [init_p_desc]
        prompt_input += [target_p_desc]

        prompt_input += [init_persona.name]
        prompt_input += [init_act_desc]
        prompt_input += [target_persona.name]
        prompt_input += [target_act_desc]

        prompt_input += [init_act_desc]
        return prompt_input

    def _extract_option(s):
        # GPT-4 emits "Answer: Option N" with N in {1,2,3}. Qwen3 may emit
        # "Option 3", "Answer: 3", "I would choose option 3", or "3".
        s = _strip_scaffolding(s)
        tail = s.split("Answer: Option")[-1].strip().lower()
        head = re.split(r"[\s,.;:!?]", tail.strip(), maxsplit=1)[0]
        if head in ("1", "2", "3"):
            return head
        m = re.search(r"option\s*([123])\b", s.lower())
        if m:
            return m.group(1)
        # Last resort: any standalone 1/2/3 in the response.
        m = re.search(r"\b([123])\b", s)
        if m:
            return m.group(1)
        return None

    def __func_validate(gpt_response, prompt=""):
        try:
            return _extract_option(gpt_response) is not None
        except:
            return False

    def __func_clean_up(gpt_response, prompt=""):
        ans = _extract_option(gpt_response)
        if ans is None:
            raise ValueError(f"decide_to_react: no option in {gpt_response!r}")
        return ans

    def get_fail_safe():
        fs = "3"
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 20,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/decide_to_react_v1.txt"
    prompt_input = create_prompt_input(persona, target_persona, retrieved,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_create_conversation(persona, target_persona, curr_loc,
                                       test_input=None, verbose=False):
    def create_prompt_input(init_persona, target_persona, curr_loc,
                            test_input=None):

        prev_convo_insert = "\n"
        if init_persona.a_mem.seq_chat:
            for i in init_persona.a_mem.seq_chat:
                if i.object == target_persona.scratch.name:
                    v1 = int((init_persona.scratch.curr_time - i.created).total_seconds() / 60)
                    prev_convo_insert += f'{str(v1)} minutes ago, they had the following conversation.\n'
                    for row in i.filling:
                        prev_convo_insert += f'{row[0]}: "{row[1]}"\n'
                    break
        if prev_convo_insert == "\n":
            prev_convo_insert = ""
        if init_persona.a_mem.seq_chat:
            if int((init_persona.scratch.curr_time - init_persona.a_mem.seq_chat[
                -1].created).total_seconds() / 60) > 480:
                prev_convo_insert = ""

        init_persona_thought_nodes = init_persona.a_mem.retrieve_relevant_thoughts(target_persona.scratch.act_event[0],
                                                                                   target_persona.scratch.act_event[1],
                                                                                   target_persona.scratch.act_event[2])
        init_persona_thought = ""
        for i in init_persona_thought_nodes:
            init_persona_thought += f"-- {i.description}\n"

        target_persona_thought_nodes = target_persona.a_mem.retrieve_relevant_thoughts(
            init_persona.scratch.act_event[0],
            init_persona.scratch.act_event[1],
            init_persona.scratch.act_event[2])
        target_persona_thought = ""
        for i in target_persona_thought_nodes:
            target_persona_thought += f"-- {i.description}\n"

        init_persona_curr_desc = ""
        if init_persona.scratch.planned_path:
            init_persona_curr_desc = f"{init_persona.name} is on the way to {init_persona.scratch.act_description}"
        else:
            init_persona_curr_desc = f"{init_persona.name} is {init_persona.scratch.act_description}"

        target_persona_curr_desc = ""
        if target_persona.scratch.planned_path:
            target_persona_curr_desc = f"{target_persona.name} is on the way to {target_persona.scratch.act_description}"
        else:
            target_persona_curr_desc = f"{target_persona.name} is {target_persona.scratch.act_description}"

        curr_loc = curr_loc["arena"]

        prompt_input = []
        prompt_input += [init_persona.scratch.get_str_iss()]
        prompt_input += [target_persona.scratch.get_str_iss()]

        prompt_input += [init_persona.name]
        prompt_input += [target_persona.name]
        prompt_input += [init_persona_thought]

        prompt_input += [target_persona.name]
        prompt_input += [init_persona.name]
        prompt_input += [target_persona_thought]

        prompt_input += [init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S")]

        prompt_input += [init_persona_curr_desc]
        prompt_input += [target_persona_curr_desc]

        prompt_input += [prev_convo_insert]

        prompt_input += [init_persona.name]
        prompt_input += [target_persona.name]

        prompt_input += [curr_loc]
        prompt_input += [init_persona.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # print ("???")
        # print (gpt_response)

        gpt_response = (prompt + gpt_response).split("What would they talk about now?")[-1].strip()
        content = re.findall('"([^"]*)"', gpt_response)

        speaker_order = []
        for i in gpt_response.split("\n"):
            name = i.split(":")[0].strip()
            if name:
                speaker_order += [name]

        ret = []
        for count, speaker in enumerate(speaker_order):
            ret += [[speaker, content[count]]]

        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe(init_persona, target_persona):
        convo = [[init_persona.name, "Hi!"],
                 [target_persona.name, "Hi!"]]
        return convo

    gpt_param = {"engine": "text-davinci-003", "max_tokens": 1000,
                 "temperature": 0.7, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/create_conversation_v2.txt"
    prompt_input = create_prompt_input(persona, target_persona, curr_loc,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe(persona, target_persona)
    # Not Called
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_summarize_conversation(persona, conversation, test_input=None, verbose=False):
    def create_prompt_input(conversation, test_input=None):
        convo_str = ""
        for row in conversation:
            convo_str += f'{row[0]}: "{row[1]}"\n'

        prompt_input = [convo_str]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        ret = "conversing about " + gpt_response.strip()
        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "conversing with a housemate about morning greetings"

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        ret = "conversing about " + gpt_response.strip()
        return ret

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 11")  ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_conversation_v1.txt"  ########
    prompt_input = create_prompt_input(conversation, test_input)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "conversing about what to eat for lunch"  ########
    special_instruction = "The output must continue the sentence above by filling in the <fill in> tag. Don't start with 'this is a conversation about...' Just finish the sentence but do not miss any important details (including who are chatting)."  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    import sys; print("[FAIL_SAFE] run_gpt_prompt_summarize_conversation: ChatGPT path returned False, using fail_safe", file=sys.stderr)
    return fail_safe, [fail_safe, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "", "max_tokens": 50,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/summarize_conversation_v1.txt"
    # prompt_input = create_prompt_input(conversation, test_input)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_extract_keywords(persona, description, test_input=None, verbose=False):
    def create_prompt_input(description, test_input=None):
        if "\n" in description:
            description = description.replace("\n", " <LINE_BREAK> ")
        prompt_input = [description]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print("???")
        print(gpt_response)
        # GPT-4 reliably emits "Factual..., ... Emotive keywords: ..., ...".
        # Qwen3 may omit the "Emotive keywords:" header. If so, treat the
        # whole response as a single comma-separated keyword list.
        s = _strip_scaffolding(gpt_response).strip()
        parts = s.split("Emotive keywords:")
        factual = [i.strip() for i in parts[0].split(",")]
        emotive = []
        if len(parts) > 1:
            emotive = [i.strip() for i in parts[1].split(",")]
        all_keywords = factual + emotive
        ret = []
        for i in all_keywords:
            if i:
                i = i.lower()
                if i[-1] == ".":
                    i = i[:-1]
                ret += [i]
        print(ret)
        return set(ret)

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return []

    gpt_param = {"engine": "text-davinci-003", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/get_keywords_v1.txt"
    prompt_input = create_prompt_input(description, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    # Not Called
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_keyword_to_thoughts(persona, keyword, concept_summary, test_input=None, verbose=False):
    def create_prompt_input(persona, keyword, concept_summary, test_input=None):
        prompt_input = [keyword, concept_summary, persona.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = gpt_response.strip()
        return gpt_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return ""

    gpt_param = {"engine": "text-davinci-003", "max_tokens": 40,
                 "temperature": 0.7, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/keyword_to_thoughts_v1.txt"
    prompt_input = create_prompt_input(persona, keyword, concept_summary)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    # Not Called
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_convo_to_thoughts(persona,
                                     init_persona_name,
                                     target_persona_name,
                                     convo_str,
                                     fin_target, test_input=None, verbose=False):
    def create_prompt_input(init_persona_name,
                            target_persona_name,
                            convo_str,
                            fin_target, test_input=None):
        prompt_input = [init_persona_name,
                        target_persona_name,
                        convo_str,
                        init_persona_name,
                        fin_target]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = gpt_response.strip()
        return gpt_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return ""

    gpt_param = {"engine": "text-davinci-003", "max_tokens": 40,
                 "temperature": 0.7, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/convo_to_thoughts_v1.txt"
    prompt_input = create_prompt_input(init_persona_name,
                                       target_persona_name,
                                       convo_str,
                                       fin_target)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    # Not Called
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_event_poignancy(persona, event_description, test_input=None, verbose=False):
    def create_prompt_input(persona, event_description, test_input=None):
        prompt_input = [persona.scratch.name,
                        persona.scratch.get_str_iss(),
                        persona.scratch.name,
                        event_description]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # GPT-4 returns a bare integer 1..10. Qwen3 may return "Score: 5",
        # "5/10", "I'd rate it a 5.", or markdown-wrapped. Strip scaffolding
        # and pull the first integer.
        s = _strip_scaffolding(gpt_response)
        try:
            return int(s.strip())
        except Exception:
            pass
        n = _extract_first_int(s)
        if n is None:
            raise ValueError(f"poignancy: no integer in {gpt_response!r}")
        # Clamp to 1..10 to keep downstream math sane.
        if n < 1:
            n = 1
        if n > 10:
            n = 10
        return n

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return 4

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        # Same Qwen3 hardening as __func_clean_up — this is the path actually
        # used by ChatGPT_safe_generate_response below.
        s = _strip_scaffolding(gpt_response)
        try:
            n = int(str(s).strip())
        except Exception:
            n = _extract_first_int(str(s))
        if n is None:
            raise ValueError(f"poignancy(chat): no integer in {gpt_response!r}")
        if n < 1:
            n = 1
        if n > 10:
            n = 10
        return n

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 7")  ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_event_v1.txt"  ########
    prompt_input = create_prompt_input(persona, event_description)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "5"  ########
    special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "", "max_tokens": 3,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/poignancy_event_v1.txt"
    # prompt_input = create_prompt_input(persona, event_description)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_thought_poignancy(persona, event_description, test_input=None, verbose=False):
    def create_prompt_input(persona, event_description, test_input=None):
        prompt_input = [persona.scratch.name,
                        persona.scratch.get_str_iss(),
                        persona.scratch.name,
                        event_description]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # GPT-4 returns a bare integer 1..10. Qwen3 may return "Score: 5",
        # "5/10", "I'd rate it a 5.", or markdown-wrapped. Strip scaffolding
        # and pull the first integer.
        s = _strip_scaffolding(gpt_response)
        try:
            return int(s.strip())
        except Exception:
            pass
        n = _extract_first_int(s)
        if n is None:
            raise ValueError(f"poignancy: no integer in {gpt_response!r}")
        # Clamp to 1..10 to keep downstream math sane.
        if n < 1:
            n = 1
        if n > 10:
            n = 10
        return n

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return 4

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        # Same Qwen3 hardening as __func_clean_up — this is the path actually
        # used by ChatGPT_safe_generate_response below.
        s = _strip_scaffolding(gpt_response)
        try:
            n = int(str(s).strip())
        except Exception:
            n = _extract_first_int(str(s))
        if n is None:
            raise ValueError(f"poignancy(chat): no integer in {gpt_response!r}")
        if n < 1:
            n = 1
        if n > 10:
            n = 10
        return n

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 8")  ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_thought_v1.txt"  ########
    prompt_input = create_prompt_input(persona, event_description)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "5"  ########
    special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "", "max_tokens": 3,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/poignancy_thought_v1.txt"
    # prompt_input = create_prompt_input(persona, event_description)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_chat_poignancy(persona, event_description, test_input=None, verbose=False):
    def create_prompt_input(persona, event_description, test_input=None):
        prompt_input = [persona.scratch.name,
                        persona.scratch.get_str_iss(),
                        persona.scratch.name,
                        event_description]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # GPT-4 returns a bare integer 1..10. Qwen3 may return "Score: 5",
        # "5/10", "I'd rate it a 5.", or markdown-wrapped. Strip scaffolding
        # and pull the first integer.
        s = _strip_scaffolding(gpt_response)
        try:
            return int(s.strip())
        except Exception:
            pass
        n = _extract_first_int(s)
        if n is None:
            raise ValueError(f"poignancy: no integer in {gpt_response!r}")
        # Clamp to 1..10 to keep downstream math sane.
        if n < 1:
            n = 1
        if n > 10:
            n = 10
        return n

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return 4

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        # Same Qwen3 hardening as __func_clean_up — this is the path actually
        # used by ChatGPT_safe_generate_response below.
        s = _strip_scaffolding(gpt_response)
        try:
            n = int(str(s).strip())
        except Exception:
            n = _extract_first_int(str(s))
        if n is None:
            raise ValueError(f"poignancy(chat): no integer in {gpt_response!r}")
        if n < 1:
            n = 1
        if n > 10:
            n = 10
        return n

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 9")  ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_chat_v1.txt"  ########
    prompt_input = create_prompt_input(persona, event_description)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "5"  ########
    special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "", "max_tokens": 3,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/poignancy_chat_v1.txt"
    # prompt_input = create_prompt_input(persona, event_description)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_focal_pt(persona, statements, n, test_input=None, verbose=False):
    def create_prompt_input(persona, statements, n, test_input=None):
        prompt_input = [statements, str(n)]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # GPT-4 emits "1) Q1\n2) Q2\n...". Qwen3 may emit "1. Q1", "- Q1", or
        # bare lines. Strip scaffolding, then accept any numbered/bulleted
        # form.
        s = _strip_scaffolding(gpt_response).strip()
        ret = []
        for line in s.split("\n"):
            line = line.strip()
            if not line:
                continue
            # remove "1)", "1.", "-", "*", "•" leading markers
            line = re.sub(r"^[\-\*•]\s*", "", line)
            line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if line:
                ret.append(line)
        if not ret:
            raise ValueError("focal_pt: empty parse")
        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe(n):
        return ["Who am I"] * n

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        ret = ast.literal_eval(gpt_response)
        return ret

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 12")  ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt"  ########
    prompt_input = create_prompt_input(persona, statements, n)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = '["What should Jane do for lunch", "Does Jane like strawberry", "Who is Jane"]'  ########
    special_instruction = "Output must be a list of str."  ########
    fail_safe = get_fail_safe(n)  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/generate_focal_pt_v1.txt"
    prompt_input = create_prompt_input(persona, statements, n)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe(n)
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_insight_and_guidance(persona, statements, n, test_input=None, verbose=False):
    def create_prompt_input(persona, statements, n, test_input=None):
        prompt_input = [statements, str(n)]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)
        gpt_response = _strip_scaffolding(gpt_response).strip()
        if not gpt_response:
            raise ValueError("insight_and_guidance: empty response")
        if gpt_response.split("\n")[0][:1] != '1':
            gpt_response = "1. " + gpt_response
        ret = dict()
        for i in gpt_response.split("\n"):
            i = i.strip()
            if not i:
                continue
            # GPT-4: "1. Thought (because of 1, 2)". Qwen3 may emit just
            # "1. Thought" or "- Thought". Tolerate missing evidence tail.
            parts = i.split(". ", 1)
            if len(parts) < 2:
                # try alternate bullet styles
                stripped = re.sub(r"^[\-\*•]\s*", "", i).strip()
                stripped = re.sub(r"^\d+[\.\)]\s*", "", stripped).strip()
                if not stripped:
                    continue
                thought_and_evi = stripped
            else:
                thought_and_evi = parts[1]
            if "(because of " in thought_and_evi:
                thought = thought_and_evi.split("(because of ")[0].strip()
                evi_raw = thought_and_evi.split("(because of ")[1].split(")")[0].strip()
                evi_raw = re.findall(r'\d+', evi_raw)
                evi_raw = [int(j.strip()) for j in evi_raw]
            else:
                # No evidence tail — accept thought with empty evidence list.
                thought = thought_and_evi.strip()
                evi_raw = []
            if thought:
                ret[thought] = evi_raw
        if not ret:
            raise ValueError("insight_and_guidance: no parseable lines")
        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe(n):
        return ["I am hungry"] * n

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 1500,
                 "temperature": 0.5, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/insight_and_evidence_v1.txt"
    prompt_template = "norm/other_prompt/insight_and_evidence_v2.txt"
    prompt_input = create_prompt_input(persona, statements, n)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe(n)
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    # output = GPT4_safe_generate_response_OLD(prompt, 3, fail_safe,
    #                                         __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_agent_chat_summarize_ideas(persona, target_persona, statements, curr_context, test_input=None,
                                              verbose=False):
    def create_prompt_input(persona, target_persona, statements, curr_context, test_input=None):
        prompt_input = [persona.scratch.get_str_curr_date_str(), curr_context, persona.scratch.currently,
                        statements, persona.scratch.name, target_persona.scratch.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "..."

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        return gpt_response.split('"')[0].strip()

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 17")  ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_ideas_v1.txt"  ########
    prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'Jane Doe is working on a project'  ########
    special_instruction = 'The output should be a string that responds to the question.'  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "", "max_tokens": 150,
    #              "temperature": 0.5, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/summarize_chat_ideas_v1.txt"
    # prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_agent_chat_summarize_relationship(persona, target_persona, statements, test_input=None,
                                                     verbose=False):
    def create_prompt_input(persona, target_persona, statements, test_input=None):
        prompt_input = [statements, persona.scratch.name, target_persona.scratch.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "..."

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        return gpt_response.split('"')[0].strip()

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 18")  ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_relationship_v2.txt"  ########
    prompt_input = create_prompt_input(persona, target_persona, statements)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'Jane Doe is working on a project'  ########
    special_instruction = 'The output should be a string that responds to the question.'  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "", "max_tokens": 150,
    #              "temperature": 0.5, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/summarize_chat_relationship_v1.txt"
    # prompt_input = create_prompt_input(persona, target_persona, statements)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_agent_chat(maze, persona, target_persona,
                              curr_context,
                              init_summ_idea,
                              target_summ_idea, test_input=None, verbose=False):
    def create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea, test_input=None):
        prev_convo_insert = "\n"
        if persona.a_mem.seq_chat:
            for i in persona.a_mem.seq_chat:
                if i.object == target_persona.scratch.name:
                    v1 = int((persona.scratch.curr_time - i.created).total_seconds() / 60)
                    prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
                    break
        if prev_convo_insert == "\n":
            prev_convo_insert = ""
        if persona.a_mem.seq_chat:
            if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds() / 60) > 480:
                prev_convo_insert = ""
        print(prev_convo_insert)

        curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        curr_arena = f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
        curr_location = f"{curr_arena} in {curr_sector}"

        prompt_input = [persona.scratch.currently,
                        target_persona.scratch.currently,
                        prev_convo_insert,
                        curr_context,
                        curr_location,

                        persona.scratch.name,
                        init_summ_idea,
                        persona.scratch.name,
                        target_persona.scratch.name,

                        target_persona.scratch.name,
                        target_summ_idea,
                        target_persona.scratch.name,
                        persona.scratch.name,

                        persona.scratch.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)

        gpt_response = (prompt + gpt_response).split("Here is their conversation.")[-1].strip()
        content = re.findall('"([^"]*)"', gpt_response)

        speaker_order = []
        for i in gpt_response.split("\n"):
            name = i.split(":")[0].strip()
            if name:
                speaker_order += [name]

        ret = []
        for count, speaker in enumerate(speaker_order):
            ret += [[speaker, content[count]]]

        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "..."

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        # ret = ast.literal_eval(gpt_response)

        print("a;dnfdap98fh4p9enf HEREE!!!")
        for row in gpt_response:
            print(row)

        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):  ############
        return True

    # print ("HERE JULY 23 -- ----- ") ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/agent_chat_v1.txt"  ########
    prompt_input = create_prompt_input(persona, target_persona, curr_context, init_summ_idea,
                                       target_summ_idea)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = '[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]'  ########
    special_instruction = 'The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"].'  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    # print ("HERE END JULY 23 -- ----- ") ########
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "", "max_tokens": 2000,
    #              "temperature": 0.7, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/agent_chat_v1.txt"
    # prompt_input = create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# =======================
# =======================
# =======================
# =======================


def run_gpt_prompt_summarize_ideas(persona, statements, question, test_input=None, verbose=False):
    def create_prompt_input(persona, statements, question, test_input=None):
        curr_act_norms = "\n"
        for norm_id, a_norm in persona.norm_database.act_norm.items():
            if a_norm.activation_state == False:
                continue
            curr_act_norms += f"- [{str(a_norm.poignancy)}] "
            curr_act_norms += a_norm.content
            curr_act_norms += "\n"
        if curr_act_norms == "\n":
            curr_act_norms = "There is no norm.\n"
        prompt_input = [statements, persona.scratch.name, question, curr_act_norms]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        print(gpt_response)
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "..."

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        return gpt_response.split('"')[0].strip()

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 16")  ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 1500,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    #prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_ideas_v1.txt"  ########
    prompt_template = "norm/norm_interview_prompt/summarize_ideas_v2.txt"
    prompt_input = create_prompt_input(persona, statements, question)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'Jane Doe is working on a project'  ########
    special_instruction = 'The output should be a string that responds to the question.'  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                         prompt_input, prompt, output)
    
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "", "max_tokens": 150,
    #              "temperature": 0.5, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/summarize_ideas_v1.txt"
    # prompt_input = create_prompt_input(persona, statements, question)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_generate_next_convo_line(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None,
                                            verbose=False):
    def create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None):
        curr_act_norms = "\n"
        for norm_id, a_norm in persona.norm_database.act_norm.items():
            if a_norm.activation_state == False:
                continue
            curr_act_norms += f"- [{str(a_norm.poignancy)}] "
            curr_act_norms += a_norm.content
            curr_act_norms += "\n"
        if curr_act_norms == "\n":
            curr_act_norms = "There is no norm.\n"
        prompt_input = [persona.scratch.name,
                        persona.scratch.get_str_iss(),
                        persona.scratch.name,
                        interlocutor_desc,
                        prev_convo,
                        persona.scratch.name,
                        retrieved_summary,
                        persona.scratch.name, 
                        curr_act_norms,]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "..."

    # # ChatGPT Plugin ===========================================================
    # def __chat_func_clean_up(gpt_response, prompt=""): ############
    #   return gpt_response.split('"')[0].strip()

    # def __chat_func_validate(gpt_response, prompt=""): ############
    #   try:
    #     __func_clean_up(gpt_response, prompt)
    #     return True
    #   except:
    #     return False

    # print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 15") ########
    # gpt_param = {"engine": "", "max_tokens": 15,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v3_ChatGPT/generate_next_convo_line_v1.txt" ########
    # prompt_input = create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary)  ########
    # prompt = generate_prompt(prompt_input, prompt_template)
    # example_output = 'Hello' ########
    # special_instruction = 'The output should be a string that responds to the question. Again, only use the context included in the "Note" to generate the response' ########
    # fail_safe = get_fail_safe() ########
    # output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
    #                                         __chat_func_validate, __chat_func_clean_up, True)
    # if output != False:
    #   return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 250,
                 "temperature": 1, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    #prompt_template = "persona/prompt_template/v2/generate_next_convo_line_v1.txt"
    prompt_template = "norm/norm_interview_prompt/generate_next_convo_line_v2.txt"
    prompt_input = create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    # Not Called
    #output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                __func_validate, __func_clean_up)
    output = GPT4_safe_generate_response_OLD_t1(prompt, 3, fail_safe,
                                                __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_generate_whisper_inner_thought(persona, whisper, test_input=None, verbose=False):
    def create_prompt_input(persona, whisper, test_input=None):
        prompt_input = [persona.scratch.name, whisper]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "..."

    gpt_param = {"engine": "text-davinci-003", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/whisper_inner_thought_v1.txt"
    prompt_input = create_prompt_input(persona, whisper)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    # Not Called
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_planning_thought_on_convo(persona, all_utt, test_input=None, verbose=False):
    def create_prompt_input(persona, all_utt, test_input=None):
        prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "..."

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/planning_thought_on_convo_v1.txt"
    prompt_input = create_prompt_input(persona, all_utt)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_memo_on_convo(persona, all_utt, test_input=None, verbose=False):
    def create_prompt_input(persona, all_utt, test_input=None):
        prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "..."

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        return gpt_response.strip()

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 15")  ########
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/memo_on_convo_v1.txt"  ########
    prompt_input = create_prompt_input(persona, all_utt)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'Jane Doe was interesting to talk to.'  ########
    special_instruction = 'The output should ONLY contain a string that summarizes anything interesting that the agent may have noticed'  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "text-davinci-003", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/memo_on_convo_v1.txt"
    prompt_input = create_prompt_input(persona, all_utt)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    # Not Called
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_generate_safety_score(persona, comment, test_input=None, verbose=False):
    def create_prompt_input(comment, test_input=None):
        prompt_input = [comment]
        return prompt_input

    def __chat_func_clean_up(gpt_response, prompt=""):
        gpt_response = json.loads(gpt_response)
        return gpt_response["output"]

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            fields = ["output"]
            response = json.loads(gpt_response)
            for field in fields:
                if field not in response:
                    return False
            return True
        except:
            return False

    def get_fail_safe():
        return None

    print("11")
    prompt_template = "persona/prompt_template/safety/anthromorphosization_v1.txt"
    prompt_input = create_prompt_input(comment)
    print("22")
    prompt = generate_prompt(prompt_input, prompt_template)
    print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __chat_func_validate, __chat_func_clean_up, verbose)
    print(output)

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def extract_first_json_dict(data_str):
    # Find the first occurrence of a JSON object within the string
    start_idx = data_str.find('{')
    end_idx = data_str.find('}', start_idx) + 1

    # Check if both start and end indices were found
    if start_idx == -1 or end_idx == 0:
        return None

    # Extract the first JSON dictionary
    json_str = data_str[start_idx:end_idx]

    try:
        # Attempt to parse the JSON data
        json_dict = json.loads(json_str)
        return json_dict
    except json.JSONDecodeError:
        # If parsing fails, return None
        return None


def run_gpt_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat,
                                        norm_conf_res, test_input=None, verbose=False):
    def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, norm_conf_res,
                            test_input=None):
        persona = init_persona
        prev_convo_insert = "\n"
        if persona.a_mem.seq_chat:
            for i in persona.a_mem.seq_chat:
                if i.object == target_persona.scratch.name:
                    v1 = int((persona.scratch.curr_time - i.created).total_seconds() / 60)
                    prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
                    break
        if prev_convo_insert == "\n":
            prev_convo_insert = ""
        if persona.a_mem.seq_chat:
            if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds() / 60) > 480:
                prev_convo_insert = ""
        print(prev_convo_insert)

        curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        curr_arena = f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
        curr_location = f"{curr_arena} in {curr_sector}"

        retrieved_str = ""
        for key, vals in retrieved.items():
            for v in vals:
                retrieved_str += f"- {v.description}\n"

        convo_str = ""
        for i in curr_chat:
            convo_str += ": ".join(i) + "\n"
        if convo_str == "":
            convo_str = "[The conversation has not started yet -- start it!]"

        init_iss = f"Here is Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"
        prompt_input = [init_iss, init_persona.scratch.name, retrieved_str, prev_convo_insert,
                        curr_location, curr_context, init_persona.scratch.name, target_persona.scratch.name,
                        convo_str, init_persona.scratch.name, target_persona.scratch.name,
                        init_persona.scratch.name, init_persona.scratch.name,
                        init_persona.scratch.name, norm_conf_res
                        ]
        return prompt_input

    def __chat_func_clean_up(gpt_response, prompt=""):
        gpt_response = extract_first_json_dict(gpt_response)

        cleaned_dict = dict()
        cleaned = []
        for key, val in gpt_response.items():
            cleaned += [val]
        cleaned_dict["utterance"] = cleaned[0]
        cleaned_dict["end"] = True
        if "f" in str(cleaned[1]) or "F" in str(cleaned[1]):
            cleaned_dict["end"] = False

        return cleaned_dict

    def __chat_func_validate(gpt_response, prompt=""):
        print("ugh...")
        try:
            # print ("debug 1")
            # print (gpt_response)
            # print ("debug 2")

            print(extract_first_json_dict(gpt_response))
            # print ("debug 3")

            return True
        except:
            return False

    def get_fail_safe():
        cleaned_dict = dict()
        cleaned_dict["utterance"] = "..."
        cleaned_dict["end"] = False
        return cleaned_dict

    print("11")
    # prompt_template = "persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt"
    prompt_template = "norm/norm_retrieve_prompt/iterative_convo_ours_v2.txt"
    prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat,
                                       norm_conf_res)
    print("22")
    prompt = generate_prompt(prompt_input, prompt_template)
    print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __chat_func_validate, __chat_func_clean_up, verbose)
    print(output)

    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
