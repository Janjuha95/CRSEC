"""
Tests for Qwen3 leniency in cleanup/parser code ported from GPT-4 baseline.

These tests cover two layers:

1. Module-level helpers in gpt_structure (_strip_scaffolding, _extract_first_int,
   _extract_yes_no) — these carry the bulk of the Qwen3 → GPT-4 format
   bridging. The nested __func_clean_up functions all delegate to them.

2. A handful of end-to-end nested cleanups (task_decomp, action_sector,
   action_arena, decide_to_talk, poignancy) where we monkey-patch the
   safe_response wrappers to capture the cleanup, then call the cleanup
   directly with both GPT-4-style and Qwen3-style sample outputs.

Run from reverie/backend_server/:
    python -m pytest tests/test_cleanups.py -v
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

_ORIGINAL_CWD = None


def setUpModule():
    # Prompt templates are loaded via paths relative to backend_server cwd.
    global _ORIGINAL_CWD
    _ORIGINAL_CWD = os.getcwd()
    os.chdir(BACKEND)


def tearDownModule():
    if _ORIGINAL_CWD is not None:
        os.chdir(_ORIGINAL_CWD)


# Importing modules in correct order is fiddly because gpt_structure imports
# llm_router which imports ollama at module load. We patch ollama before
# importing so unit tests don't require Ollama to be running.
sys.modules.setdefault("ollama", MagicMock())

from persona.prompt_template import gpt_structure  # noqa: E402
from persona.prompt_template.gpt_structure import (  # noqa: E402
    _strip_scaffolding,
    _extract_first_int,
    _extract_yes_no,
)


class TestStripScaffolding(unittest.TestCase):
    """_strip_scaffolding is lossless on GPT-4 output and strips Qwen3 noise."""

    def test_gpt4_bare_response_unchanged(self):
        # Class 4 invariant: clean GPT-4 outputs must pass through.
        for s in ["7", "kitchen", "yes", "5", "Tom Watson is eating breakfast"]:
            self.assertEqual(_strip_scaffolding(s), s)

    def test_strips_answer_prefix(self):
        self.assertEqual(_strip_scaffolding("Answer: kitchen"), "kitchen")
        self.assertEqual(_strip_scaffolding("Answer : 7"), "7")
        self.assertEqual(_strip_scaffolding("Output: bedroom"), "bedroom")

    def test_strips_answer_with_open_brace(self):
        # The concrete Class-4 example from the porting brief.
        self.assertEqual(_strip_scaffolding("Answer: {cafe"), "cafe")
        self.assertEqual(_strip_scaffolding("Answer: {kitchen}"), "{kitchen}")

    def test_strips_code_fences(self):
        self.assertEqual(_strip_scaffolding("```json\n{\"a\": 1}\n```"), '{"a": 1}')
        self.assertEqual(_strip_scaffolding("```\nhello\n```"), "hello")

    def test_strips_persona_name_leakage(self):
        self.assertEqual(
            _strip_scaffolding("Maeve is sleeping", persona_name="Maeve"),
            "sleeping",
        )

    def test_non_string_passthrough(self):
        self.assertEqual(_strip_scaffolding(None), None)
        self.assertEqual(_strip_scaffolding(7), 7)


class TestExtractFirstInt(unittest.TestCase):
    def test_bare_int(self):
        self.assertEqual(_extract_first_int("7"), 7)

    def test_with_prefix(self):
        # Qwen3 patterns: "Score: 7", "I'd rate it a 5", "5/10".
        self.assertEqual(_extract_first_int("Score: 7"), 7)
        self.assertEqual(_extract_first_int("I'd rate it a 5"), 5)
        self.assertEqual(_extract_first_int("5/10"), 5)

    def test_none_when_no_digits(self):
        self.assertIsNone(_extract_first_int("no digits here"))
        self.assertIsNone(_extract_first_int(""))


class TestExtractYesNo(unittest.TestCase):
    def test_bare_yes_no(self):
        self.assertEqual(_extract_yes_no("yes"), "yes")
        self.assertEqual(_extract_yes_no("no"), "no")

    def test_qwen3_variants(self):
        # Qwen3 tends to add punctuation or justification.
        self.assertEqual(_extract_yes_no("Yes."), "yes")
        self.assertEqual(_extract_yes_no("Yes, because she is friendly"), "yes")
        self.assertEqual(_extract_yes_no("No - the norm forbids it"), "no")
        self.assertEqual(_extract_yes_no("Answer: Yes"), "yes")

    def test_none_when_absent(self):
        self.assertIsNone(_extract_yes_no("maybe later"))


class TestCreationSliceJson(unittest.TestCase):
    """norm/creation.py: _slice_json must handle GPT-4 raw JSON and Qwen3 fences."""

    def test_raw_json_unchanged(self):
        from norm.creation import _slice_json
        s = '{"norm_1": {"id": "1"}}'
        self.assertEqual(_slice_json(s), s)

    def test_strips_code_fence(self):
        from norm.creation import _slice_json
        out = _slice_json('```json\n{"a": 1}\n```')
        self.assertIn('"a"', out)
        self.assertNotIn("```", out)

    def test_strips_prelude(self):
        from norm.creation import _slice_json
        out = _slice_json('Here are the norms:\n{"a": 1}\nDone.')
        self.assertEqual(out, '{"a": 1}')


# ---------------------------------------------------------------------------
# End-to-end nested cleanup tests
# ---------------------------------------------------------------------------
#
# The nested __func_clean_up closures aren't importable directly, so we
# monkey-patch the safe_response wrappers used inside each run_gpt_prompt_*
# function to capture the cleanup callable. Then we invoke it ourselves with
# both GPT-4-style and Qwen3-style sample inputs.

class _CleanupCaptor:
    """Captures the func_clean_up + fail_safe passed to a safe_response call."""

    def __init__(self):
        self.cleanup = None
        self.validate = None
        self.fail_safe = None

    def make_safe(self, signature):
        """Return a function with the signature shape of a safe_response wrapper."""
        captor = self

        if signature == "safe_generate_response":
            # (prompt, gpt_param, repeat, fail_safe, validate, cleanup, verbose=False)
            def fake(prompt, gpt_param, repeat, fail_safe, validate, cleanup, verbose=False):
                captor.fail_safe = fail_safe
                captor.validate = validate
                captor.cleanup = cleanup
                return fail_safe
            return fake

        if signature == "GPT4_safe_generate_response_OLD":
            # (prompt, repeat, fail_safe, validate, cleanup, verbose=False)
            def fake(prompt, repeat, fail_safe, validate, cleanup, verbose=False):
                captor.fail_safe = fail_safe
                captor.validate = validate
                captor.cleanup = cleanup
                return fail_safe
            return fake

        if signature == "ChatGPT_safe_generate_response":
            # (prompt, example_output, special_instruction, repeat, fail_safe, validate, cleanup, verbose=False)
            def fake(prompt, example_output, special_instruction, repeat, fail_safe,
                     validate, cleanup, verbose=False):
                captor.fail_safe = fail_safe
                captor.validate = validate
                captor.cleanup = cleanup
                return fail_safe
            return fake

        raise ValueError(f"unknown signature: {signature}")


def _stub_persona(first_name="Maeve", full_name="Maeve Jenson"):
    """Minimal persona stub with the attributes touched by the cleanups."""
    p = MagicMock()
    p.name = full_name
    p.scratch = MagicMock()
    p.scratch.get_str_firstname.return_value = first_name
    p.scratch.get_str_name.return_value = full_name
    p.scratch.get_str_iss.return_value = ""
    p.scratch.get_str_lifestyle.return_value = ""
    p.scratch.get_str_curr_date_str.return_value = "Saturday May 10"
    p.scratch.get_str_daily_plan_req.return_value = ""
    p.scratch.living_area = "world:home"
    p.scratch.last_name = "Jenson"
    p.scratch.f_daily_schedule_hourly_org = [["sleeping", 60]]
    p.scratch.curr_time = MagicMock()
    p.scratch.curr_time.strftime = MagicMock(return_value="May 10, 2024")
    p.scratch.get_f_daily_schedule_hourly_org_index = MagicMock(return_value=0)
    p.s_mem = MagicMock()
    p.s_mem.get_str_accessible_sectors.return_value = "home, cafe"
    p.s_mem.get_str_accessible_sector_arenas.return_value = "kitchen, bedroom"
    return p


def _capture_cleanup(run_gpt_fn, signature, *args, **kwargs):
    """Call run_gpt_fn with the safe wrapper patched out; return the captured cleanup."""
    captor = _CleanupCaptor()
    fake = captor.make_safe(signature)
    with patch.object(gpt_structure, signature, fake), \
         patch(f"persona.prompt_template.run_gpt_prompt.{signature}", fake, create=False), \
         patch("persona.prompt_template.run_gpt_prompt.print_run_prompts", lambda *a, **k: None), \
         patch("persona.prompt_template.run_gpt_prompt.generate_prompt", return_value="dummy prompt"):
        try:
            run_gpt_fn(*args, **kwargs)
        except Exception:
            # The orchestration after the safe call may crash on our fail_safe
            # value; we don't care — we only need the cleanup captor populated.
            pass
    return captor


class TestActionSectorCleanup(unittest.TestCase):
    """action_sector cleanup must handle GPT-4 '{kitchen}' AND Qwen3 'Answer: {cafe'."""

    def setUp(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_action_sector
        self.fn = run_gpt_prompt_action_sector
        self.persona = _stub_persona()
        self.maze = MagicMock()
        tile = {"world": "world", "sector": "home", "arena": "kitchen"}
        self.maze.access_tile.return_value = tile
        self.persona.scratch.curr_tile = (0, 0)
        captor = _capture_cleanup(self.fn, "safe_generate_response",
                                   "go to cafe", self.persona, self.maze)
        self.cleanup = captor.cleanup
        self.assertIsNotNone(self.cleanup, "cleanup must be captured")

    def test_gpt4_braced_form(self):
        self.assertEqual(self.cleanup("kitchen}", prompt=""), "kitchen")

    def test_qwen3_unclosed_brace(self):
        # The concrete Class-4 example from the brief.
        self.assertEqual(self.cleanup("Answer: {cafe", prompt=""), "cafe")

    def test_qwen3_bare(self):
        self.assertEqual(self.cleanup("cafe", prompt=""), "cafe")

    def test_qwen3_with_trailing_punct(self):
        self.assertEqual(self.cleanup("cafe.", prompt=""), "cafe")


class TestActionArenaCleanup(unittest.TestCase):
    def setUp(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_action_arena
        self.fn = run_gpt_prompt_action_arena
        self.persona = _stub_persona()
        self.maze = MagicMock()
        captor = _capture_cleanup(self.fn, "safe_generate_response",
                                   "go to cafe", self.persona, self.maze,
                                   "world", "Hobbs Cafe")
        self.cleanup = captor.cleanup
        self.assertIsNotNone(self.cleanup)

    def test_gpt4_braced(self):
        self.assertEqual(self.cleanup("cafe}", prompt=""), "cafe")

    def test_qwen3_unclosed(self):
        # Same concrete leakage pattern that produced "cafe" + extra colons
        # downstream in the smoke test.
        self.assertEqual(self.cleanup("Answer: {cafe", prompt=""), "cafe")


class TestDecideToTalkCleanup(unittest.TestCase):
    def setUp(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_decide_to_talk
        self.fn = run_gpt_prompt_decide_to_talk
        self.persona = _stub_persona()
        self.target = _stub_persona("Klaus", "Klaus Mueller")
        self.target.scratch.act_event = ("Klaus", "is", "working")
        self.persona.scratch.act_event = ("Maeve", "is", "reading")
        self.persona.a_mem = MagicMock()
        self.persona.a_mem.get_last_chat.return_value = None
        captor = _capture_cleanup(self.fn, "safe_generate_response",
                                   self.persona, self.target,
                                   {"events": [], "thoughts": []})
        self.cleanup = captor.cleanup
        self.assertIsNotNone(self.cleanup)

    def test_gpt4_strict(self):
        self.assertEqual(self.cleanup("Answer in yes or no: yes", prompt=""), "yes")

    def test_qwen3_with_punct(self):
        self.assertEqual(self.cleanup("Yes.", prompt=""), "yes")

    def test_qwen3_with_justification(self):
        self.assertEqual(
            self.cleanup("Yes, because Maeve and Klaus are friends", prompt=""),
            "yes",
        )

    def test_qwen3_bare_no(self):
        self.assertEqual(self.cleanup("No - they don't know each other", prompt=""), "no")


class TestEventPoignancyCleanup(unittest.TestCase):
    def setUp(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_event_poignancy
        self.fn = run_gpt_prompt_event_poignancy
        self.persona = _stub_persona()
        # ChatGPT_safe_generate_response is used for poignancy.
        captor = _capture_cleanup(self.fn, "ChatGPT_safe_generate_response",
                                   self.persona, "a fire alarm goes off")
        self.cleanup = captor.cleanup
        self.assertIsNotNone(self.cleanup)

    def test_gpt4_bare_int(self):
        self.assertEqual(self.cleanup("7", prompt=""), 7)

    def test_qwen3_with_label(self):
        self.assertEqual(self.cleanup("Score: 5", prompt=""), 5)

    def test_qwen3_with_slash(self):
        self.assertEqual(self.cleanup("5/10", prompt=""), 5)

    def test_qwen3_with_text(self):
        self.assertEqual(self.cleanup("I'd rate this a 8.", prompt=""), 8)

    def test_clamps_high(self):
        # Defensive: model returns 11 → clamp to 10.
        self.assertEqual(self.cleanup("11", prompt=""), 10)


class TestWakeUpHourCleanup(unittest.TestCase):
    def setUp(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_wake_up_hour
        self.fn = run_gpt_prompt_wake_up_hour
        self.persona = _stub_persona()
        captor = _capture_cleanup(self.fn, "safe_generate_response", self.persona)
        self.cleanup = captor.cleanup
        self.assertIsNotNone(self.cleanup)

    def test_gpt4_strict(self):
        # GPT-4 emits e.g. "7 am"
        self.assertEqual(self.cleanup("7am", prompt=""), 7)
        self.assertEqual(self.cleanup("7 AM", prompt=""), 7)

    def test_qwen3_bare(self):
        self.assertEqual(self.cleanup("7", prompt=""), 7)

    def test_qwen3_with_prefix(self):
        self.assertEqual(self.cleanup("Answer: 7", prompt=""), 7)
        self.assertEqual(self.cleanup("I'd say 7 in the morning", prompt=""), 7)


class TestTaskDecompCleanup(unittest.TestCase):
    """task_decomp cleanup must handle GPT-4 duration tails AND Qwen3 bare lines.

    GPT-4 emits each line with '(duration in minutes: N, minutes left: M)'.
    Qwen3 may omit the tail; the cleanup must evenly split the total instead.
    """

    def setUp(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_task_decomp_v2
        self.fn = run_gpt_prompt_task_decomp_v2
        self.persona = _stub_persona("Maeve")
        captor = _capture_cleanup(self.fn, "GPT4_safe_generate_response_OLD",
                                   self.persona, "having lunch", 30, "no active norms")
        self.cleanup = captor.cleanup
        self.assertIsNotNone(self.cleanup)

    def _prompt_with_total(self, total):
        # The cleanup parses "(total duration in minutes <N>)" out of prompt.
        return f"Maeve does (total duration in minutes {total}):"

    def test_gpt4_format(self):
        text = (
            "Maeve is sitting down (duration in minutes: 5, minutes left: 25)\n"
            "Maeve is ordering lunch (duration in minutes: 10, minutes left: 15)\n"
            "Maeve is eating (duration in minutes: 15, minutes left: 0)"
        )
        # Use 1. prefix wrapper expected by cleanup.
        wrapped = "1. Maeve is " + text
        out = self.cleanup(wrapped, prompt=self._prompt_with_total(30))
        # Should produce a list of [task, duration] entries.
        self.assertIsInstance(out, list)
        self.assertTrue(len(out) >= 1)
        # Total durations should land near 30 (cleanup does its own slot math
        # which snaps to 5-min increments and may pad/trim).
        total = sum(d for _, d in out)
        self.assertEqual(total, 30)

    def test_qwen3_no_durations_evenly_splits(self):
        # Bare numbered tasks without the (duration in minutes: ...) tail.
        text = (
            "1. Maeve is sitting down.\n"
            "2. Maeve is ordering lunch.\n"
            "3. Maeve is eating."
        )
        out = self.cleanup(text, prompt=self._prompt_with_total(30))
        self.assertIsInstance(out, list)
        self.assertEqual(sum(d for _, d in out), 30)
        # All three subtasks should be preserved.
        joined = " | ".join(t for t, _ in out)
        self.assertIn("sitting down", joined)
        self.assertIn("ordering lunch", joined)
        self.assertIn("eating", joined)


# ---------------------------------------------------------------------------
# None-return regression tests (False-injection)
# ---------------------------------------------------------------------------
#
# The functions below use ChatGPT_safe_generate_response, which returns False
# when Qwen3 output fails validation on every retry. The originals fell through
# a commented-out OpenAI fallback to an implicit `return None`, so callers doing
# `func(...)[0]` raised `TypeError: 'NoneType' object is not subscriptable`.
#
# Each function now has an explicit fail_safe return after the
# `if output != False:` block. These tests force the wrapper to return False and
# assert the function still returns a non-None (value, debug_list) tuple whose
# [0] is the fail_safe.

def _force_false_chatgpt(run_gpt_fn, *args, **kwargs):
    """Run run_gpt_fn with ChatGPT_safe_generate_response forced to return False.

    Mirrors the production failure mode (validation failed on all retries).
    generate_prompt + print_run_prompts are stubbed so no template files or
    Ollama are needed.
    """
    def fake(*a, **k):
        return False
    with patch("persona.prompt_template.run_gpt_prompt.ChatGPT_safe_generate_response", fake), \
         patch("persona.prompt_template.run_gpt_prompt.print_run_prompts", lambda *a, **k: None), \
         patch("persona.prompt_template.run_gpt_prompt.generate_prompt", return_value="dummy prompt"):
        return run_gpt_fn(*args, **kwargs)


class TestNoneReturnFailSafe(unittest.TestCase):
    """Regression for the None-return audit.

    When ChatGPT_safe_generate_response returns False, every patched function
    must return a non-None tuple whose [0] is its own fail_safe (never None).
    """

    def setUp(self):
        self.persona = _stub_persona()
        self.target = _stub_persona("Klaus", "Klaus Mueller")

    def _assert_failsafe(self, ret, expected):
        self.assertIsNotNone(ret, "function returned None on the ChatGPT False path")
        self.assertIsInstance(ret, tuple)
        self.assertEqual(len(ret), 2)
        self.assertEqual(ret[0], expected)
        # The debug list mirrors fail_safe in slot 0 as well.
        self.assertEqual(ret[1][0], expected)

    def test_pronunciatio(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_pronunciatio
        ret = _force_false_chatgpt(run_gpt_prompt_pronunciatio,
                                   "cooking breakfast", self.persona)
        self._assert_failsafe(ret, "😋")

    def test_act_obj_desc(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_act_obj_desc
        ret = _force_false_chatgpt(run_gpt_prompt_act_obj_desc,
                                   "stove", "cooking", self.persona)
        self._assert_failsafe(ret, "stove is idle")

    def test_summarize_conversation(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_summarize_conversation
        ret = _force_false_chatgpt(run_gpt_prompt_summarize_conversation,
                                   self.persona, [["Maeve", "Hi"], ["Klaus", "Hey"]])
        self._assert_failsafe(ret, "conversing with a housemate about morning greetings")

    def test_event_poignancy(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_event_poignancy
        ret = _force_false_chatgpt(run_gpt_prompt_event_poignancy,
                                   self.persona, "a fire alarm goes off")
        self._assert_failsafe(ret, 4)

    def test_thought_poignancy(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_thought_poignancy
        ret = _force_false_chatgpt(run_gpt_prompt_thought_poignancy,
                                   self.persona, "I should call my mother")
        self._assert_failsafe(ret, 4)

    def test_chat_poignancy(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_chat_poignancy
        ret = _force_false_chatgpt(run_gpt_prompt_chat_poignancy,
                                   self.persona, "a friendly chat about the weather")
        self._assert_failsafe(ret, 4)

    def test_agent_chat_summarize_ideas(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_agent_chat_summarize_ideas
        ret = _force_false_chatgpt(run_gpt_prompt_agent_chat_summarize_ideas,
                                   self.persona, self.target, "some statements", "some context")
        self._assert_failsafe(ret, "...")

    def test_agent_chat_summarize_relationship(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_agent_chat_summarize_relationship
        ret = _force_false_chatgpt(run_gpt_prompt_agent_chat_summarize_relationship,
                                   self.persona, self.target, "some statements")
        self._assert_failsafe(ret, "...")

    def test_agent_chat(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_agent_chat
        # create_prompt_input iterates a_mem.seq_chat; an empty list keeps it
        # from touching the LLM path before our fail_safe return.
        self.persona.a_mem.seq_chat = []
        maze = MagicMock()
        ret = _force_false_chatgpt(run_gpt_prompt_agent_chat,
                                   maze, self.persona, self.target,
                                   "some context", "init idea", "target idea")
        self._assert_failsafe(ret, "...")

    def test_summarize_ideas(self):
        from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_summarize_ideas
        # create_prompt_input iterates norm_database.act_norm; empty dict is fine.
        self.persona.norm_database.act_norm = {}
        ret = _force_false_chatgpt(run_gpt_prompt_summarize_ideas,
                                   self.persona, "some statements", "a question")
        self._assert_failsafe(ret, "...")


if __name__ == "__main__":
    unittest.main()
