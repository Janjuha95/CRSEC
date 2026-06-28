"""
Regression tests for the performance pass (memoization, headless stub,
violation pre-filter, tiered routing). See OPTIMIZATION_LOG.md.

Run from reverie/backend_server/:
    python -m pytest tests/test_optimizations.py -v
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
    global _ORIGINAL_CWD
    _ORIGINAL_CWD = os.getcwd()
    os.chdir(BACKEND)


def tearDownModule():
    if _ORIGINAL_CWD is not None:
        os.chdir(_ORIGINAL_CWD)


# Patch ollama before importing so unit tests don't require Ollama running.
sys.modules.setdefault("ollama", MagicMock())

import call_profiler  # noqa: E402
import llm_router  # noqa: E402
from persona.prompt_template import run_gpt_prompt  # noqa: E402
from norm import run_gpt_prompt_norm  # noqa: E402
from norm import violation_detection  # noqa: E402


class FakeScratch:
    def __init__(self, name):
        self.name = name

    def get_str_iss(self):
        return f"{self.name} ISS block"


class FakePersona:
    def __init__(self, name):
        self.name = name
        self.scratch = FakeScratch(name)


class FakeNorm:
    def __init__(self, id, content, subject, predicate, object):
        self.id = id
        self.content = content
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.activation_state = True


class FakeEvent:
    def __init__(self, subject, predicate, object):
        self.subject = subject
        self.predicate = predicate
        self.object = object


def make_observer(name, norms):
    observer = FakePersona(name)
    db = types.SimpleNamespace(act_norm={f"norm_{i}": n for i, n in enumerate(norms)})
    observer.norm_database = db
    return observer


class EnvMixin:
    def setenv(self, key, value):
        old = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        self.addCleanup(self._restore_env, key, old)

    @staticmethod
    def _restore_env(key, old):
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


class TestViolationCheckMemoization(EnvMixin, unittest.TestCase):
    def setUp(self):
        run_gpt_prompt_norm._VIOLATION_CACHE.clear()
        self.setenv("CRSEC_MEMOIZE", "1")

    def test_cache_hit_identical_shape_and_single_llm_call(self):
        calls = []
        result_dict = {"violation": True, "severity": 5,
                       "certainty": 80, "response": "confront"}

        def fake_safe(prompt, repeat, fail_safe, validate, cleanup, verbose=False):
            calls.append(prompt)
            return dict(result_dict)

        with patch.object(run_gpt_prompt_norm,
                          "GPT4_safe_generate_response_OLD", fake_safe):
            out1 = run_gpt_prompt_norm.run_gpt_prompt_violation_check(
                "Klaus is smoking inside the cafe", "No smoking in the cafe", "Maria")
            out2 = run_gpt_prompt_norm.run_gpt_prompt_violation_check(
                "Klaus is smoking inside the cafe", "No smoking in the cafe", "Maria")

        self.assertEqual(len(calls), 1)  # second call served from cache
        # Identical object shape: (dict, [dict, str, dict, list, dict])
        self.assertEqual(out1[0], out2[0])
        self.assertIsInstance(out2[0], dict)
        self.assertIsInstance(out2[1], list)
        self.assertEqual(len(out2[1]), 5)
        self.assertEqual(out1[0], result_dict)
        # Mutating a returned dict must not poison the cache.
        out1[0]["severity"] = 999
        out3 = run_gpt_prompt_norm.run_gpt_prompt_violation_check(
            "Klaus is smoking inside the cafe", "No smoking in the cafe", "Maria")
        self.assertEqual(out3[0]["severity"], 5)

    def test_fail_safe_not_cached(self):
        calls = []

        def fake_safe(prompt, repeat, fail_safe, validate, cleanup, verbose=False):
            calls.append(prompt)
            return fail_safe  # simulate LLM failure -> fail_safe identity

        with patch.object(run_gpt_prompt_norm,
                          "GPT4_safe_generate_response_OLD", fake_safe):
            run_gpt_prompt_norm.run_gpt_prompt_violation_check("e", "n", "o")
            run_gpt_prompt_norm.run_gpt_prompt_violation_check("e", "n", "o")
        self.assertEqual(len(calls), 2)  # not memoized

    def test_memoize_disabled(self):
        self.setenv("CRSEC_MEMOIZE", "0")
        calls = []

        def fake_safe(prompt, repeat, fail_safe, validate, cleanup, verbose=False):
            calls.append(prompt)
            return {"violation": False, "severity": 0, "certainty": 0,
                    "response": "ignore"}

        with patch.object(run_gpt_prompt_norm,
                          "GPT4_safe_generate_response_OLD", fake_safe):
            run_gpt_prompt_norm.run_gpt_prompt_violation_check("e", "n", "o")
            run_gpt_prompt_norm.run_gpt_prompt_violation_check("e", "n", "o")
        self.assertEqual(len(calls), 2)


class TestPoignancyAndTripleMemoization(EnvMixin, unittest.TestCase):
    def setUp(self):
        run_gpt_prompt._PROMPT_CACHE.clear()
        self.setenv("CRSEC_MEMOIZE", "1")
        self.setenv("CRSEC_HEADLESS", None)

    def test_event_poignancy_cached(self):
        persona = FakePersona("Klaus Mueller")
        calls = []

        def fake_chat(prompt, example, special, repeat, fail_safe,
                      validate, cleanup, verbose=False):
            calls.append(prompt)
            return 7

        with patch.object(run_gpt_prompt,
                          "ChatGPT_safe_generate_response", fake_chat):
            out1 = run_gpt_prompt.run_gpt_prompt_event_poignancy(
                persona, "Klaus Mueller is writing his thesis")
            out2 = run_gpt_prompt.run_gpt_prompt_event_poignancy(
                persona, "Klaus Mueller is writing his thesis")

        self.assertEqual(len(calls), 1)
        self.assertEqual(out1[0], 7)
        self.assertEqual(out1[0], out2[0])
        self.assertIsInstance(out2[1], list)
        self.assertEqual(len(out2[1]), 5)

    def test_event_triple_cached(self):
        persona = FakePersona("Klaus Mueller")
        calls = []

        def fake_safe(prompt, repeat, fail_safe, validate, cleanup, verbose=False):
            calls.append(prompt)
            return ["is", "writing"]

        with patch.object(run_gpt_prompt,
                          "GPT4_safe_generate_response_OLD", fake_safe):
            out1 = run_gpt_prompt.run_gpt_prompt_event_triple("writing thesis", persona)
            out2 = run_gpt_prompt.run_gpt_prompt_event_triple("writing thesis", persona)

        self.assertEqual(len(calls), 1)
        self.assertEqual(out1[0], ("Klaus Mueller", "is", "writing"))
        self.assertEqual(out1[0], out2[0])

    def test_poignancy_keyed_on_iss(self):
        """Changed ISS (revised 'currently') must miss the cache."""
        persona = FakePersona("Klaus Mueller")
        calls = []

        def fake_chat(prompt, example, special, repeat, fail_safe,
                      validate, cleanup, verbose=False):
            calls.append(prompt)
            return 5

        with patch.object(run_gpt_prompt,
                          "ChatGPT_safe_generate_response", fake_chat):
            run_gpt_prompt.run_gpt_prompt_event_poignancy(persona, "desc")
            persona.scratch.get_str_iss = lambda: "revised ISS"
            run_gpt_prompt.run_gpt_prompt_event_poignancy(persona, "desc")
        self.assertEqual(len(calls), 2)


class TestPronunciatioHeadlessStub(EnvMixin, unittest.TestCase):
    def setUp(self):
        run_gpt_prompt._PROMPT_CACHE.clear()

    def test_headless_stub_returns_str_without_llm(self):
        self.setenv("CRSEC_HEADLESS", "1")

        def boom(*args, **kwargs):
            raise AssertionError("LLM should not be called in headless mode")

        persona = FakePersona("Klaus Mueller")
        with patch.object(run_gpt_prompt,
                          "ChatGPT_safe_generate_response", boom):
            out = run_gpt_prompt.run_gpt_prompt_pronunciatio("sleeping", persona)
        self.assertIsInstance(out[0], str)
        self.assertEqual(out[0], "💬")
        self.assertIsInstance(out[1], list)
        self.assertEqual(len(out[1]), 5)

    def test_non_headless_calls_llm_and_caches(self):
        self.setenv("CRSEC_HEADLESS", None)
        self.setenv("CRSEC_MEMOIZE", "1")
        calls = []

        def fake_chat(prompt, example, special, repeat, fail_safe,
                      validate, cleanup, verbose=False):
            calls.append(prompt)
            return "🛌"

        persona = FakePersona("Klaus Mueller")
        with patch.object(run_gpt_prompt,
                          "ChatGPT_safe_generate_response", fake_chat):
            out1 = run_gpt_prompt.run_gpt_prompt_pronunciatio("sleeping", persona)
            out2 = run_gpt_prompt.run_gpt_prompt_pronunciatio("sleeping", persona)
        self.assertEqual(len(calls), 1)
        self.assertEqual(out1[0], out2[0])
        self.assertIsInstance(out1[0], str)


class TestViolationPrefilter(EnvMixin, unittest.TestCase):
    def setUp(self):
        self.setenv("CRSEC_VIOLATION_PREFILTER", "1")
        self.norm = FakeNorm(1, "No smoking is allowed inside the cafe.",
                             "no one", "is allowed", "to smoke inside the cafe")
        self.observer = make_observer("Maria Lopez", [self.norm])
        self.personas = {"Maria Lopez": None, "Klaus Mueller": None}

    def test_benign_event_skips_llm(self):
        def boom(*args, **kwargs):
            raise AssertionError("LLM should not be called for benign events")

        events = [FakeEvent("Klaus Mueller", "is idle", ""),
                  FakeEvent("Klaus Mueller", "sleeping", "bed"),
                  FakeEvent("Klaus Mueller", "chat with", "Maria Lopez")]
        with patch.object(violation_detection,
                          "run_gpt_prompt_violation_check", boom):
            out = violation_detection.detect_violations(
                self.observer, events, self.personas)
        self.assertEqual(out, [])

    def test_no_stem_overlap_skips_llm(self):
        def boom(*args, **kwargs):
            raise AssertionError("LLM should not be called without stem overlap")

        events = [FakeEvent("Klaus Mueller", "watering", "the garden plants")]
        with patch.object(violation_detection,
                          "run_gpt_prompt_violation_check", boom):
            out = violation_detection.detect_violations(
                self.observer, events, self.personas)
        self.assertEqual(out, [])

    def test_overlap_reaches_llm(self):
        calls = []

        def fake_check(event_desc, norm_content, observer_name, verbose=False):
            calls.append(event_desc)
            return ({"violation": True, "severity": 5, "certainty": 80,
                     "response": "confront"}, None)

        events = [FakeEvent("Klaus Mueller", "smoking", "inside the cafe")]
        with patch.object(violation_detection,
                          "run_gpt_prompt_violation_check", fake_check):
            out = violation_detection.detect_violations(
                self.observer, events, self.personas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["violator"], "Klaus Mueller")

    def test_prefilter_never_raises_on_malformed_events(self):
        events = [FakeEvent(None, None, None),
                  FakeEvent("Klaus Mueller", None, None),
                  FakeEvent("Klaus Mueller", "", ""),
                  FakeEvent("not a persona", "smoking", "cafe"),
                  "not even an event object"]

        def fake_check(*args, **kwargs):
            return ({"violation": False}, None)

        with patch.object(violation_detection,
                          "run_gpt_prompt_violation_check", fake_check):
            out = violation_detection.detect_violations(
                self.observer, events, self.personas)
        self.assertIsInstance(out, list)

    def test_prefilter_disabled_calls_llm_for_benign(self):
        self.setenv("CRSEC_VIOLATION_PREFILTER", "0")
        calls = []

        def fake_check(event_desc, norm_content, observer_name, verbose=False):
            calls.append(event_desc)
            return ({"violation": False}, None)

        events = [FakeEvent("Klaus Mueller", "is idle", "")]
        with patch.object(violation_detection,
                          "run_gpt_prompt_violation_check", fake_check):
            violation_detection.detect_violations(
                self.observer, events, self.personas)
        self.assertEqual(len(calls), 1)

    def test_empty_norm_stems_fail_open(self):
        """A norm with no extractable stems must still reach the LLM."""
        norm = FakeNorm(2, "", "", "", "")
        observer = make_observer("Maria Lopez", [norm])
        calls = []

        def fake_check(event_desc, norm_content, observer_name, verbose=False):
            calls.append(event_desc)
            return ({"violation": False}, None)

        events = [FakeEvent("Klaus Mueller", "smoking", "inside the cafe")]
        with patch.object(violation_detection,
                          "run_gpt_prompt_violation_check", fake_check):
            violation_detection.detect_violations(observer, events, self.personas)
        self.assertEqual(len(calls), 1)


class TestTieredRouting(EnvMixin, unittest.TestCase):
    def _call_via(self, fn_name):
        """Invoke llm_call from a frame named fn_name; return model used."""
        captured = {}

        def fake_chat(**kwargs):
            captured.update(kwargs)
            return {"message": {"content": "ok"}}

        wrapper_code = compile(
            f"def {fn_name}():\n"
            f"    return llm_router.llm_call('p', 'conversation')\n",
            "<test>", "exec")
        ns = {"llm_router": llm_router}
        exec(wrapper_code, ns)
        with patch.object(llm_router.ollama, "chat", fake_chat), \
                patch.object(llm_router, "_log_call", lambda *a, **k: None):
            ns[fn_name]()
        return captured

    def test_small_model_for_cosmetic_fns(self):
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        kwargs = self._call_via("run_gpt_prompt_pronunciatio")
        self.assertEqual(kwargs["model"], llm_router.SMALL_MODEL)

    def test_reasoning_model_for_other_fns(self):
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        kwargs = self._call_via("run_gpt_prompt_violation_check")
        self.assertEqual(kwargs["model"], llm_router.REASONING_MODEL)

    def test_poignancy_uses_primary_model(self):
        """Part B regression: poignancy must NOT route to the small model even
        with tiered routing on (qwen3:4b can't satisfy the strict JSON
        envelope of ChatGPT_safe_generate_response)."""
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        for fn in ("run_gpt_prompt_event_poignancy",
                   "run_gpt_prompt_thought_poignancy",
                   "run_gpt_prompt_chat_poignancy"):
            kwargs = self._call_via(fn)
            self.assertEqual(kwargs["model"], llm_router.REASONING_MODEL,
                             f"{fn} should route to the primary model")

    def test_tiered_routing_disabled(self):
        self.setenv("CRSEC_TIERED_ROUTING", "0")
        kwargs = self._call_via("run_gpt_prompt_pronunciatio")
        self.assertEqual(kwargs["model"], llm_router.REASONING_MODEL)

    def test_options_pin_temperature_and_num_ctx(self):
        kwargs = self._call_via("run_gpt_prompt_violation_check")
        self.assertEqual(kwargs["options"]["temperature"], 0.0)
        self.assertEqual(kwargs["options"]["num_ctx"], llm_router.OLLAMA_NUM_CTX)
        self.assertFalse(kwargs["think"])


class TestConflictParserUnit(unittest.TestCase):
    """Part C: _final_output_decision reads the FINAL OUTPUT line
    authoritatively (markdown / brackets / preamble tolerant)."""

    def test_reasons_no_final_output_no(self):
        sample = (
            "- Question 1: whether there is a conflict?\n"
            "Reasoning: It would be a conflict, yes, only if Klaus were "
            "littering, but he is idle, so there is no conflict.\n"
            'Answer in "yes" or "no" and provide a reason: No // idle.\n'
            "- FINAL OUTPUT: No."
        )
        self.assertEqual(run_gpt_prompt_norm._final_output_decision(sample), "no")

    def test_final_output_yes(self):
        sample = ("Reasoning: there is clearly a conflict here.\n"
                  "- FINAL OUTPUT: Yes.")
        self.assertEqual(run_gpt_prompt_norm._final_output_decision(sample), "yes")

    def test_markdown_and_brackets(self):
        self.assertEqual(
            run_gpt_prompt_norm._final_output_decision("FINAL OUTPUT: **No**"), "no")
        self.assertEqual(
            run_gpt_prompt_norm._final_output_decision("FINAL OUTPUT: [Yes]"), "yes")

    def test_answer_on_next_line(self):
        self.assertEqual(
            run_gpt_prompt_norm._final_output_decision("FINAL OUTPUT:\nNo."), "no")

    def test_no_decision_returns_none(self):
        self.assertIsNone(
            run_gpt_prompt_norm._final_output_decision("Unable to assess."))


class TestConflictParserIntegration(unittest.TestCase):
    """Part C end-to-end: run_gpt_prompt_decide_if_norm_conflict must not
    invert No->Yes, and ['ERROR'] must cleanly mean skip."""

    def _decide(self, response_text):
        from persona.prompt_template import gpt_structure
        with patch.object(gpt_structure, "GPT4_request",
                          lambda prompt: response_text), \
                patch.object(run_gpt_prompt_norm, "debug", False):
            out, _meta = (
                run_gpt_prompt_norm.run_gpt_prompt_decide_if_norm_conflict(
                    "Klaus Mueller is idle",
                    "People should not litter in the park.",
                    "Klaus Mueller", "citizen",
                    "friendly and conscientious"))
        return out

    def test_reasons_no_final_output_no_is_no_conflict(self):
        # Reasoning carries a stray 'yes'; conclusion + FINAL OUTPUT are No.
        # The old parser grabbed the stray 'yes' (no->yes inversion).
        sample = (
            "Let's think step by step.\n"
            "- Question 1: whether there is a conflict?\n"
            "Reasoning: The norm is about littering. It would be a conflict, "
            "yes, only if Klaus were littering, but he is idle. So there is "
            "no conflict.\n"
            'Answer in "yes" or "no" and provide a reason: No // Klaus is idle.\n'
            "- Question 2: whether to have a conversation about the conflict?\n"
            "Reasoning: There is no conflict, so no conversation is needed.\n"
            'Answer in "yes" or "no" and provide a reason: No // nothing.\n'
            "- Question 3: ...\nAnswer: []\n"
            "- FINAL OUTPUT: No."
        )
        out = self._decide(sample)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "no")   # conversation/talk decision
        self.assertEqual(out[2], "no")   # conflict=False (not inverted)

    def test_final_output_yes_is_conflict(self):
        sample = ("- Question 1: whether there is a conflict?\n"
                  'Answer in "yes" or "no" and provide a reason: Yes // smoking.\n'
                  "- FINAL OUTPUT: Yes.")
        out = self._decide(sample)
        self.assertEqual(out[0], "yes")
        self.assertEqual(out[2], "yes")

    def test_unparseable_returns_error_skip(self):
        out = self._decide("Unable to assess the situation right there.")
        self.assertEqual(out, ["ERROR"])
        # Caller checks output[0] == "yes"; "ERROR" cleanly means skip.
        self.assertNotEqual(out[0], "yes")


class TestProfiler(unittest.TestCase):
    def test_record_and_snapshot(self):
        call_profiler.reset()
        call_profiler.record_call("run_gpt_prompt_x", 1.5)
        call_profiler.record_call("run_gpt_prompt_x", 0.5)
        call_profiler.incr("some_counter")
        snap = call_profiler.snapshot()
        self.assertEqual(snap["total_llm_calls"], 2)
        self.assertEqual(snap["per_function"]["run_gpt_prompt_x"]["count"], 2)
        self.assertAlmostEqual(snap["total_llm_seconds"], 2.0)
        self.assertEqual(snap["counters"]["some_counter"], 1)

    def test_dump_never_raises(self):
        call_profiler.dump(os.path.join(HERE, "..", "nonexistent_dir_zz",
                                        "x", "profile.json"))


if __name__ == "__main__":
    unittest.main()
