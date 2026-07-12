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
    def _call_via(self, fn_name, call_type="conversation"):
        """Invoke llm_call from a frame named fn_name; return model kwargs used."""
        captured = {}

        def fake_chat(**kwargs):
            captured.update(kwargs)
            return {"message": {"content": "ok"}}

        wrapper_code = compile(
            f"def {fn_name}():\n"
            f"    return llm_router.llm_call('p', {call_type!r})\n",
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

    def test_primary_model_for_bulk_fns(self):
        """Bulk functions not in SMALL or REASONING sets default to PRIMARY_MODEL."""
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        kwargs = self._call_via("run_gpt_prompt_violation_check")
        self.assertEqual(kwargs["model"], llm_router.PRIMARY_MODEL)

    def test_poignancy_uses_primary_model(self):
        """Poignancy must NOT go to SMALL (qwen3:4b fails the strict JSON
        envelope) and is not in REASONING_ROUTE_PROMPT_FNS, so it goes to
        PRIMARY_MODEL (tier 4 default with call_type='conversation')."""
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        for fn in ("run_gpt_prompt_event_poignancy",
                   "run_gpt_prompt_thought_poignancy",
                   "run_gpt_prompt_chat_poignancy"):
            kwargs = self._call_via(fn)
            self.assertEqual(kwargs["model"], llm_router.PRIMARY_MODEL,
                             f"{fn} should route to PRIMARY_MODEL")

    def test_tiered_routing_disabled(self):
        self.setenv("CRSEC_TIERED_ROUTING", "0")
        kwargs = self._call_via("run_gpt_prompt_pronunciatio")
        self.assertEqual(kwargs["model"], llm_router.REASONING_MODEL)

    def test_options_pin_temperature_and_num_ctx(self):
        kwargs = self._call_via("run_gpt_prompt_violation_check")
        self.assertEqual(kwargs["options"]["temperature"], 0.0)
        self.assertEqual(kwargs["options"]["num_ctx"], llm_router.OLLAMA_NUM_CTX)
        self.assertFalse(kwargs["think"])

    def test_three_tier_mapping(self):
        """Consolidated: verify the five required routing assertions end-to-end."""
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        # run_gpt_prompt_daily_plan → tier 4 default → PRIMARY
        self.assertEqual(
            self._call_via("run_gpt_prompt_daily_plan")["model"],
            llm_router.PRIMARY_MODEL)
        # run_gpt_generate_iterative_chat_utt → tier 2 REASONING_ROUTE_PROMPT_FNS → REASONING
        self.assertEqual(
            self._call_via("run_gpt_generate_iterative_chat_utt")["model"],
            llm_router.REASONING_MODEL)
        # run_gpt_prompt_pronunciatio → tier 1 SMALL_ROUTE_PROMPT_FNS → SMALL
        self.assertEqual(
            self._call_via("run_gpt_prompt_pronunciatio")["model"],
            llm_router.SMALL_MODEL)
        # call_type="norm_evaluation" from generic frame → tier 3 REASONING_CALL_TYPES → REASONING
        self.assertEqual(
            self._call_via("run_gpt_prompt_generic", call_type="norm_evaluation")["model"],
            llm_router.REASONING_MODEL)
        # call_type="violation_check" from generic frame → tier 4 → PRIMARY
        self.assertEqual(
            self._call_via("run_gpt_prompt_generic", call_type="violation_check")["model"],
            llm_router.PRIMARY_MODEL)

    def test_norm_prefix_fn_routes_to_reasoning(self):
        """Functions starting with run_gpt_prompt_norm hit REASONING via
        startswith — unless they are in NORM_PRIMARY_OVERRIDE_FNS."""
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        kwargs = self._call_via("run_gpt_prompt_norm_format")
        self.assertEqual(kwargs["model"], llm_router.REASONING_MODEL)

    def test_norm_on_primary_default(self):
        """CRSEC_NORM_ON_PRIMARY defaults on: the two dominant norm fns go to
        PRIMARY, checked before the tier-2 set / norm prefix rule."""
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        for fn in ("run_gpt_prompt_decide_if_norm_conflict",
                   "run_gpt_prompt_norm_reflect_from_thoughts"):
            self.assertEqual(self._call_via(fn)["model"],
                             llm_router.PRIMARY_MODEL,
                             f"{fn} should route to PRIMARY_MODEL by default")

    def test_norm_on_primary_disabled_restores_reasoning(self):
        """CRSEC_NORM_ON_PRIMARY=0 restores the previous 32b routing exactly."""
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        self.setenv("CRSEC_NORM_ON_PRIMARY", "0")
        for fn in ("run_gpt_prompt_decide_if_norm_conflict",
                   "run_gpt_prompt_norm_reflect_from_thoughts"):
            self.assertEqual(self._call_via(fn)["model"],
                             llm_router.REASONING_MODEL,
                             f"{fn} should route to REASONING_MODEL with flag=0")

    def _call_with_prompt_fn(self, prompt_fn, call_type):
        captured = {}

        def fake_chat(**kwargs):
            captured.update(kwargs)
            return {"message": {"content": "ok"}}

        with patch.object(llm_router.ollama, "chat", fake_chat), \
                patch.object(llm_router, "_log_call", lambda *a, **k: None):
            llm_router.llm_call("p", call_type, prompt_fn=prompt_fn)
        return captured

    def test_prompt_fn_override_routes_and_caps_utility_scorer(self):
        """The explicit prompt_fn= override replaces the stack walk: the
        utility scorer follows CRSEC_NORM_ON_PRIMARY (default on) and gets
        its data-derived cap instead of the unknown-fn default."""
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        kwargs = self._call_with_prompt_fn("run_gpt_specific_norm_utility",
                                           "norm_evaluation")
        self.assertEqual(kwargs["model"], llm_router.PRIMARY_MODEL)
        self.assertEqual(kwargs["options"]["num_predict"], 2712)

    def test_prompt_fn_override_flag_off_restores_32b(self):
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        self.setenv("CRSEC_NORM_ON_PRIMARY", "0")
        kwargs = self._call_with_prompt_fn("run_gpt_specific_norm_utility",
                                           "norm_evaluation")
        # tier-3 call_type fallback: exactly the old routing
        self.assertEqual(kwargs["model"], llm_router.REASONING_MODEL)

    def test_event_triple_routes_to_primary(self):
        """run_gpt_prompt_event_triple was moved from SMALL to PRIMARY."""
        self.setenv("CRSEC_TIERED_ROUTING", "1")
        kwargs = self._call_via("run_gpt_prompt_event_triple")
        self.assertEqual(kwargs["model"], llm_router.PRIMARY_MODEL)


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


class TestPoignancyEnvelopeTolerance(EnvMixin, unittest.TestCase):
    """Part B: ChatGPT_safe_generate_response tolerates non-envelope qwen3
    output, so poignancy returns real ints instead of [FAIL_SAFE]."""

    def setUp(self):
        run_gpt_prompt._PROMPT_CACHE.clear()
        self.setenv("CRSEC_MEMOIZE", "1")
        self.setenv("CRSEC_HEADLESS", None)

    def _poignancy(self, raw_response, desc):
        # Patch at the request boundary so the REAL
        # ChatGPT_safe_generate_response envelope handling runs.
        from persona.prompt_template import gpt_structure
        with patch.object(gpt_structure, "ChatGPT_request",
                          lambda prompt: raw_response):
            out, _meta = run_gpt_prompt.run_gpt_prompt_event_poignancy(
                FakePersona("Klaus Mueller"), desc)
        return out

    def test_envelope_tolerant_paths(self):
        cases = [
            ("5", 5),                         # bare int
            ("I'd rate it 5/10.", 5),         # prose + slash
            ('text\n{"output":"7"}', 7),      # prose before the envelope
            ('{"output":"3"}', 3),            # clean envelope (unchanged path)
        ]
        for i, (raw, expected) in enumerate(cases):
            out = self._poignancy(raw, f"event number {i}")
            self.assertEqual(out, expected, f"raw={raw!r} -> {out!r}")
            self.assertNotEqual(out, 4, f"fail-safe fired for raw={raw!r}")


class TestSpecificNormUtilityPort(unittest.TestCase):
    """Part C: SpecificNormUtility routes through llm_call and always returns a
    length-2 [score, reason] (no OpenAI NameError, no bare False)."""

    def test_parses_output_line(self):
        snu = run_gpt_prompt_norm.SpecificNormUtility("system msg")
        with patch.object(run_gpt_prompt_norm, "llm_call",
                          return_value="OUTPUT: 7. because X"):
            res = snu.specific_norm_utility("No smoking in the cafe.")
        self.assertEqual(res, [7, "because X"])

    def test_failsafe_is_length_2(self):
        snu = run_gpt_prompt_norm.SpecificNormUtility("system msg")

        def boom(*a, **k):
            raise RuntimeError("inference down")

        with patch.object(run_gpt_prompt_norm, "llm_call", boom):
            res = snu.specific_norm_utility("No smoking in the cafe.")
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 2)

    def test_overnight_hardening_caps(self):
        """Jul 12 hardening: embed guard + text-bloat caps. No behavior
        change for well-sized text; oversized text clips loudly instead of
        crashing (exp1_base_r1_c2 died on an unguarded day-2 embedding)."""
        import datetime as dt

        from persona.prompt_template import gpt_structure

        # 1. oversized input to get_embedding clips and succeeds
        seen = []

        def fake_embed(model=None, prompt=None):
            seen.append(len(prompt))
            return {"embedding": [0.0]}

        with patch("ollama.embeddings", side_effect=fake_embed), \
                patch.object(gpt_structure, "temp_sleep", lambda *a: None):
            out = gpt_structure.get_embedding("x" * 20000)
        self.assertEqual(out, [0.0])
        self.assertEqual(seen, [8000])  # clipped to CRSEC_EMBED_MAX_CHARS default

        # small input passes through unchanged (identical behavior under cap)
        seen.clear()
        with patch("ollama.embeddings", side_effect=fake_embed), \
                patch.object(gpt_structure, "temp_sleep", lambda *a: None):
            gpt_structure.get_embedding("short text")
        self.assertEqual(seen, [len("short text")])

        # 2. a "context length" error triggers the halving retry
        calls = []

        def flaky_embed(model=None, prompt=None):
            calls.append(len(prompt))
            if len(prompt) > 2500:
                raise Exception("the input length exceeds the context length (status code: 500)")
            return {"embedding": [1.0]}

        with patch("ollama.embeddings", side_effect=flaky_embed), \
                patch.object(gpt_structure, "temp_sleep", lambda *a: None):
            out = gpt_structure.get_embedding("y" * 9000)
        self.assertEqual(out, [1.0])
        self.assertEqual(calls, [8000, 4000, 2000])  # clip, halve, halve, ok

        # an unrelated error re-raises immediately (no silent junk)
        with patch("ollama.embeddings",
                   side_effect=Exception("connection refused")), \
                patch.object(gpt_structure, "temp_sleep", lambda *a: None):
            with self.assertRaises(Exception):
                gpt_structure.get_embedding("z" * 100)

        # 3. daily_plan_v2 cleanup caps a 50-line ramble via the real
        # parse path (line-fallback), wired through the safe wrapper
        ramble = "\n".join(f"- item {i}: " + "w" * 260 for i in range(50))
        plan_persona = types.SimpleNamespace(name="Bob Johnson")
        with patch.object(gpt_structure, "ChatGPT_request",
                          return_value=ramble):
            output, _ = run_gpt_prompt_norm.run_gpt_prompt_daily_plan_v2(
                plan_persona, 7, "norms",
                test_input=["ISS", "lifestyle", "Monday", "First",
                            "7:00 am", "norms"])
        items = output[1:]  # [0] is the prepended wake-up item
        self.assertEqual(len(items), 20)
        self.assertTrue(all(len(i) <= 200 for i in items))

        # an under-cap plan passes through unchanged
        ok_items = ["have breakfast", "walk to the cafe"]
        self.assertEqual(run_gpt_prompt_norm._cap_plan_items(list(ok_items)),
                         ok_items)

        # 4. revise-identity cleanup caps a 5000-char tail to <=1000 chars,
        # via the real cleanup path (daily_plan_req variant)
        tail = ("Bob plans to keep the cafe tidy. " * 200).strip()  # ~6600 chars
        persona = types.SimpleNamespace(scratch=types.SimpleNamespace(
            get_str_iss=lambda: "ISS block",
            curr_time=dt.datetime(2023, 2, 14, 9, 0),
            name="Bob Johnson"))
        with patch.object(gpt_structure, "ChatGPT_request", return_value=tail):
            output, _ = run_gpt_prompt_norm.run_gpt_revise_identity_daily_plan_req(
                persona, "norms")
        self.assertLessEqual(len(output), 1000)
        self.assertTrue(output.endswith("."))  # cut at a sentence boundary

        # under-limit identity text passes through unchanged
        self.assertEqual(
            run_gpt_prompt_norm._cap_identity_text("Short status.", "t"),
            "Short status.")

    def test_utility_parser_tolerates_markdown(self):
        """Pass 4.5: the dominant calib_010/011 shape — bold marker + bold
        score + bullet — must parse; a refusal must raise (defer)."""
        parse = run_gpt_prompt_norm._parse_norm_utility_response
        self.assertEqual(
            parse("- **INPUT**: Loud tipping talk.\n"
                  "- **OUTPUT**: **30**. Because it disrupts the cafe."),
            [30, "Because it disrupts the cafe."])
        self.assertEqual(
            parse("- OUTPUT: **75**.  \n- **Because**: volume norms matter."),
            [75, "Because volume norms matter."])
        self.assertEqual(parse("OUTPUT: 7. because X"), [7, "because X"])
        with self.assertRaises(ValueError):
            parse("**N/A**. Because this reflects personal relationships.")

    def test_consumer_does_not_raise_on_failsafe(self):
        """generate_normal_norm_utility (the norm_evaluate consumer feeding the
        `len(utility) != 2` check) must return a length-2 value, not crash,
        when inference fails."""
        from norm import norm_evaluate
        persona = types.SimpleNamespace(scratch=types.SimpleNamespace(
            is_defector=lambda: False, get_str_iss=lambda: "ISS block"))
        norm = types.SimpleNamespace(content="No smoking in the cafe.")

        def boom(*a, **k):
            raise RuntimeError("inference down")

        with patch.object(run_gpt_prompt_norm, "llm_call", boom):
            util = norm_evaluate.generate_normal_norm_utility(norm, persona)
        self.assertTrue(isinstance(util, (list, tuple)) and len(util) == 2)


class TestDefectorUtilityTolerantParser(unittest.TestCase):
    """calib_014: 93 defector utility evals deferred because the parser
    demanded the literal 'OUTPUT: <int>.' scaffold. The tolerant parser must
    accept realistic qwen3-instruct shapes; garbage still fail-safes."""

    def _parse(self, response):
        from norm import defection_engine
        return defection_engine._parse_defector_utility(response)

    def test_exact_template_shape_unchanged(self):
        score, reason = self._parse(
            "OUTPUT: 85. Because getting caught would hurt my reputation.")
        self.assertEqual(score, 85)
        self.assertEqual(reason,
                         "Because getting caught would hurt my reputation.")

    def test_bare_score_line(self):
        score, reason = self._parse("85. Because nobody enforces this.")
        self.assertEqual(score, 85)
        self.assertEqual(reason, "Because nobody enforces this.")

    def test_markdown_bold_output_with_dash(self):
        score, reason = self._parse("**OUTPUT:** 12 — trivial to ignore.")
        self.assertEqual(score, 12)
        self.assertEqual(reason, "trivial to ignore.")

    def test_prose_preamble_with_score_label(self):
        score, reason = self._parse(
            "Sure! Here's my rating.\nScore: 40. Because it depends who's watching.")
        self.assertEqual(score, 40)
        self.assertEqual(reason, "Because it depends who's watching.")

    def test_inline_rating_sentence(self):
        score, reason = self._parse(
            "I would rate this norm 95 out of 100. Because legal consequences.")
        self.assertEqual(score, 95)
        self.assertIn("legal consequences", reason)

    def test_reason_clipped_and_never_empty(self):
        score, reason = self._parse("OUTPUT: 70. " + "Because reasons. " * 50)
        self.assertEqual(score, 70)
        self.assertLessEqual(len(reason), 300)
        score, reason = self._parse("OUTPUT: 70")
        self.assertEqual(reason, "no reason given")

    def test_garbage_fail_safes_through_caller(self):
        from norm import defection_engine
        persona = types.SimpleNamespace(scratch=types.SimpleNamespace())
        for bad in ("total garbage response", "", "OUTPUT: high",
                    "score is nine hundred: 900"):
            with patch.object(defection_engine, "llm_call", return_value=bad):
                res = defection_engine.get_defector_norm_utility("norm", persona)
            self.assertEqual(res, [4, "fail_safe"], f"input {bad!r}")


class TestDefectorNormUtilityShape(unittest.TestCase):
    """Part C: the defector mirror is shape-matched to SpecificNormUtility."""

    def test_parses_and_failsafe_length_2(self):
        from norm import defection_engine
        persona = types.SimpleNamespace(scratch=types.SimpleNamespace())

        with patch.object(defection_engine, "llm_call",
                          return_value="OUTPUT: 7. because X"):
            res = defection_engine.get_defector_norm_utility("norm", persona)
        self.assertEqual(res, [7, "because X"])

        def boom(*a, **k):
            raise RuntimeError("inference down")

        with patch.object(defection_engine, "llm_call", boom):
            res = defection_engine.get_defector_norm_utility("norm", persona)
        self.assertTrue(isinstance(res, (list, tuple)) and len(res) == 2)


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

    def test_set_dump_path_writes_file_with_counts(self):
        """set_dump_path creates the file immediately (step 0) and a later
        dump reflects the recorded calls."""
        import json
        import tempfile
        call_profiler.reset()
        self.addCleanup(call_profiler.reset)
        # Reset the global dump path so a later test's record_call doesn't try
        # to flush to the (deleted) temp dir.
        self.addCleanup(call_profiler.set_dump_path, None)

        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "profile.json")
            call_profiler.set_dump_path(target)
            # File exists from step 0, before any calls are recorded.
            self.assertTrue(os.path.exists(target))

            n = 7
            for _ in range(n):
                call_profiler.record_call("run_gpt_prompt_demo", 0.01)
            call_profiler.dump(target)

            with open(target, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["total_llm_calls"], n)
            self.assertTrue(data["per_function"])  # non-empty
            self.assertEqual(
                data["per_function"]["run_gpt_prompt_demo"]["count"], n)


if __name__ == "__main__":
    unittest.main()
