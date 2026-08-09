"""
Tests for Modification 1 (Strategic Defector Agent) and Modification 2
(Dual Utility Function).

Run from reverie/backend_server/:
    python -m pytest tests/test_modifications.py -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

PROJECT_ROOT = os.path.dirname(os.path.dirname(BACKEND))  # .../CRSEC


_ORIGINAL_CWD = None


def setUpModule():
    # Several production code paths (defection_engine, run_gpt_prompt_norm) read
    # prompt files via paths relative to the backend_server dir. Pin cwd so the
    # tests pass regardless of how they were launched.
    global _ORIGINAL_CWD
    _ORIGINAL_CWD = os.getcwd()
    os.chdir(BACKEND)


def tearDownModule():
    if _ORIGINAL_CWD is not None:
        os.chdir(_ORIGINAL_CWD)
BASE_SIM_DIR = os.path.join(
    PROJECT_ROOT,
    "environment",
    "frontend_server",
    "storage",
    "base_ville_n10_with_norm",
)
CARLOS_SCRATCH = os.path.join(
    BASE_SIM_DIR, "personas", "Carlos Gomez", "bootstrap_memory", "scratch.json"
)


def _load_template_scratch():
    with open(CARLOS_SCRATCH, "r") as f:
        return json.load(f)


def _write_scratch(tmpdir, data, name="scratch.json"):
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestScratchDefectorFields(unittest.TestCase):
    def test_scratch_defector_defaults(self):
        """A citizen scratch.json (no defector fields) should load with safe defaults."""
        from persona.memory_structures.scratch import Scratch

        with tempfile.TemporaryDirectory() as tmp:
            data = _load_template_scratch()
            # Carlos's JSON has identity="citizen" and no defector fields.
            self.assertNotIn("agent_type", data)
            scratch_path = _write_scratch(tmp, data)

            s = Scratch(scratch_path)

            # agent_type falls back to identity ("citizen").
            self.assertEqual(s.agent_type, data["identity"])
            self.assertEqual(s.boldness, 0)
            self.assertEqual(s.trust_score, 100)
            self.assertFalse(s.is_defector())
            self.assertEqual(s.reputation_beliefs, {})
            self.assertEqual(s.violation_history, [])
            self.assertEqual(s.observed_violations, [])

    def test_scratch_defector_loading(self):
        """A scratch.json with defector fields set should load them."""
        from persona.memory_structures.scratch import Scratch

        with tempfile.TemporaryDirectory() as tmp:
            data = _load_template_scratch()
            data["identity"] = "defector"
            data["agent_type"] = "defector"
            data["boldness"] = 8
            data["trust_score"] = 100
            scratch_path = _write_scratch(tmp, data)

            s = Scratch(scratch_path)

            self.assertTrue(s.is_defector())
            self.assertEqual(s.boldness, 8)
            self.assertEqual(s.trust_score, 100)
            self.assertEqual(s.agent_type, "defector")

    def test_scratch_save_roundtrip(self):
        """Defector fields should survive a save()/reload cycle."""
        import datetime

        from persona.memory_structures.scratch import Scratch

        with tempfile.TemporaryDirectory() as tmp:
            data = _load_template_scratch()
            scratch_path = _write_scratch(tmp, data)
            s = Scratch(scratch_path)

            # save() requires datetime instances for these fields.
            s.curr_time = datetime.datetime(2023, 2, 13, 9, 0, 0)
            s.act_start_time = datetime.datetime(2023, 2, 13, 9, 0, 0)

            s.agent_type = "defector"
            s.boldness = 7
            s.vengefulness = 3
            s.risk_tolerance = 9
            s.reputation_concern = 4
            s.trust_score = 88
            s.reputation_beliefs = {"Bob Johnson": 42}
            s.violation_history = [{"norm": "no smoking", "when": "2023-02-13"}]
            s.observed_violations = [{"agent": "Derek Nash", "norm": "tipping"}]

            out_path = os.path.join(tmp, "scratch_out.json")
            s.save(out_path)

            s2 = Scratch(out_path)
            self.assertTrue(s2.is_defector())
            self.assertEqual(s2.agent_type, "defector")
            self.assertEqual(s2.boldness, 7)
            self.assertEqual(s2.vengefulness, 3)
            self.assertEqual(s2.risk_tolerance, 9)
            self.assertEqual(s2.reputation_concern, 4)
            self.assertEqual(s2.trust_score, 88)
            self.assertEqual(s2.reputation_beliefs, {"Bob Johnson": 42})
            self.assertEqual(s2.violation_history,
                             [{"norm": "no smoking", "when": "2023-02-13"}])
            self.assertEqual(s2.observed_violations,
                             [{"agent": "Derek Nash", "norm": "tipping"}])


class _FakeScratch:
    """Minimal scratch stub for defection_engine tests."""

    def __init__(self, agent_type="defector", boldness=8,
                 reputation_concern=3, trust_score=100, iss="Name: Test"):
        self.agent_type = agent_type
        self.boldness = boldness
        self.reputation_concern = reputation_concern
        self.trust_score = trust_score
        self._iss = iss

    def get_str_iss(self):
        return self._iss


class _FakePersona:
    def __init__(self, scratch):
        self.scratch = scratch


class _FakeNorm:
    def __init__(self, content, poignancy=50, activation_state=True):
        self.content = content
        self.poignancy = poignancy
        self.activation_state = activation_state


class TestDefectionEngine(unittest.TestCase):
    def test_defection_engine_non_defector(self):
        """A citizen persona skips the LLM and short-circuits to comply."""
        from norm import defection_engine

        persona = _FakePersona(_FakeScratch(agent_type="citizen"))
        norm = _FakeNorm("No smoking in the cafe.")

        with patch.object(defection_engine, "llm_call") as mock_call:
            decision, reasoning = defection_engine.calculate_defection_utility(
                persona, norm, {"description": "daily planning"}
            )

        self.assertEqual(decision, "comply")
        self.assertEqual(reasoning, "Not a defector")
        mock_call.assert_not_called()

    def test_defection_engine_parsing(self):
        """Decision lines in LLM output map to the right tuple."""
        from norm import defection_engine

        persona = _FakePersona(_FakeScratch(agent_type="defector"))
        norm = _FakeNorm("No smoking in the cafe.")

        defect_response = (
            "Decision: defect\n"
            "Reasoning: The benefit outweighs a low detection risk.\n"
            "Detection_risk: 3\n"
            "Expected_benefit: 8\n"
        )
        with patch.object(defection_engine, "llm_call", return_value=defect_response):
            decision, reasoning = defection_engine.calculate_defection_utility(
                persona, norm, {"description": "daily planning"}
            )
        self.assertEqual(decision, "defect")
        self.assertIn("benefit", reasoning.lower())

        comply_response = (
            "Decision: comply\n"
            "Reasoning: Too many people are watching right now.\n"
            "Detection_risk: 9\n"
            "Expected_benefit: 2\n"
        )
        with patch.object(defection_engine, "llm_call", return_value=comply_response):
            decision, reasoning = defection_engine.calculate_defection_utility(
                persona, norm, {"description": "daily planning"}
            )
        self.assertEqual(decision, "comply")
        self.assertIn("watching", reasoning.lower())

    def test_defection_engine_parse_failsafe(self):
        """Garbage LLM output falls back to comply."""
        from norm import defection_engine

        persona = _FakePersona(_FakeScratch(agent_type="defector"))
        norm = _FakeNorm("No smoking in the cafe.")

        with patch.object(defection_engine, "llm_call", return_value="garbage"):
            decision, reasoning = defection_engine.calculate_defection_utility(
                persona, norm, {"description": "daily planning"}
            )
        self.assertEqual(decision, "comply")
        self.assertTrue(reasoning)


class _DefectorScratchStub:
    """Scratch stub whose is_defector() returns True for the filtering test."""

    def is_defector(self):
        return True


class _FakeNormDatabase:
    def __init__(self, act_norm):
        self.act_norm = act_norm


class _PersonaWithNorms:
    def __init__(self, norm_database):
        self.scratch = _DefectorScratchStub()
        self.norm_database = norm_database


class TestDefectorNormFiltering(unittest.TestCase):
    def test_defector_norm_filtering(self):
        """Only norms where defection_engine returns 'comply' should end up in
        the curr_act_norms passed to the daily-plan prompt."""
        from norm import norm_compliance
        from norm import defection_engine

        norms = {
            "norm_1": _FakeNorm("Do not smoke in the cafe.", poignancy=90),
            "norm_2": _FakeNorm("Tip the staff after eating.", poignancy=70),
            "norm_3": _FakeNorm("Clean up after yourself.", poignancy=50),
        }
        persona = _PersonaWithNorms(_FakeNormDatabase(norms))

        # norm_1 -> defect, norm_2 and norm_3 -> comply.
        def fake_decide(p, a_norm, ctx, metrics=None):
            if a_norm.content == "Do not smoke in the cafe.":
                return ("defect", "Would rather smoke.")
            return ("comply", "Not worth the risk.")

        captured = {}

        def fake_daily_plan(p, wake_up_hour, curr_act_norms):
            captured["curr_act_norms"] = curr_act_norms
            return [None]

        with patch.object(defection_engine, "calculate_defection_utility",
                          side_effect=fake_decide), \
             patch.object(norm_compliance, "run_gpt_prompt_daily_plan_v2",
                          side_effect=fake_daily_plan):
            norm_compliance.generate_new_daily_plan(persona, wake_up_hour=7)

        curr_act_norms = captured.get("curr_act_norms", "")
        self.assertNotIn("Do not smoke in the cafe.", curr_act_norms)
        self.assertIn("Tip the staff after eating.", curr_act_norms)
        self.assertIn("Clean up after yourself.", curr_act_norms)


class TestCreateDefectorPersonas(unittest.TestCase):
    def test_create_defector_personas(self):
        """End-to-end: run the creation script against a temp copy of the sim."""
        if not os.path.isdir(BASE_SIM_DIR):
            self.skipTest(f"base sim dir not found: {BASE_SIM_DIR}")

        with tempfile.TemporaryDirectory() as tmp:
            sim_copy = os.path.join(tmp, "sim")
            shutil.copytree(BASE_SIM_DIR, sim_copy)

            script = os.path.join(BACKEND, "norm", "create_defector_personas.py")
            result = subprocess.run(
                [sys.executable, script, sim_copy],
                cwd=BACKEND,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                msg=f"script failed:\nstdout={result.stdout}\nstderr={result.stderr}",
            )

            expected = {
                "Marcus Webb": 8,
                "Elena Voss": 6,
                "Derek Nash": 9,
            }
            personas_dir = os.path.join(sim_copy, "personas")
            for name, expected_boldness in expected.items():
                persona_dir = os.path.join(personas_dir, name)
                self.assertTrue(os.path.isdir(persona_dir), f"missing {persona_dir}")

                scratch_path = os.path.join(persona_dir, "bootstrap_memory", "scratch.json")
                self.assertTrue(os.path.isfile(scratch_path))
                with open(scratch_path, "r") as f:
                    scratch = json.load(f)
                self.assertEqual(scratch["agent_type"], "defector")
                self.assertEqual(scratch["identity"], "defector")
                self.assertEqual(scratch["boldness"], expected_boldness)
                self.assertEqual(scratch["trust_score"], 100)
                self.assertEqual(scratch["name"], name)

                norms_dir = os.path.join(persona_dir, "norms")
                self.assertTrue(os.path.isdir(norms_dir))
                db_path = os.path.join(norms_dir, "personal_norm_database.json")
                validity_path = os.path.join(norms_dir, "personal_norm_database_validity.json")
                self.assertTrue(os.path.isfile(db_path))
                self.assertTrue(os.path.isfile(validity_path))
                with open(db_path, "r") as f:
                    self.assertEqual(json.load(f), {})
                with open(validity_path, "r") as f:
                    self.assertEqual(json.load(f), {})

            meta_path = os.path.join(sim_copy, "reverie", "meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for name in expected:
                self.assertIn(name, meta["persona_names"])


class TestReputationSystem(unittest.TestCase):
    def test_init_builds_full_matrix(self):
        from norm.reputation import ReputationSystem

        rs = ReputationSystem(["A", "B", "C"])
        self.assertEqual(set(rs.global_trust.keys()), {"A", "B", "C"})
        for name, row in rs.global_trust.items():
            self.assertNotIn(name, row)
            for other, trust in row.items():
                self.assertEqual(trust, 100)

    def test_update_trust_clamps(self):
        from norm.reputation import ReputationSystem

        rs = ReputationSystem(["A", "B"])
        rs.update_trust("A", "B", -30)
        self.assertEqual(rs.global_trust["A"]["B"], 70)
        rs.update_trust("A", "B", -999)
        self.assertEqual(rs.global_trust["A"]["B"], 0)
        rs.update_trust("A", "B", 999)
        self.assertEqual(rs.global_trust["A"]["B"], 100)

        # self-directed updates are ignored
        rs.update_trust("A", "A", -50)
        self.assertNotIn("A", rs.global_trust["A"])

    def test_spread_gossip_only_trusted(self):
        from norm.reputation import ReputationSystem

        rs = ReputationSystem(["A", "B", "C", "D"])
        # C distrusts A heavily; D still trusts A at 100.
        rs.global_trust["C"]["A"] = 20
        rs.global_trust["D"]["A"] = 100

        before_d = rs.global_trust["D"]["B"]
        before_c = rs.global_trust["C"]["B"]
        rs.spread_gossip("A", "B", severity=6, personas=None)

        # D trusted A -> impact = 6 * 1.0 * 0.5 = 3
        self.assertAlmostEqual(rs.global_trust["D"]["B"], before_d - 3.0, places=4)
        # C did not trust A -> no change
        self.assertEqual(rs.global_trust["C"]["B"], before_c)
        # gossiper and target unaffected
        self.assertNotIn("A", rs.global_trust.get("A", {}))

    def test_snapshot_is_deep_copy(self):
        from norm.reputation import ReputationSystem

        rs = ReputationSystem(["A", "B"])
        snap = rs.get_trust_network_snapshot()
        snap["A"]["B"] = 0
        self.assertEqual(rs.global_trust["A"]["B"], 100)

    def test_save_and_load_roundtrip(self):
        from norm.reputation import ReputationSystem

        rs = ReputationSystem(["A", "B", "C"])
        rs.update_trust("A", "B", -25)
        rs.update_trust("C", "A", -10)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rep.json")
            rs.save(path)

            rs2 = ReputationSystem([])
            rs2.load(path)

        self.assertEqual(rs2.global_trust["A"]["B"], 75)
        self.assertEqual(rs2.global_trust["C"]["A"], 90)
        self.assertEqual(sorted(rs2.persona_names), ["A", "B", "C"])


class _FakeNormForDetect:
    def __init__(self, id_, content, activation_state=True):
        self.id = id_
        self.content = content
        self.activation_state = activation_state


class _FakeNormDatabaseForDetect:
    def __init__(self, act_norm):
        self.act_norm = act_norm


class _FakeScratchForDetect:
    def __init__(self, name):
        self.name = name
        self.reputation_beliefs = {}
        self.observed_violations = []
        self.violation_history = []
        self.norm_conflict = False


class _FakePersonaForDetect:
    def __init__(self, name, act_norm):
        self.scratch = _FakeScratchForDetect(name)
        self.norm_database = _FakeNormDatabaseForDetect(act_norm)


class _FakeEvent:
    def __init__(self, subject, predicate, obj, description=""):
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.description = description or f"{subject} is {predicate} {obj}"


class TestDetectViolations(unittest.TestCase):
    def _persona(self, name="Isabella Rodriguez"):
        norms = {
            "norm_2": _FakeNormForDetect(2, "No smoking is allowed inside the cafe."),
            "norm_4": _FakeNormForDetect(4, "It is customary to tip the cafe staff."),
            "norm_off": _FakeNormForDetect(99, "Inactive norm.", activation_state=False),
        }
        return _FakePersonaForDetect(name, norms)

    def test_skips_self_and_non_persona_subjects(self):
        from norm import violation_detection

        observer = self._persona("Isabella Rodriguez")
        events = [
            # Self event — must be skipped.
            _FakeEvent("Isabella Rodriguez", "walking to", "the cafe"),
            # Ambient event — subject not a persona — must be skipped.
            _FakeEvent("the cafe", "is", "open"),
        ]
        personas = {"Isabella Rodriguez": observer, "Carlos Gomez": object()}

        with patch.object(violation_detection, "run_gpt_prompt_violation_check") as mock_check:
            result = violation_detection.detect_violations(observer, events, personas)

        self.assertEqual(result, [])
        mock_check.assert_not_called()

    def test_returns_violations_and_skips_inactive(self):
        from norm import violation_detection

        observer = self._persona("Isabella Rodriguez")
        event = _FakeEvent("Carlos Gomez", "smoking at", "the cafe counter")
        personas = {"Isabella Rodriguez": observer, "Carlos Gomez": object()}

        def fake_check(event_desc, norm_content, observer_name):
            if "smoking" in norm_content.lower():
                return [{"violation": True, "severity": 8,
                         "certainty": 9, "response": "confront"}]
            return [{"violation": False, "severity": 0,
                     "certainty": 0, "response": "ignore"}]

        with patch.object(violation_detection, "run_gpt_prompt_violation_check",
                          side_effect=fake_check) as mock_check:
            result = violation_detection.detect_violations(observer, [event], personas)

        # Only 2 active norms were checked (inactive one skipped).
        self.assertEqual(mock_check.call_count, 2)
        self.assertEqual(len(result), 1)
        v = result[0]
        self.assertEqual(v["violator"], "Carlos Gomez")
        self.assertEqual(v["severity"], 8)
        self.assertEqual(v["certainty"], 9)
        self.assertEqual(v["response"], "confront")
        self.assertEqual(v["event"], "Carlos Gomez is smoking at the cafe counter")
        self.assertEqual(v["norm"].id, 2)


class TestProcessViolations(unittest.TestCase):
    def _observer(self):
        observer = _FakePersonaForDetect("Isabella Rodriguez", {})
        return observer

    def _violation(self, response, severity=8, certainty=9, violator="Carlos Gomez"):
        norm = _FakeNormForDetect(2, "No smoking is allowed inside the cafe.")
        return {
            "violator": violator,
            "norm": norm,
            "event": f"{violator} is smoking at the cafe",
            "severity": severity,
            "certainty": certainty,
            "response": response,
        }

    def test_confront_sets_norm_conflict_and_updates_trust(self):
        from norm import violation_detection
        from norm.reputation import ReputationSystem

        observer = self._observer()
        rs = ReputationSystem(["Isabella Rodriguez", "Carlos Gomez", "Bob Johnson"])
        personas = {name: object() for name in rs.persona_names}

        violation_detection.process_violations(
            observer, [self._violation("confront", 8, 9)], personas, rs)

        # decay = 8*9/100 = 0.72
        self.assertAlmostEqual(observer.scratch.reputation_beliefs["Carlos Gomez"],
                               100 - 0.72, places=4)
        self.assertAlmostEqual(rs.global_trust["Isabella Rodriguez"]["Carlos Gomez"],
                               100 - 0.72, places=4)
        self.assertTrue(observer.scratch.norm_conflict)
        self.assertEqual(len(observer.scratch.observed_violations), 1)
        self.assertEqual(observer.scratch.observed_violations[0]["response"], "confront")

    def test_gossip_calls_spread_gossip(self):
        from norm import violation_detection

        observer = self._observer()
        rs = MagicMock()
        personas = {"Isabella Rodriguez": object(), "Carlos Gomez": object()}

        violation_detection.process_violations(
            observer, [self._violation("gossip", 6, 5)], personas, rs)

        rs.update_trust.assert_called_once()
        rs.spread_gossip.assert_called_once_with(
            "Isabella Rodriguez", "Carlos Gomez", 6, personas)
        self.assertFalse(observer.scratch.norm_conflict)

    def test_ignore_does_not_confront_or_gossip(self):
        from norm import violation_detection

        observer = self._observer()
        rs = MagicMock()
        personas = {"Isabella Rodriguez": object(), "Carlos Gomez": object()}

        violation_detection.process_violations(
            observer, [self._violation("ignore", 4, 4)], personas, rs)

        rs.update_trust.assert_called_once()
        rs.spread_gossip.assert_not_called()
        self.assertFalse(observer.scratch.norm_conflict)
        # Belief still decayed (4*4/100 = 0.16).
        self.assertAlmostEqual(observer.scratch.reputation_beliefs["Carlos Gomez"],
                               100 - 0.16, places=4)

    def test_belief_floors_at_zero(self):
        from norm import violation_detection

        observer = self._observer()
        observer.scratch.reputation_beliefs["Carlos Gomez"] = 0.1
        rs = MagicMock()
        personas = {"Isabella Rodriguez": object(), "Carlos Gomez": object()}

        violation_detection.process_violations(
            observer, [self._violation("ignore", 10, 10)], personas, rs)

        self.assertEqual(observer.scratch.reputation_beliefs["Carlos Gomez"], 0)


class TestRunGptPromptViolationCheck(unittest.TestCase):
    def test_parses_json_and_returns_dict(self):
        from norm import run_gpt_prompt_norm
        from persona.prompt_template import gpt_structure

        canned = '{"violation": true, "severity": 7, "certainty": 8, "response": "gossip"}'
        # GPT4_safe_generate_response_OLD looks up GPT4_request in its own module.
        with patch.object(gpt_structure, "GPT4_request", return_value=canned):
            result, _ = run_gpt_prompt_norm.run_gpt_prompt_violation_check(
                "Carlos Gomez is smoking at the cafe",
                "No smoking is allowed inside the cafe.",
                "Isabella Rodriguez",
            )

        self.assertEqual(result, {
            "violation": True,
            "severity": 7,
            "certainty": 8,
            "response": "gossip",
        })

    def test_fail_safe_on_bad_output(self):
        from norm import run_gpt_prompt_norm
        from persona.prompt_template import gpt_structure

        with patch.object(gpt_structure, "GPT4_request", return_value="not json"):
            result, _ = run_gpt_prompt_norm.run_gpt_prompt_violation_check(
                "Carlos Gomez is smoking at the cafe",
                "No smoking is allowed inside the cafe.",
                "Isabella Rodriguez",
            )

        self.assertEqual(result, {
            "violation": False,
            "severity": 0,
            "certainty": 0,
            "response": "ignore",
        })


class _NS:
    """Bare attribute container (MagicMock would auto-create `metrics`)."""


def _adoption_persona(metrics=None):
    p = _NS()
    p.scratch = _NS()
    p.scratch.name = "Carlos Gomez"
    p.scratch.identity = "citizen"
    p.scratch.curr_time = "2026-07-08 09:00"
    p.scratch.norm_evaluate = True
    p.scratch.norm_evaluate_trigger_curr = 100
    p.scratch.norm_evaluate_trigger_max = 100
    seed = _NS()
    seed.id = "1"
    seed.poignancy = -1
    seed.content = "No smoking indoors."
    seed.activation_state = False
    seed.validity_state = False
    p.norm_database = _NS()
    p.norm_database.norm_seed = {"1": seed}
    p.norm_database.added = []
    p.norm_database.add_act_norm = p.norm_database.added.append
    if metrics is not None:
        p.metrics = metrics
    return p, seed


def _accepted_norm():
    n = _NS()
    n.content = "No smoking indoors."
    n.poignancy = 50
    n.activation_state = False
    n.validity_state = False
    return n


class TestNormAdoptionMetricsHook(unittest.TestCase):
    """Change D: log_norm_adoption fires at the adoption decision site,
    observation-only (a metrics failure must never touch the sim path)."""

    def _run(self, persona, save_tag, new_norm=None):
        from norm import norm_evaluate
        result = (save_tag, new_norm if new_norm is not None else _accepted_norm())
        with patch.object(norm_evaluate, "norm_evaluate_check",
                          lambda *a, **k: result):
            norm_evaluate.norms_evaluate(persona, {})

    def test_accepted_norm_logged(self):
        class FakeMetrics:
            def __init__(self):
                self.calls = []

            def log_norm_adoption(self, **kw):
                self.calls.append(kw)

        metrics = FakeMetrics()
        persona, seed = _adoption_persona(metrics)
        self._run(persona, True)
        self.assertEqual(len(metrics.calls), 1)
        call = metrics.calls[0]
        self.assertTrue(call["accepted"])
        self.assertEqual(call["agent_name"], "Carlos Gomez")
        self.assertEqual(call["agent_identity"], "citizen")
        self.assertEqual(call["norm_content"], "No smoking indoors.")
        # the norm was actually adopted
        self.assertEqual(len(persona.norm_database.added), 1)

    def test_rejected_norm_logged(self):
        class FakeMetrics:
            def __init__(self):
                self.calls = []

            def log_norm_adoption(self, **kw):
                self.calls.append(kw)

        metrics = FakeMetrics()
        persona, seed = _adoption_persona(metrics)
        seed.poignancy = -1
        seed.reject_stage = "type_check_not_norm"
        self._run(persona, False)
        self.assertEqual(len(metrics.calls), 1)
        call = metrics.calls[0]
        self.assertFalse(call["accepted"])
        # rejections never carry sentinel scores; the stage says why
        self.assertIsNone(call["utility_score"])
        self.assertEqual(call["reject_stage"], "type_check_not_norm")
        self.assertEqual(persona.norm_database.added, [])

    def test_raising_collector_does_not_break_adoption(self):
        class BoomMetrics:
            def log_norm_adoption(self, **kw):
                raise RuntimeError("boom")

        persona, seed = _adoption_persona(BoomMetrics())
        self._run(persona, True)  # must not raise
        self.assertEqual(len(persona.norm_database.added), 1)

    def test_missing_metrics_attribute_is_fine(self):
        persona, seed = _adoption_persona(metrics=None)
        self.assertFalse(hasattr(persona, "metrics"))
        self._run(persona, True)  # must not raise
        self.assertEqual(len(persona.norm_database.added), 1)

    def test_deferral_not_logged_not_adopted(self):
        """save_tag=None (parse failure) emits NO adoption event and does not
        adopt; the seed stays pending for the next trigger."""
        class FakeMetrics:
            def __init__(self):
                self.calls = []

            def log_norm_adoption(self, **kw):
                self.calls.append(kw)

        metrics = FakeMetrics()
        persona, seed = _adoption_persona(metrics)
        self._run(persona, None)
        self.assertEqual(metrics.calls, [])
        self.assertEqual(persona.norm_database.added, [])
        self.assertEqual(seed.poignancy, -1)  # still pending


class TestEvalDeferral(unittest.TestCase):
    """Pass 3 Change B: a parse/LLM failure at any evaluation stage defers
    (save_tag None, seed pending) instead of rejecting; parsed verdicts keep
    their exact previous behavior."""

    def _check(self, **overrides):
        from contextlib import ExitStack

        from norm import norm_evaluate
        from norm.normNode import NormNode

        stages = dict(
            generate_norm_fact_consistency_check=lambda n: (True, ''),
            generate_norm_duplicate_check=lambda n, p: False,
            generate_seeds_type_check_v2=lambda n: [True, "injunctive"],
            generate_recognize_conflict_check=lambda n, p: False,
            generate_normal_norm_utility=lambda n, p: [50, "useful"],
        )
        stages.update(overrides)
        norm = NormNode(1, "injunctive", "No smoking indoors.", "everyone",
                        "should not smoke", "indoors")
        with ExitStack() as stack:
            for name, fake in stages.items():
                stack.enter_context(patch.object(norm_evaluate, name, fake))
            save_tag, new_norm = norm_evaluate.norm_evaluate_check(
                norm, _NS(), {})
        return save_tag, new_norm, norm

    def test_all_parsed_adopts_identically(self):
        save_tag, new_norm, norm = self._check()
        self.assertIs(save_tag, True)
        self.assertEqual(new_norm.poignancy, 50)
        self.assertEqual(norm.poignancy, 50)
        self.assertIsNone(norm.reject_stage)

    def test_parsed_rejections_carry_reject_stage(self):
        save_tag, _, norm = self._check(
            generate_norm_duplicate_check=lambda n, p: True)
        self.assertEqual(norm.reject_stage, "duplicate_check")

        save_tag, _, norm = self._check(
            generate_norm_fact_consistency_check=lambda n: (False, ''))
        self.assertEqual(norm.reject_stage, "fact_consistency")

        save_tag, _, norm = self._check(
            generate_seeds_type_check_v2=lambda n: [False])
        self.assertEqual(norm.reject_stage, "type_check_not_norm")

        save_tag, _, norm = self._check(
            generate_recognize_conflict_check=lambda n, p: True)
        self.assertEqual(norm.reject_stage, "conflict_check")

    def test_type_check_failure_defers(self):
        save_tag, _, norm = self._check(
            generate_seeds_type_check_v2=lambda n: None)
        self.assertIsNone(save_tag)
        self.assertEqual(norm.poignancy, -1)  # pending, not rejected

    def test_type_check_parsed_no_still_rejects(self):
        save_tag, _, norm = self._check(
            generate_seeds_type_check_v2=lambda n: [False])
        self.assertIs(save_tag, False)
        self.assertEqual(norm.poignancy, -1)  # same sentinel as before

    def test_duplicate_failure_defers_but_verdict_rejects(self):
        save_tag, _, norm = self._check(
            generate_norm_duplicate_check=lambda n, p: None)
        self.assertIsNone(save_tag)
        self.assertEqual(norm.poignancy, -1)

        save_tag, _, norm = self._check(
            generate_norm_duplicate_check=lambda n, p: True)
        self.assertIs(save_tag, False)
        self.assertEqual(norm.poignancy, -3)

    def test_fact_consistency_failure_defers_but_verdict_rejects(self):
        save_tag, _, norm = self._check(
            generate_norm_fact_consistency_check=lambda n: (None, ''))
        self.assertIsNone(save_tag)
        self.assertEqual(norm.poignancy, -1)

        save_tag, _, norm = self._check(
            generate_norm_fact_consistency_check=lambda n: (False, ''))
        self.assertIs(save_tag, False)
        self.assertEqual(norm.poignancy, -2)

    def test_conflict_failure_defers_but_verdict_rejects(self):
        save_tag, _, norm = self._check(
            generate_recognize_conflict_check=lambda n, p: None)
        self.assertIsNone(save_tag)

        save_tag, _, norm = self._check(
            generate_recognize_conflict_check=lambda n, p: True)
        self.assertIs(save_tag, False)

    def test_utility_failsafe_defers_instead_of_fake_adopt(self):
        """[4, "fail_safe"] previously ADOPTED the seed with utility 4."""
        save_tag, _, norm = self._check(
            generate_normal_norm_utility=lambda n, p: [4, "fail_safe"])
        self.assertIsNone(save_tag)
        self.assertEqual(norm.poignancy, -1)


class TestTransactionalSynthesis(unittest.TestCase):
    """Pass 3 Change C: long-term synthesis deactivates a replaced norm only
    AFTER its verified replacement is in the database; a failed replacement
    leaves the originals active."""

    def _run(self, save_tag):
        from contextlib import ExitStack

        from norm import norm_evaluate
        from norm.normNode import NormNode

        events = []
        counters = []
        persona = _NS()
        persona.scratch = _NS()
        persona.scratch.norm_evaluate_trigger_curr = 100
        persona.norm_database = _NS()
        persona.norm_database.add_norm_seed = \
            lambda n: events.append(("seed", n))
        persona.norm_database.add_act_norm = \
            lambda n: events.append(("add_act", n))

        node = NormNode(1, "injunctive", "Keep noise low.", "everyone",
                        "keeps", "noise low")
        replacement = NormNode(2, "injunctive", "Keep noise low.", "everyone",
                               "keeps", "noise low", poi=60)

        stages = dict(
            generate_active_norms_classfication=lambda p: ("CLS", True),
            norm_long_term_synthesis_check=lambda s: ("CHK", True, ["group-1"]),
            generate_norm_long_term_synthesis=lambda s: ["synth-norm-1"],
            generate_format_norm=lambda s, d: node,
            norm_evaluate_check=(
                lambda n, p, ps, long_term_tag=False: (save_tag, replacement)),
            specific_norm_deactive=lambda p, d: events.append(("deactivate", d)),
        )
        with ExitStack() as stack:
            for name, fake in stages.items():
                stack.enter_context(patch.object(norm_evaluate, name, fake))
            stack.enter_context(patch.object(
                norm_evaluate.call_profiler, "incr",
                lambda name, n=1: counters.append(name)))
            norm_evaluate.run_long_term_norm_evaluate(persona, {})
        return events, counters, persona, replacement

    def test_success_swaps_deactivate_last(self):
        events, counters, persona, replacement = self._run(True)
        kinds = [e[0] for e in events]
        self.assertEqual(kinds, ["seed", "add_act", "deactivate"])
        self.assertLess(kinds.index("add_act"), kinds.index("deactivate"))
        self.assertTrue(replacement.activation_state)
        self.assertTrue(replacement.validity_state)
        self.assertEqual(persona.scratch.norm_evaluate_trigger_curr, 40)
        self.assertNotIn("synthesis_aborted", counters)

    def test_rejected_replacement_leaves_originals_active(self):
        events, counters, persona, _ = self._run(False)
        kinds = [e[0] for e in events]
        self.assertNotIn("add_act", kinds)
        self.assertNotIn("deactivate", kinds)
        self.assertEqual(persona.scratch.norm_evaluate_trigger_curr, 100)
        self.assertIn("synthesis_aborted", counters)

    def test_deferred_replacement_leaves_originals_active(self):
        events, counters, persona, _ = self._run(None)
        kinds = [e[0] for e in events]
        self.assertNotIn("add_act", kinds)
        self.assertNotIn("deactivate", kinds)
        self.assertIn("synthesis_aborted", counters)


class TestNormLoadMismatch(unittest.TestCase):
    """Pass 4 Change B: the loader must never silently zero a persona whose
    scratch claims norms; a short file must partially load, not crash."""

    def _entry(self, i):
        return {"ID": i, "type": "injunctive", "content": f"Norm {i}.",
                "subject": "everyone", "predicate": "follows",
                "object": f"rule {i}", "utility": 50,
                "activation_state": True, "validity_state": True}

    def _base(self, tmp, seed_entries, act_entries):
        d = os.path.join(tmp, "norms")
        os.makedirs(d, exist_ok=True)
        if seed_entries is not None:
            with open(os.path.join(d, "personal_norm_database.json"), "w") as f:
                json.dump({f"norm_{i}": self._entry(i)
                           for i in range(1, seed_entries + 1)}, f)
        if act_entries is not None:
            with open(os.path.join(d, "personal_norm_database_validity.json"),
                      "w") as f:
                json.dump({f"norm_{i}": self._entry(i)
                           for i in range(1, act_entries + 1)}, f)
        return d

    def _load(self, d, expected):
        from norm import normDatabase
        counters = []
        with patch.object(normDatabase.call_profiler, "incr",
                          lambda name, n=1: counters.append(name)):
            db = normDatabase.NormDatabase(d, expected, expected, None)
        return db, counters

    def test_missing_files_with_expected_counts_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._base(tmp, None, None)
            db, counters = self._load(d, 5)
        self.assertEqual(db.norm_count, 0)
        self.assertEqual(counters.count("norm_load_mismatch"), 2)

    def test_short_file_partially_loads_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._base(tmp, 3, 3)
            db, counters = self._load(d, 5)  # scratch claims 5, files have 3
        self.assertEqual(db.norm_count, 3)
        self.assertEqual(db.act_norm_count, 3)
        self.assertEqual(counters.count("norm_load_mismatch"), 2)

    def test_consistent_base_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._base(tmp, 5, 5)
            db, counters = self._load(d, 5)
        self.assertEqual(db.norm_count, 5)
        self.assertEqual(db.act_norm_count, 5)
        self.assertNotIn("norm_load_mismatch", counters)

    def test_zero_expected_missing_files_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._base(tmp, None, None)
            db, counters = self._load(d, 0)
        self.assertNotIn("norm_load_mismatch", counters)

    def test_strict_mode_raises(self):
        from norm import normDatabase
        with tempfile.TemporaryDirectory() as tmp:
            d = self._base(tmp, None, None)
            with patch.dict(os.environ, {"CRSEC_STRICT_NORM_LOAD": "1"}):
                with self.assertRaises(RuntimeError):
                    normDatabase.NormDatabase(d, 5, 5, None)


class TestViolationContentLivePath(unittest.TestCase):
    """Pass 4 Change D: norm content must flow from a REAL NormNode through
    detect_violations -> process_violations into the metrics logs and the
    observer's observed_violations. Regression test for the reported (not
    locally reproducible — calib_009 logs are fully populated) norm_content=
    None symptom: if any refactor drops the attribute, this fails."""

    def test_content_reaches_metrics_and_scratch(self):
        from norm import violation_detection
        from norm.normNode import NormNode

        norm = NormNode(2, "injunctive", "No smoking is allowed inside the cafe.",
                        "no one", "is allowed", "to smoke inside the cafe",
                        poi=100, activation_state=True, validity_state=True)
        observer = _FakePersonaForDetect("Isabella Rodriguez", {"norm_2": norm})

        class FakeMetrics:
            def __init__(self):
                self.violations = []
                self.enforcements = []

            def log_violation(self, **kw):
                self.violations.append(kw)

            def log_enforcement(self, **kw):
                self.enforcements.append(kw)

        metrics = FakeMetrics()
        event = _FakeEvent("Carlos Gomez", "smoking at", "the cafe counter")
        personas = {"Isabella Rodriguez": observer, "Carlos Gomez": object()}

        def fake_check(event_desc, norm_content, observer_name):
            return [{"violation": True, "severity": 7, "certainty": 9,
                     "response": "confront"}]

        with patch.object(violation_detection, "run_gpt_prompt_violation_check",
                          side_effect=fake_check):
            violations = violation_detection.detect_violations(
                observer, [event], personas)
        self.assertEqual(len(violations), 1)
        self.assertIs(violations[0]["norm"], norm)

        violation_detection.process_violations(
            observer, violations, personas, None, metrics)

        self.assertEqual(len(metrics.violations), 1)
        self.assertEqual(metrics.violations[0]["norm_content"],
                         "No smoking is allowed inside the cafe.")
        self.assertEqual(len(metrics.enforcements), 1)
        self.assertEqual(metrics.enforcements[0]["norm_content"],
                         "No smoking is allowed inside the cafe.")
        ov = observer.scratch.observed_violations
        self.assertEqual(len(ov), 1)
        self.assertEqual(ov[0]["norm_content"],
                         "No smoking is allowed inside the cafe.")
        self.assertEqual(ov[0]["norm_id"], 2)


if __name__ == "__main__":
    unittest.main()
