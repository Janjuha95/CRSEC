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
        def fake_decide(p, a_norm, ctx):
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


if __name__ == "__main__":
    unittest.main()
