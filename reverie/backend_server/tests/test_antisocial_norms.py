"""
Tests for the Antisocial Norm Entrepreneurs system (Experiment 2 / RQ2).

Run from reverie/backend_server/:
    python -m pytest tests/test_antisocial_norms.py -v
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


_ORIGINAL_CWD = None


def setUpModule():
    # creation.Create() reads prompt files via paths relative to backend_server.
    # Pin cwd so the test passes regardless of how it was launched.
    global _ORIGINAL_CWD
    _ORIGINAL_CWD = os.getcwd()
    os.chdir(BACKEND)


def tearDownModule():
    if _ORIGINAL_CWD is not None:
        os.chdir(_ORIGINAL_CWD)


ANTISOCIAL_SYS = os.path.join(BACKEND, "norm", "creation_prompt", "sys_prompt_antisocial.txt")
ANTISOCIAL_USR = os.path.join(BACKEND, "norm", "creation_prompt", "usr_prompt_antisocial_v1.txt")
PROSOCIAL_SYS = os.path.join(BACKEND, "norm", "creation_prompt", "sys_prompt.txt")
PROSOCIAL_USR = os.path.join(BACKEND, "norm", "creation_prompt", "usr_prompt_v6.txt")


class TestAntisocialPromptFiles(unittest.TestCase):
    def test_antisocial_sys_prompt_exists_and_has_marker(self):
        self.assertTrue(os.path.isfile(ANTISOCIAL_SYS))
        with open(ANTISOCIAL_SYS, "rt", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("cynical social observer", content.lower())
        self.assertIn("personal freedom", content.lower())

    def test_antisocial_usr_prompt_exists_and_has_marker(self):
        self.assertTrue(os.path.isfile(ANTISOCIAL_USR))
        with open(ANTISOCIAL_USR, "rt", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("individual freedom and personal autonomy", content)
        # Same JSON schema fields as usr_prompt_v6.txt.
        for field in ("ID", "type", "content", "subject", "predicate",
                      "object", "utility", "activation_state", "validity_state"):
            self.assertIn(field, content)
        self.assertIn("ATTENTION: Do not output anything else except for the content in JSON.",
                      content)

    def test_prosocial_prompt_files_still_present(self):
        # Sanity: we did not remove the originals.
        self.assertTrue(os.path.isfile(PROSOCIAL_SYS))
        self.assertTrue(os.path.isfile(PROSOCIAL_USR))


class TestIsAntisocialNorm(unittest.TestCase):
    def test_positive_matches(self):
        from norm.metrics import is_antisocial_norm

        positives = [
            "Everyone should mind their own business in the cafe.",
            "Rule-breaking is a personal choice.",
            "People should not interfere with others.",
            "Live and let live, even in public spaces.",
            "No one should tell others how to behave.",
        ]
        for content in positives:
            self.assertTrue(is_antisocial_norm(content),
                            msg=f"expected antisocial: {content!r}")

    def test_negative_matches(self):
        from norm.metrics import is_antisocial_norm

        negatives = [
            "Everyone is not allowed to smoke indoors.",
            "Tip the staff after eating.",
            "Clean up after yourself.",
            "Drive on the right side of the road.",
        ]
        for content in negatives:
            self.assertFalse(is_antisocial_norm(content),
                             msg=f"expected prosocial: {content!r}")

    def test_case_insensitive(self):
        from norm.metrics import is_antisocial_norm

        self.assertTrue(is_antisocial_norm("MIND YOUR OWN BUSINESS"))
        self.assertTrue(is_antisocial_norm("Personal Freedom matters."))


class _FakeNormForMetrics:
    def __init__(self, content, activation_state=True):
        self.content = content
        self.activation_state = activation_state


class _FakeNormDatabaseForMetrics:
    def __init__(self, act_norm):
        self.act_norm = act_norm


class _FakePersonaForMetrics:
    def __init__(self, act_norm):
        self.norm_database = _FakeNormDatabaseForMetrics(act_norm)


class TestClassifyNorms(unittest.TestCase):
    def test_counts_prosocial_and_antisocial(self):
        from norm.metrics import classify_norms

        persona = _FakePersonaForMetrics({
            "norm_1": _FakeNormForMetrics("Everyone should mind their own business."),
            "norm_2": _FakeNormForMetrics("Rule-breaking is a personal choice."),
            "norm_3": _FakeNormForMetrics("Tip the staff after eating."),
            "norm_4": _FakeNormForMetrics("No smoking is allowed in the cafe."),
        })
        self.assertEqual(classify_norms(persona), {"prosocial": 2, "antisocial": 2})

    def test_ignores_inactive_norms(self):
        from norm.metrics import classify_norms

        persona = _FakePersonaForMetrics({
            "norm_1": _FakeNormForMetrics("Everyone should mind their own business.",
                                          activation_state=True),
            "norm_2": _FakeNormForMetrics("Live and let live.",
                                          activation_state=False),
            "norm_3": _FakeNormForMetrics("Clean up after yourself.",
                                          activation_state=True),
            "norm_4": _FakeNormForMetrics("Tip the staff.",
                                          activation_state=False),
        })
        # norm_2 (antisocial) and norm_4 (prosocial) are inactive and ignored.
        self.assertEqual(classify_norms(persona), {"prosocial": 1, "antisocial": 1})

    def test_empty_database(self):
        from norm.metrics import classify_norms

        persona = _FakePersonaForMetrics({})
        self.assertEqual(classify_norms(persona), {"prosocial": 0, "antisocial": 0})


def _setup_fake_sim(tmp, name, sim_code="test_sim"):
    """Build the minimal fs_storage layout that Create() reads/writes."""
    persona_dir = os.path.join(tmp, sim_code, "personas", name)
    os.makedirs(os.path.join(persona_dir, "bootstrap_memory"))
    os.makedirs(os.path.join(persona_dir, "norms"))
    with open(os.path.join(persona_dir, "bootstrap_memory", "scratch.json"), "w") as f:
        json.dump({"name": name}, f)
    return sim_code


_FAKE_NORM_JSON = json.dumps({
    "norm_1": {
        "ID": 1, "type": "injunctive", "content": "x",
        "subject": "a", "predicate": "b", "object": "c",
        "utility": 50, "activation_state": True, "validity_state": True,
    }
})


class TestCreateIdentityRouting(unittest.TestCase):
    """Create(rs) must pick antisocial prompts for defectors, prosocial otherwise."""

    def _run_create(self, identity, name="Marcus Webb"):
        from norm import creation

        captured = {"sys_prompts": [], "usr_prompts": []}

        class _SpyCreation:
            def __init__(self, sys_prompt):
                captured["sys_prompts"].append(sys_prompt)

            def creation(self, prompt, agent_description):
                captured["usr_prompts"].append(prompt)
                return _FAKE_NORM_JSON

        scratch = MagicMock()
        scratch.identity = identity
        scratch.norm_count = 0
        scratch.act_norm_count = 0
        persona = MagicMock()
        persona.scratch = scratch

        rs = MagicMock()
        rs.personas = {name: persona}

        with tempfile.TemporaryDirectory() as tmp:
            sim_code = _setup_fake_sim(tmp, name)
            rs.sim_code = sim_code

            with patch.object(creation, "fs_storage", tmp), \
                 patch.object(creation, "Creation", _SpyCreation), \
                 patch.object(creation, "NormDatabase", MagicMock()), \
                 patch("builtins.input", side_effect=["y", name, "n"]):
                creation.Create(rs)

        self.assertEqual(len(captured["sys_prompts"]), 1)
        self.assertEqual(len(captured["usr_prompts"]), 1)
        return captured["sys_prompts"][0], captured["usr_prompts"][0]

    def test_defector_uses_antisocial_prompts(self):
        sys_prompt, usr_prompt = self._run_create("defector")
        self.assertIn("cynical social observer", sys_prompt.lower())
        self.assertIn("individual freedom and personal autonomy", usr_prompt)

    def test_citizen_uses_prosocial_prompts(self):
        sys_prompt, usr_prompt = self._run_create("citizen")
        self.assertIn("experienced social scientist", sys_prompt.lower())
        self.assertIn("socially accepted norms in a Cafe", usr_prompt)

    def test_entrepreneur_uses_prosocial_prompts(self):
        sys_prompt, usr_prompt = self._run_create("entrepreneur")
        self.assertIn("experienced social scientist", sys_prompt.lower())
        self.assertIn("socially accepted norms in a Cafe", usr_prompt)

    def test_unknown_entrepreneur_name_does_not_call_llm(self):
        """Names not in rs.personas must not trigger any prompt reads or LLM calls."""
        from norm import creation

        called = {"n": 0}

        class _SpyCreation:
            def __init__(self, sys_prompt):
                called["n"] += 1

            def creation(self, prompt, agent_description):
                called["n"] += 1
                return _FAKE_NORM_JSON

        rs = MagicMock()
        rs.personas = {}  # no one matches

        with patch.object(creation, "Creation", _SpyCreation), \
             patch.object(creation, "NormDatabase", MagicMock()), \
             patch("builtins.input", side_effect=["y", "Ghost Agent", "n"]):
            creation.Create(rs)

        self.assertEqual(called["n"], 0)


if __name__ == "__main__":
    unittest.main()
