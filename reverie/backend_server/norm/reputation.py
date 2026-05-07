"""
File: reputation.py
Description: Global trust matrix + gossip propagation for the norm detection
and reputation system.
"""
import copy
import json


class ReputationSystem:
    def __init__(self, persona_names_list):
        self.persona_names = list(persona_names_list)
        self.global_trust = {
            name: {other: 100 for other in self.persona_names if other != name}
            for name in self.persona_names
        }

    def _clamp(self, value):
        if value < 0:
            return 0
        if value > 100:
            return 100
        return value

    def update_trust(self, observer, target, delta):
        if observer == target:
            return
        if observer not in self.global_trust:
            self.global_trust[observer] = {}
        if target not in self.global_trust[observer]:
            self.global_trust[observer][target] = 100
        new_val = self.global_trust[observer][target] + delta
        self.global_trust[observer][target] = self._clamp(new_val)

    def spread_gossip(self, gossiper, target, severity, personas):
        """
        For each other agent, if they trust the gossiper (their belief about
        gossiper > 40), apply severity * (gossiper_trust / 100) * 0.5 as a
        negative delta to target's trust in their eyes.
        """
        if personas is None:
            listener_names = list(self.global_trust.keys())
        else:
            listener_names = list(personas.keys()) if hasattr(personas, "keys") else list(personas)

        for listener in listener_names:
            if listener == gossiper or listener == target:
                continue
            gossiper_trust = self.global_trust.get(listener, {}).get(gossiper, 100)
            if gossiper_trust <= 40:
                continue
            impact = severity * (gossiper_trust / 100.0) * 0.5
            self.update_trust(listener, target, -impact)

    def get_trust_network_snapshot(self):
        return copy.deepcopy(self.global_trust)

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump({
                "persona_names": self.persona_names,
                "global_trust": self.global_trust,
            }, f, indent=2)

    def load(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.persona_names = data.get("persona_names", list(data.get("global_trust", {}).keys()))
        self.global_trust = data.get("global_trust", {})
