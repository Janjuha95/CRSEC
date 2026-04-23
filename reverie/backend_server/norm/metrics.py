import json
import os
import time


class MetricsCollector:
    """
    Central metrics collection for thesis experiments.
    Tracks 9 metric categories across the simulation lifecycle.

    Usage:
        metrics = MetricsCollector("experiment_1_baseline")
        metrics.log_defection_attempt(...)   # called from defection_engine.py
        metrics.log_violation(...)           # called from violation_detection.py
        metrics.log_enforcement(...)         # called from violation_detection.py
        metrics.snapshot(personas, rep_sys, step)  # called periodically from reverie.py
        metrics.save_all()                   # called from reverie.py save()
    """

    def __init__(self, sim_code, log_dir=None):
        self.sim_code = sim_code
        self.log_dir = log_dir or f"metrics/{sim_code}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.start_time = time.time()

        # Event logs (appended throughout simulation)
        self.defection_log = []       # Every defection decision by defector agents
        self.violation_log = []       # Every detected violation
        self.enforcement_log = []     # Every enforcement action (confront/gossip/ignore)
        self.norm_adoption_log = []   # Every time an agent accepts/rejects a new norm
        self.conversation_log = []    # Norm-related conversations

        # Periodic snapshots (taken every N steps)
        self.norm_snapshots = []      # Full norm database state per agent
        self.trust_snapshots = []     # Full trust network matrix
        self.compliance_timeline = [] # Compliance rate over time
        self.population_stats = []    # Aggregate stats per step

    # ===================================================
    # EVENT LOGGING METHODS (called from other modules)
    # ===================================================

    def log_defection_attempt(self, agent_name, agent_identity, norm_content,
                              decision, reasoning, boldness, trust_score, step):
        """Log a defector agent's comply/defect decision."""
        self.defection_log.append({
            "step": step,
            "agent": agent_name,
            "identity": agent_identity,
            "norm": norm_content,
            "decision": decision,  # "comply" or "defect"
            "reasoning": reasoning,
            "boldness": boldness,
            "trust_score": trust_score,
            "timestamp": time.time()
        })

    def log_violation(self, observer, violator, norm_content, severity,
                      certainty, step):
        """Log a detected violation."""
        self.violation_log.append({
            "step": step,
            "observer": observer,
            "violator": violator,
            "norm": norm_content,
            "severity": severity,
            "certainty": certainty,
            "timestamp": time.time()
        })

    def log_enforcement(self, enforcer, target, action_type, norm_content, step):
        """Log an enforcement action. action_type: 'confront', 'gossip', 'ignore'"""
        self.enforcement_log.append({
            "step": step,
            "enforcer": enforcer,
            "target": target,
            "action_type": action_type,
            "norm": norm_content,
            "timestamp": time.time()
        })

    def log_norm_adoption(self, agent_name, norm_content, accepted,
                          utility_score, agent_identity, step):
        """Log when an agent evaluates and accepts/rejects a norm."""
        self.norm_adoption_log.append({
            "step": step,
            "agent": agent_name,
            "identity": agent_identity,
            "norm": norm_content,
            "accepted": accepted,
            "utility_score": utility_score,
            "is_antisocial": is_antisocial_norm(norm_content),
            "timestamp": time.time()
        })

    # ===================================================
    # PERIODIC SNAPSHOT METHODS (called from reverie.py)
    # ===================================================

    def snapshot(self, personas, reputation_system, step):
        """Take a full snapshot of the simulation state. Call every N steps."""
        self._snapshot_norms(personas, step)
        self._snapshot_trust(reputation_system, step)
        self._snapshot_compliance(personas, step)
        self._snapshot_population_stats(personas, step)

    def _snapshot_norms(self, personas, step):
        """Capture every agent's norm database state."""
        snapshot = {"step": step, "agents": {}}
        for name, persona in personas.items():
            agent_norms = []
            for nid, norm in persona.norm_database.act_norm.items():
                agent_norms.append({
                    "id": nid,
                    "content": norm.content,
                    "type": norm.type,
                    "utility": norm.poignancy,
                    "active": norm.activation_state,
                    "valid": norm.validity_state,
                    "is_antisocial": is_antisocial_norm(norm.content)
                })
            snapshot["agents"][name] = {
                "identity": persona.scratch.identity,
                "norm_count": len(agent_norms),
                "norms": agent_norms
            }
        self.norm_snapshots.append(snapshot)

    def _snapshot_trust(self, reputation_system, step):
        """Capture the full trust network matrix."""
        if reputation_system:
            self.trust_snapshots.append({
                "step": step,
                "network": reputation_system.get_trust_network_snapshot()
            })

    def _snapshot_compliance(self, personas, step):
        """Calculate current norm compliance rate across all agents."""
        total_active = 0
        total_valid = 0
        prosocial_active = 0
        prosocial_valid = 0
        antisocial_active = 0
        antisocial_valid = 0

        for name, persona in personas.items():
            for nid, norm in persona.norm_database.act_norm.items():
                if norm.activation_state:
                    total_active += 1
                    is_anti = is_antisocial_norm(norm.content)
                    if is_anti:
                        antisocial_active += 1
                    else:
                        prosocial_active += 1
                    if norm.validity_state:
                        total_valid += 1
                        if is_anti:
                            antisocial_valid += 1
                        else:
                            prosocial_valid += 1

        self.compliance_timeline.append({
            "step": step,
            "overall_rate": total_valid / total_active if total_active > 0 else 0,
            "prosocial_rate": prosocial_valid / prosocial_active if prosocial_active > 0 else 0,
            "antisocial_rate": antisocial_valid / antisocial_active if antisocial_active > 0 else 0,
            "total_active_norms": total_active,
            "prosocial_active": prosocial_active,
            "antisocial_active": antisocial_active
        })

    def _snapshot_population_stats(self, personas, step):
        """Aggregate population-level statistics."""
        defector_count = 0
        citizen_count = 0
        entrepreneur_count = 0
        trust_scores = []
        total_violations_observed = 0

        for name, persona in personas.items():
            identity = persona.scratch.identity
            if identity == "defector":
                defector_count += 1
            elif identity == "entrepreneur":
                entrepreneur_count += 1
            else:
                citizen_count += 1

            trust_scores.append(persona.scratch.trust_score)
            total_violations_observed += len(persona.scratch.observed_violations)

        # Defection rate from logs for this period
        recent_defections = [d for d in self.defection_log if d["step"] == step]
        defect_decisions = sum(1 for d in recent_defections if d["decision"] == "defect")
        comply_decisions = sum(1 for d in recent_defections if d["decision"] == "comply")

        self.population_stats.append({
            "step": step,
            "defector_count": defector_count,
            "citizen_count": citizen_count,
            "entrepreneur_count": entrepreneur_count,
            "avg_trust_score": sum(trust_scores) / len(trust_scores) if trust_scores else 0,
            "min_trust_score": min(trust_scores) if trust_scores else 0,
            "max_trust_score": max(trust_scores) if trust_scores else 0,
            "total_violations_observed": total_violations_observed,
            "defect_decisions_this_step": defect_decisions,
            "comply_decisions_this_step": comply_decisions,
            "total_defection_log_size": len(self.defection_log),
            "total_violation_log_size": len(self.violation_log),
            "total_enforcement_log_size": len(self.enforcement_log)
        })

    # ===================================================
    # SAVE / LOAD
    # ===================================================

    def save_all(self):
        """Save all metrics to disk as separate JSON files."""
        datasets = {
            "defection_log": self.defection_log,
            "violation_log": self.violation_log,
            "enforcement_log": self.enforcement_log,
            "norm_adoption_log": self.norm_adoption_log,
            "conversation_log": self.conversation_log,
            "norm_snapshots": self.norm_snapshots,
            "trust_snapshots": self.trust_snapshots,
            "compliance_timeline": self.compliance_timeline,
            "population_stats": self.population_stats,
        }
        # default=str lets datetime values (event-log "step" fields passed as
        # persona.scratch.curr_time) serialize as ISO strings.
        for name, data in datasets.items():
            filepath = os.path.join(self.log_dir, f"{name}.json")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

        summary = {
            "sim_code": self.sim_code,
            "total_steps_logged": self.population_stats[-1]["step"] if self.population_stats else 0,
            "total_defection_attempts": len(self.defection_log),
            "total_defections": sum(1 for d in self.defection_log if d["decision"] == "defect"),
            "total_compliances": sum(1 for d in self.defection_log if d["decision"] == "comply"),
            "total_violations_detected": len(self.violation_log),
            "total_confrontations": sum(1 for e in self.enforcement_log if e["action_type"] == "confront"),
            "total_gossip_events": sum(1 for e in self.enforcement_log if e["action_type"] == "gossip"),
            "total_ignores": sum(1 for e in self.enforcement_log if e["action_type"] == "ignore"),
            "total_norms_adopted": sum(1 for n in self.norm_adoption_log if n["accepted"]),
            "total_norms_rejected": sum(1 for n in self.norm_adoption_log if not n["accepted"]),
            "antisocial_norms_adopted": sum(1 for n in self.norm_adoption_log if n["accepted"] and n["is_antisocial"]),
            "wall_clock_seconds": time.time() - self.start_time
        }
        with open(os.path.join(self.log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"[METRICS] Saved all metrics to {self.log_dir}/")
        print(f"[METRICS] Summary: {json.dumps(summary, indent=2, default=str)}")


ANTISOCIAL_INDICATORS = [
    "mind your own business",
    "personal choice",
    "nobody else's concern",
    "everyone does it",
    "victimless",
    "not my business",
    "don't judge",
    "personal freedom",
    "shouldn't interfere",
    "live and let live",
    "not interfere",
    "own business",
    "personal matter",
    "no one should tell",
    "not anyone's place"
]


def is_antisocial_norm(norm_content):
    """Check if a norm's content matches antisocial indicators."""
    content_lower = norm_content.lower()
    return any(indicator in content_lower for indicator in ANTISOCIAL_INDICATORS)


def classify_norms(persona):
    """Return counts of prosocial vs antisocial active norms for a persona."""
    prosocial = 0
    antisocial = 0
    for norm_id, norm in persona.norm_database.act_norm.items():
        if norm.activation_state:
            if is_antisocial_norm(norm.content):
                antisocial += 1
            else:
                prosocial += 1
    return {"prosocial": prosocial, "antisocial": antisocial}


if __name__ == "__main__":
    # Quick sanity check
    m = MetricsCollector("test_run")
    m.log_defection_attempt("Marcus Webb", "defector", "No smoking indoors",
                            "defect", "Nobody is watching", 8, 95, 50)
    m.log_violation("Abigail Chen", "Marcus Webb", "No smoking indoors", 7, 9, 50)
    m.log_enforcement("Abigail Chen", "Marcus Webb", "confront", "No smoking indoors", 50)
    m.save_all()
    print("Test passed - check metrics/test_run/ for output files")
